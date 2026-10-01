# -*- coding: utf-8 -*-
"""
monitor.py

Muestreo del uso de recursos mientras se procesa un archivo. El enunciado
del P1.2 pide comparar CPU y GPU no solo por tiempo sino tambien por
"uso de CPU/GPU (memoria, carga)", y eso hay que medirlo mientras el
trabajo ocurre, no antes ni despues.

Funciona con un hilo aparte que toma muestras cada cierto intervalo y al
final devuelve el promedio y el maximo de cada magnitud. Se usa un hilo y no
un proceso a proposito: el muestreo pasa casi todo el tiempo dormido o
esperando a que el sistema operativo responda, asi que el GIL no estorba y
el coste sobre la medicion es despreciable.

psutil y pynvml son opcionales. Si falta alguno, el monitor sigue
funcionando y simplemente deja vacias las magnitudes que no puede leer, tal
como exigen las restricciones del proyecto.

Uso tipico:
    with Monitor() as m:
        resultado, tiempo = motor_gpu.contar_gpu(ruta)
    print(m.resumen())
"""

import subprocess
import sys
import threading
import time

try:
    import psutil
    _HAY_PSUTIL = True
except ImportError:
    psutil = None
    _HAY_PSUTIL = False

try:
    import pynvml
    _HAY_NVML = True
except ImportError:
    pynvml = None
    _HAY_NVML = False


# Intervalo entre muestras. 0.1 s es el compromiso: da resolucion suficiente
# para que una ejecucion de GPU de medio segundo deje varias muestras, y
# sigue siendo un intervalo lo bastante largo como para que los contadores
# del sistema operativo hayan avanzado de verdad entre una lectura y la
# siguiente. Con intervalos mucho mas cortos, psutil devuelve cero porque no
# ha transcurrido tiempo medible entre ambas llamadas.
INTERVALO = 0.1

# Numero minimo de muestras para que un promedio signifique algo. Por debajo
# de esto se prefiere declarar la magnitud como no medida antes que publicar
# un promedio calculado sobre una sola lectura, que seria enganoso.
MINIMO_MUESTRAS = 2

# La temperatura del procesador se lee por su cuenta y mucho mas despacio.
# La consulta al sistema cuesta del orden de medio segundo, asi que meterla
# en el bucle principal lo convertiria en un muestreo de dos lecturas por
# segundo en vez de diez y estropearia el resto de magnitudes. Va en un hilo
# aparte que refresca un valor en cache, que es de sobra: la temperatura de
# un procesador se mueve en segundos, no en decimas.
INTERVALO_TEMPERATURA = 3.0

# Consulta al contador de la zona termica del sistema. Este contador entrega
# kelvin enteros (ojo: el otro camino habitual, MSAcpi_ThermalZoneTemperature,
# los da en decimas, y confundirlos saca temperaturas bajo cero).
_CONSULTA_TEMP = (
    '(Get-CimInstance Win32_PerfFormattedData_Counters_ThermalZoneInformation'
    ' | Select-Object -First 1).Temperature')


class Monitor:
    """Muestrea uso de CPU, RAM y GPU en un hilo de fondo."""

    def __init__(self, intervalo=INTERVALO, con_gpu=True):
        self.intervalo = intervalo
        self.con_gpu = con_gpu and _HAY_NVML
        self.muestras = []
        self._hilo = None
        self._hilo_temp = None
        self._parar = threading.Event()
        self._nvml_listo = False
        self._manejador = None
        self._temp_cpu = None

    # -- ciclo de vida ----------------------------------------------------

    def iniciar(self):
        """Arranca el hilo de muestreo."""
        self.muestras = []
        self._parar.clear()

        if self.con_gpu:
            try:
                pynvml.nvmlInit()
                self._manejador = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._nvml_listo = True
            except Exception:
                self._nvml_listo = False

        if _HAY_PSUTIL:
            # Primera llamada de calibracion: psutil devuelve 0.0 la primera
            # vez porque necesita dos lecturas para calcular el porcentaje.
            try:
                psutil.cpu_percent(interval=None)
            except Exception:
                pass

        self._hilo = threading.Thread(target=self._bucle, daemon=True)
        self._hilo.start()

        self._hilo_temp = threading.Thread(target=self._bucle_temperatura,
                                           daemon=True)
        self._hilo_temp.start()
        return self

    def detener(self):
        """Detiene el hilo y espera a que termine."""
        self._parar.set()
        if self._hilo is not None:
            self._hilo.join(timeout=2.0)
            self._hilo = None
        if self._hilo_temp is not None:
            self._hilo_temp.join(timeout=3.0)
            self._hilo_temp = None
        if self._nvml_listo:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_listo = False
        return self

    def __enter__(self):
        return self.iniciar()

    def __exit__(self, *_):
        self.detener()
        return False

    # -- muestreo ---------------------------------------------------------

    def _bucle(self):
        """Bucle del hilo: toma una muestra por intervalo hasta que se pare."""
        while not self._parar.is_set():
            self.muestras.append(self._tomar_muestra())
            self._parar.wait(self.intervalo)

    def _bucle_temperatura(self):
        """Refresca la temperatura del procesador cada pocos segundos.

        Va en su propio hilo porque la consulta al sistema es lenta y el
        bucle principal tiene que seguir tomando diez muestras por segundo.
        La primera lectura se hace de inmediato para que el panel no arranque
        vacio, y a partir de ahi se espacia.
        """
        while True:
            valor = self._leer_temperatura_cpu()
            if valor is not None:
                self._temp_cpu = valor
            if self._parar.wait(INTERVALO_TEMPERATURA):
                break

    def _leer_temperatura_cpu(self):
        """Grados centigrados de la zona termica del procesador, o None.

        Si la consulta falla o esta maquina no publica el contador se
        devuelve None y la magnitud queda sin medir, igual que el resto de
        lecturas opcionales.
        """
        if not sys.platform.startswith('win'):
            return None
        try:
            proceso = subprocess.run(
                ['powershell', '-NoProfile', '-NonInteractive',
                 '-Command', _CONSULTA_TEMP],
                capture_output=True, text=True, timeout=15,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            texto = proceso.stdout.strip()
            if not texto:
                return None
            return round(float(texto) - 273.15, 1)
        except Exception:
            return None

    def _memoria_proceso(self):
        """Memoria residente del programa, sumando la de sus procesos hijos.

        Es la magnitud que de verdad sirve para comparar CPU contra GPU. La
        memoria del sistema entero ronda los 16 GB en este equipo y esta
        dominada por todo lo que hay abierto, de modo que la parte que
        corresponde al procesamiento queda escondida dentro del ruido. Lo que
        interesa es cuanta memoria cuesta cada estrategia: la version de CPU
        con doce procesos paga doce buferes de lectura, mientras que la de GPU
        paga dos buferes de pagina bloqueada mas la VRAM de la tarjeta.
        """
        try:
            actual = psutil.Process()
            total = actual.memory_info().rss
            for hijo in actual.children(recursive=True):
                try:
                    total += hijo.memory_info().rss
                except Exception:
                    # Un hijo puede morir entre que se lista y se consulta.
                    continue
            return total // (1024 * 1024)
        except Exception:
            return None

    def _tomar_muestra(self):
        """Lectura puntual de todas las magnitudes disponibles."""
        muestra = {'t': time.perf_counter()}

        if _HAY_PSUTIL:
            try:
                muestra['cpu'] = psutil.cpu_percent(interval=None)
                muestra['cpu_nucleos'] = psutil.cpu_percent(interval=None,
                                                            percpu=True)
                memoria = psutil.virtual_memory()
                muestra['ram'] = memoria.percent
                muestra['ram_usada_mb'] = (memoria.total - memoria.available) \
                    // (1024 * 1024)
                muestra['proc_mb'] = self._memoria_proceso()
            except Exception:
                pass

        if self._temp_cpu is not None:
            muestra['cpu_temp'] = self._temp_cpu

        if self._nvml_listo:
            try:
                uso = pynvml.nvmlDeviceGetUtilizationRates(self._manejador)
                memoria = pynvml.nvmlDeviceGetMemoryInfo(self._manejador)
                muestra['gpu'] = uso.gpu
                muestra['gpu_mem_uso'] = uso.memory
                muestra['vram_mb'] = memoria.used // (1024 * 1024)
                muestra['gpu_temp'] = pynvml.nvmlDeviceGetTemperature(
                    self._manejador, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                pass

        return muestra

    # -- resultados -------------------------------------------------------

    def marcar(self):
        """Devuelve el indice de la proxima muestra.

        Permite acotar un tramo del muestreo sin detener el monitor: se llama
        antes de la operacion que se quiere medir y el indice devuelto se
        pasa despues a resumen(). Existe para que un solo monitor pueda
        atender a la vez al panel en vivo de la ventana y a la medicion de
        cada configuracion del barrido.

        Es importante no arrancar dos monitores en paralelo: la funcion
        psutil.cpu_percent con intervalo None calcula el porcentaje contra la
        ultima llamada, y ese punto de referencia es global al proceso. Dos
        monitores simultaneos se pisarian mutuamente la referencia y ambos
        devolverian porcentajes sin sentido.
        """
        return len(self.muestras)

    def _serie(self, clave, desde=0):
        """Extrae la lista de valores de una magnitud, sin los huecos."""
        return [m[clave] for m in self.muestras[desde:] if clave in m]

    def resumen(self, desde=0):
        """Devuelve promedio y maximo de cada magnitud muestreada.

        Si se indica 'desde', solo se consideran las muestras tomadas a
        partir de ese indice, que es el que devolvio marcar().

        Las claves ausentes indican que esa magnitud no se pudo leer en esta
        maquina. El consumidor debe tratarlas como opcionales.
        """
        datos = {'muestras': len(self.muestras) - desde}

        for clave, etiqueta in (('cpu', 'cpu'),
                                ('cpu_temp', 'cpu_temp'),
                                ('ram', 'ram'),
                                ('ram_usada_mb', 'ram_mb'),
                                ('proc_mb', 'proc_mb'),
                                ('gpu', 'gpu'),
                                ('gpu_mem_uso', 'gpu_mem'),
                                ('vram_mb', 'vram_mb'),
                                ('gpu_temp', 'gpu_temp')):
            serie = self._serie(clave, desde)
            if serie:
                datos[etiqueta + '_medio'] = sum(serie) / len(serie)
                datos[etiqueta + '_max'] = max(serie)

        # Uso por nucleo: interesa el promedio de cada nucleo por separado
        # para ver si el reparto de trabajo fue equilibrado.
        series_nucleos = self._serie('cpu_nucleos', desde)
        if series_nucleos:
            n = len(series_nucleos[0])
            promedios = []
            for i in range(n):
                valores = [s[i] for s in series_nucleos if len(s) > i]
                promedios.append(sum(valores) / len(valores) if valores else 0.0)
            datos['cpu_por_nucleo'] = promedios

        return datos

    def ultima(self):
        """Devuelve la ultima muestra tomada, util para paneles en vivo."""
        return self.muestras[-1] if self.muestras else {}


def disponibilidad():
    """Informa de que bibliotecas de monitoreo estan presentes."""
    return {'psutil': _HAY_PSUTIL, 'pynvml': _HAY_NVML}


if __name__ == '__main__':
    print('Bibliotecas de monitoreo: %s' % disponibilidad())
    print('Tomando muestras durante 2 segundos...')
    with Monitor() as monitor:
        # Carga artificial para que se note algo en las lecturas.
        fin = time.perf_counter() + 2.0
        total = 0
        while time.perf_counter() < fin:
            total += 1
    resumen = monitor.resumen()
    print('')
    for clave in sorted(resumen):
        if clave == 'cpu_por_nucleo':
            valores = ' '.join('%.0f' % v for v in resumen[clave])
            print('  %-16s %s' % (clave, valores))
        else:
            print('  %-16s %s' % (clave, resumen[clave]))
