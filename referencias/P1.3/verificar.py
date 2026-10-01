# -*- coding: utf-8 -*-
"""
verificar.py

Prueba de correccion de los motores del P1.3. Antes de medir rendimiento hay
que demostrar que cuentan bien, porque un motor rapido que cuenta mal no
sirve de nada.

Lo que de verdad se pone a prueba aqui es el reparto del motor hibrido. Que
la CPU sola cuente bien y que la GPU sola cuente bien no garantiza que el
hibrido lo haga: si los bordes entre trozos no encajan al milimetro, un
nucleotido puede contarse dos veces o perderse justo en la costura entre un
trozo que atendio la CPU y el siguiente que atendio la GPU. Por eso la
comparacion es sobre el histograma completo de 256 bins, que es mucho mas
exigente que mirar solo A, C, G y T.

NOTA SOBRE WINDOWS
Este archivo tiene que ejecutarse como script real, no por la entrada
estandar. En Windows multiprocessing usa el metodo 'spawn': cada proceso
hijo vuelve a importar el modulo principal, asi que necesita un archivo en
disco y el guard if __name__ == '__main__'.

Uso:
    python verificar.py <archivo.fna> [procesos] [trozo_mb]
"""

import multiprocessing
import os
import sys

import motor_cpu
import motor_hibrido

# motor_gpu NO se importa aqui a proposito: los procesos hijos reimportan
# este modulo y cargarian el stack de CUDA para no usarlo. Se importa dentro
# de main(), que los hijos nunca ejecutan.


def mostrar(nombre, resultado, tiempo, tamano):
    """Imprime una linea de resumen por motor."""
    mbs = (tamano / (1024 * 1024) / tiempo) if tiempo > 0 else 0.0
    print('  %-20s %8.3f s  %8.1f MB/s   A=%d C=%d G=%d T=%d N=%d '
          'iupac=%d inval=%d'
          % (nombre, tiempo, mbs, resultado['A'], resultado['C'],
             resultado['G'], resultado['T'], resultado['N'],
             resultado['ambiguos'], resultado['invalidos']))


def main():
    if len(sys.argv) < 2:
        print('Uso: python verificar.py <archivo.fna> [procesos] [trozo_mb]')
        return 1

    ruta = sys.argv[1]
    if not os.path.exists(ruta):
        print('No existe el archivo: %s' % ruta)
        return 1

    # Import tardio: ver la nota de la cabecera del modulo.
    import motor_gpu

    fisicos, logicos = motor_cpu.detectar_nucleos()
    procesos = int(sys.argv[2]) if len(sys.argv) > 2 else logicos
    trozo = int(sys.argv[3]) if len(sys.argv) > 3 else motor_hibrido.TROZO_MB

    tamano = os.path.getsize(ruta)
    print('Archivo : %s  (%.1f MB)' % (ruta, tamano / (1024 * 1024)))
    print('CPU     : %d nucleos fisicos / %d logicos' % (fisicos, logicos))

    datos_gpu = motor_gpu.info_gpu()
    hay_gpu = datos_gpu['disponible']
    if hay_gpu:
        print('GPU     : %s (CC %s, %d SMs, %d MB VRAM)'
              % (datos_gpu['nombre'], datos_gpu['compute_capability'],
                 datos_gpu['sms'], datos_gpu['vram_total_mb']))
        print('          compilacion JIT: %.3f s (fuera de la medicion)'
              % motor_gpu.precalentar())
    else:
        print('GPU     : no disponible, se omite todo lo de GPU')

    print('')
    print('Resultados:')

    resultados = {}

    res, tiempo = motor_cpu.contar_secuencial(ruta)
    resultados['CPU secuencial'] = res
    mostrar('CPU secuencial', res, tiempo, tamano)
    referencia = res

    res, tiempo = motor_cpu.contar_paralelo(ruta, procesos)
    resultados['CPU paralelo x%d' % procesos] = res
    mostrar('CPU paralelo x%d' % procesos, res, tiempo, tamano)

    if hay_gpu:
        res, tiempo = motor_hibrido.contar_hibrido(ruta, trozo_mb=trozo,
                                                   usar_cpu=False)
        resultados['GPU sola'] = res
        mostrar('GPU sola', res, tiempo, tamano)

        # La prueba que de verdad importa: varias configuraciones de reparto,
        # para que las costuras entre trozos caigan en sitios distintos cada
        # vez. Si hubiera un error de borde, basta con que una sola de estas
        # combinaciones lo destape.
        for n in (2, procesos):
            for grano in (trozo, trozo * 2):
                etiqueta = 'Hibrido x%d t%d' % (n, grano)
                res, tiempo = motor_hibrido.contar_hibrido(
                    ruta, procesos=n, trozo_mb=grano)
                resultados[etiqueta] = res
                mostrar(etiqueta, res, tiempo, tamano)
                datos = res['hibrido']
                total = datos['bytes_cpu'] + datos['bytes_gpu']
                print('       reparto CPU %.0f%% / GPU %.0f%%, '
                      'a la vez %.2f s'
                      % (100.0 * datos['bytes_cpu'] / total,
                         100.0 * datos['bytes_gpu'] / total,
                         datos['solape_s']))

    print('')
    print('Consistencia entre motores (254 bins, sin contar saltos de linea):')
    todo_bien = True
    patron = motor_cpu.bins_significativos(referencia['histograma'])
    for nombre, res in resultados.items():
        igual = motor_cpu.bins_significativos(res['histograma']) == patron
        print('  %-20s %s' % (nombre, 'identico' if igual else 'DIFIERE'))
        if not igual:
            todo_bien = False
            # Se detalla en que bytes esta la discrepancia, para poder
            # depurar sin tener que volver a lanzar todo el proceso.
            for i in range(motor_cpu.BINS):
                if i in motor_cpu.IGNORADOS:
                    continue
                mio = res['histograma'][i]
                suyo = referencia['histograma'][i]
                if mio != suyo:
                    print('      byte %3d (%-6s): %d frente a %d'
                          % (i, motor_cpu.nombre_byte(i), mio, suyo))

    # El bin del salto de linea se informa aparte porque se espera que
    # difiera, y no es un error. Ver motor_cpu.bins_significativos.
    saltos = {n: r['histograma'][10] for n, r in resultados.items()}
    if len(set(saltos.values())) > 1:
        print('')
        print('  Nota: los saltos de linea difieren entre motores, y es lo')
        print('  esperado. La GPU sobreescribe las cabeceras con saltos de')
        print('  linea en vez de eliminarlas. No afecta a ningun conteo.')

    if referencia['detalle_iupac']:
        print('')
        print('Codigos IUPAC de ambiguedad (validos, NO son errores):')
        significados = {
            'R': 'A o G (purina)', 'Y': 'C o T (pirimidina)',
            'S': 'G o C (enlace fuerte)', 'W': 'A o T (enlace debil)',
            'K': 'G o T (ceto)', 'M': 'A o C (amino)',
            'B': 'C, G o T (no A)', 'D': 'A, G o T (no C)',
            'H': 'A, C o T (no G)', 'V': 'A, C o G (no T)',
        }
        for caracter, veces in sorted(referencia['detalle_iupac'].items(),
                                      key=lambda x: -x[1]):
            print('    %-3s %8d   %s'
                  % (caracter, veces, significados.get(caracter, '')))

    if referencia['detalle']:
        print('')
        print('Caracteres INVALIDOS encontrados (%d distintos, %d en total):'
              % (len(referencia['detalle']), referencia['invalidos']))
        for caracter, veces in sorted(referencia['detalle'].items(),
                                      key=lambda x: -x[1]):
            print('    %-6s %d' % (repr(caracter), veces))

    print('')
    print('RESULTADO GLOBAL: %s' % ('TODO CORRECTO' if todo_bien
                                    else 'HAY DISCREPANCIAS'))
    return 0 if todo_bien else 2


if __name__ == '__main__':
    multiprocessing.freeze_support()
    sys.exit(main())
