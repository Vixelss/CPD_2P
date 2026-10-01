# -*- coding: utf-8 -*-
"""
interfaz.py

Ventana de la aplicacion del P1.3: CPU y GPU trabajando en paralelo sobre el
mismo archivo de ADN.

Se arranca desde app_p13.py, nunca directamente, porque ese archivo es el que
pone el guard de multiprocessing que Windows exige.

ESTRUCTURA
  - Columna izquierda: controles, hardware detectado y conteos.
  - Columna derecha : pestanas con las graficas, el detalle por nucleo, la
                      matriz de evidencias, la tabla y el registro.
  - Pie             : barra de progreso y lecturas de recursos en vivo.

CONCURRENCIA
Todo el procesamiento ocurre en un hilo aparte. Tkinter no es seguro para
usarse desde varios hilos, asi que el hilo trabajador nunca toca un widget:
deja mensajes en una cola y la ventana la vacia cada 100 ms desde el hilo
principal con after(). Es el unico patron correcto aqui, y ademas es lo que
mantiene la ventana viva mientras se procesan los 3 GB del genoma.
"""

import os
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import benchmark
import evidencias
import monitor as mod_monitor
import motor_cpu
import motor_hibrido


# Paleta sobria, pensada para que las graficas se lean igual impresas en el
# informe en blanco y negro.
COLOR_CPU = '#2f6f9f'
COLOR_CPU2 = '#7fb3d5'
COLOR_GPU = '#76b041'
COLOR_HIB = '#c07a2a'
COLOR_AVISO = '#b03030'
COLOR_FONDO = '#f4f4f4'

INTERVALO_COLA = 100        # ms entre revisiones de la cola de mensajes
INTERVALO_RECURSOS = 500    # ms entre refrescos del panel de recursos

COLOR_PLATAFORMA = {'CPU': COLOR_CPU, 'GPU': COLOR_GPU, 'HIBRIDO': COLOR_HIB}


def _corto(nombre):
    """Acorta el nombre comercial de una tarjeta para que quepa en el panel."""
    limpio = nombre.replace('NVIDIA GeForce ', '').replace('(R)', '')
    return limpio.replace(' Laptop GPU', '').replace('Graphics', 'Gfx').strip()


class Aplicacion:
    """Ventana principal de la aplicacion."""

    def __init__(self, raiz):
        self.raiz = raiz
        self.raiz.title('P1.3 - Computacion CPU & GPU Paralela | '
                        'CPU y GPU trabajando a la vez')

        # La ventana se adapta a la pantalla en lugar de imponer un tamano
        # fijo. En el portatil donde se desarrollo esto la pantalla util son
        # 1536x864 menos la barra de tareas, asi que una ventana de mas de
        # 800 px de alto deja la barra de progreso fuera del borde inferior.
        ancho_pantalla = self.raiz.winfo_screenwidth()
        alto_pantalla = self.raiz.winfo_screenheight()
        self.ancho = min(1320, ancho_pantalla - 40)
        self.alto = min(800, alto_pantalla - 110)
        self.raiz.geometry('%dx%d+%d+%d'
                           % (self.ancho, self.alto,
                              max(0, (ancho_pantalla - self.ancho) // 2), 10))
        self.raiz.minsize(940, 620)

        # Estado compartido entre el hilo trabajador y la ventana.
        self.cola = queue.Queue()
        self.hilo = None
        self.trabajando = False
        self.monitor = None
        self.inicio_proceso = None

        self.ruta = tk.StringVar(value='')
        self.modo = tk.StringVar(value='hibrido')
        self.procesos = tk.IntVar(value=0)
        self.trozo_mb = tk.IntVar(value=motor_hibrido.TROZO_MB)
        self.estado = tk.StringVar(value='Listo. Selecciona un archivo FASTA.')
        self.velocidad = tk.StringVar(value='')

        # Resultados acumulados, para las graficas y el export a CSV.
        self.filas = []
        self.ultimo_resultado = None
        self.ultimo_censo = None
        self.ultimo_registro = []
        self.ultimos_recursos = {}

        # Fotografia de los controles tomada al pulsar un boton, para que el
        # hilo trabajador no tenga que leer variables de Tkinter.
        self.config = {'modo': 'hibrido', 'procesos': 1, 'trozo_mb': 16}

        # Se pone a True al cerrar la ventana. Los bucles que se reprograman
        # con after lo consultan antes de volver a encolarse: sin esto, una
        # llamada pendiente puede dispararse cuando el interprete de Tcl ya
        # no existe y provocar un error al salir.
        self.cerrando = False

        self._detectar_hardware()
        self._construir()
        self.raiz.protocol('WM_DELETE_WINDOW', self.cerrar)

        # Deja las graficas con su aviso de "sin datos" en lugar de unos ejes
        # vacios con la escala por defecto de 0 a 1.
        self._dibujar_graficas()
        self._dibujar_nucleos()
        self._rellenar_evidencias()

        # Un monitor vivo desde el arranque: alimenta el pie de la ventana y
        # se le presta despues a cada medicion del barrido.
        self.monitor = mod_monitor.Monitor()
        self.monitor.iniciar()

        self._revisar_cola()
        self._refrescar_recursos()

    # ------------------------------------------------------------------
    # Hardware
    # ------------------------------------------------------------------

    def _detectar_hardware(self):
        """Lee las caracteristicas de CPU y GPU una sola vez al arrancar."""
        self.fisicos, self.logicos = motor_cpu.detectar_nucleos()
        self.procesos.set(max(1, self.logicos // 2))

        # Import tardio del motor de GPU: hasta aqui la ventana puede
        # arrancar aunque numba no este instalado.
        try:
            import motor_gpu
            self.motor_gpu = motor_gpu
            self.datos_gpu = motor_gpu.info_gpu()
        except Exception as error:
            self.motor_gpu = None
            self.datos_gpu = {'disponible': False, 'error': str(error)}

        self.hay_gpu = self.datos_gpu.get('disponible', False)
        if not self.hay_gpu:
            self.modo.set('cpu')

        self.recursos_kernel = None

    # ------------------------------------------------------------------
    # Construccion de la interfaz
    # ------------------------------------------------------------------

    def _construir(self):
        self.raiz.configure(bg=COLOR_FONDO)

        # El pie se empaqueta antes que el contenedor principal a proposito.
        # Tk reparte el espacio en el orden en que se empaqueta, de modo que
        # si el contenedor va primero con expand=True se queda con todo el
        # alto y al pie no le sobra ni un pixel: la barra de progreso y las
        # lecturas de recursos desaparecen, y encima las graficas se dibujan
        # sobre un lienzo mas alto que la zona visible y quedan cortadas por
        # el borde inferior de la ventana.
        self._panel_pie(self.raiz)

        contenedor = ttk.Frame(self.raiz, padding=8)
        contenedor.pack(fill='both', expand=True)

        izquierda = ttk.Frame(contenedor, width=395)
        izquierda.pack(side='left', fill='y', padx=(0, 8))
        izquierda.pack_propagate(False)

        derecha = ttk.Frame(contenedor)
        derecha.pack(side='left', fill='both', expand=True)

        self._panel_archivo(izquierda)
        self._panel_configuracion(izquierda)
        self._panel_hardware(izquierda)
        self._panel_conteo(izquierda)
        self._panel_pestanas(derecha)

    def _panel_archivo(self, padre):
        marco = ttk.LabelFrame(padre, text=' Archivo ', padding=8)
        marco.pack(fill='x', pady=(0, 6))

        ttk.Entry(marco, textvariable=self.ruta).pack(fill='x', pady=(0, 6))

        fila = ttk.Frame(marco)
        fila.pack(fill='x')
        ttk.Button(fila, text='Seleccionar FASTA...',
                   command=self.elegir_archivo).pack(side='left')
        self.etiqueta_tamano = ttk.Label(fila, text='')
        self.etiqueta_tamano.pack(side='left', padx=8)

    def _panel_configuracion(self, padre):
        marco = ttk.LabelFrame(padre, text=' Configuracion ', padding=8)
        marco.pack(fill='x', pady=(0, 6))

        fila = ttk.Frame(marco)
        fila.pack(fill='x', pady=2)
        ttk.Label(fila, text='Modo:').pack(side='left')
        for texto, valor in (('CPU', 'cpu'), ('GPU', 'gpu'),
                             ('Las dos a la vez', 'hibrido')):
            boton = ttk.Radiobutton(fila, text=texto, value=valor,
                                    variable=self.modo)
            boton.pack(side='left', padx=3)
            if valor in ('gpu', 'hibrido') and not self.hay_gpu:
                boton.state(['disabled'])

        fila = ttk.Frame(marco)
        fila.pack(fill='x', pady=2)
        ttk.Label(fila, text='Procesos CPU:').pack(side='left')
        ttk.Spinbox(fila, from_=1, to=max(32, self.logicos), width=5,
                    textvariable=self.procesos).pack(side='left', padx=6)
        ttk.Label(fila, text='(logicos: %d)' % self.logicos).pack(side='left')

        fila = ttk.Frame(marco)
        fila.pack(fill='x', pady=2)
        ttk.Label(fila, text='Trozo del reparto (MB):').pack(side='left')
        combo = ttk.Combobox(fila, width=5, state='readonly',
                             textvariable=self.trozo_mb,
                             values=list(benchmark.TROZOS_POR_DEFECTO))
        combo.pack(side='left', padx=6)

        botones = ttk.Frame(marco)
        botones.pack(fill='x', pady=(8, 0))
        self.boton_procesar = ttk.Button(botones, text='Procesar',
                                         command=self.procesar_una)
        self.boton_procesar.pack(side='left')
        self.boton_analisis = ttk.Button(botones, text='Analisis completo',
                                         command=self.analisis_completo)
        self.boton_analisis.pack(side='left', padx=6)
        self.boton_csv = ttk.Button(botones, text='Exportar',
                                    command=self.exportar_csv)
        self.boton_csv.pack(side='left')

    def _panel_hardware(self, padre):
        marco = ttk.LabelFrame(padre, text=' Hardware detectado ', padding=8)
        marco.pack(fill='x', pady=(0, 6))

        ttk.Label(marco, text='CPU: %d nucleos fisicos / %d logicos'
                  % (self.fisicos, self.logicos)).pack(anchor='w')

        if self.hay_gpu:
            d = self.datos_gpu
            for linea in (
                'GPU: %s' % _corto(d['nombre']),
                '     Compute capability %s, %d SMs, warp %d'
                % (d['compute_capability'], d['sms'], d['warp']),
                '     VRAM %d MB (%d MB libres)'
                % (d['vram_total_mb'], d['vram_libre_mb']),
            ):
                ttk.Label(marco, text=linea).pack(anchor='w')
        else:
            ttk.Label(marco, text='GPU: no disponible (solo modo CPU)',
                      foreground=COLOR_AVISO).pack(anchor='w')

        disp = mod_monitor.disponibilidad()
        ttk.Label(marco, text='Monitoreo: psutil %s, pynvml %s'
                  % ('si' if disp['psutil'] else 'no',
                     'si' if disp['pynvml'] else 'no')).pack(anchor='w')

    def _panel_conteo(self, padre):
        marco = ttk.LabelFrame(padre, text=' Conteo ', padding=8)
        marco.pack(fill='x', pady=(0, 6))

        # Dos columnas en vez de una lista de ocho filas: a la izquierda las
        # cuatro bases y su total, a la derecha las tres categorias que no
        # son base. Ahorra la mitad de alto, que es justo lo que hace falta
        # para que el reparto no quede fuera del borde inferior.
        self.etiquetas_conteo = {}
        rejilla = ttk.Frame(marco)
        rejilla.pack(fill='x')

        columnas = (
            (('A', 'Adenina  (A)'), ('C', 'Citosina (C)'),
             ('G', 'Guanina  (G)'), ('T', 'Timina   (T)'),
             ('bases', 'Total bases')),
            (('N', 'N desc.'), ('ambiguos', 'IUPAC'),
             ('invalidos', 'Invalidos')),
        )

        for indice, columna in enumerate(columnas):
            base = indice * 2
            for fila_n, (clave, etiqueta) in enumerate(columna):
                ttk.Label(rejilla, text=etiqueta).grid(
                    row=fila_n, column=base, sticky='w', padx=(0, 3))
                # El ancho de la casilla del numero se ajusta a los conteos
                # del genoma completo, que llegan a diez digitos con sus
                # separadores de miles. Con una casilla mas estrecha la
                # segunda columna se sale del panel y queda recortada.
                valor = ttk.Label(rejilla, text='-', anchor='e', width=12)
                valor.grid(row=fila_n, column=base + 1, sticky='e',
                           padx=(0, 6))
                self.etiquetas_conteo[clave] = valor

        ttk.Separator(marco, orient='horizontal').pack(fill='x', pady=5)
        ttk.Label(marco, text='Reparto de la ultima corrida:').pack(anchor='w')
        self.texto_reparto = tk.Text(marco, height=5, width=38,
                                     font=('Consolas', 8), wrap='none')
        self.texto_reparto.pack(fill='x', pady=(2, 0))
        self.texto_reparto.configure(state='disabled')

    def _panel_recursos(self, padre):
        """Construye el panel de recursos como una tira horizontal en el pie.

        El pie de la ventana mide lo mismo que la ventana entera y estaria
        practicamente vacio, asi que las lecturas caben de sobra en una fila.
        En la columna izquierda no cabrian: Tk no tiene espacio que repartir
        y comprime los ultimos widgets en vez de desbordarlos.
        """
        self.etiquetas_recursos = {}
        for clave, etiqueta in (('cpu', 'CPU'), ('tcpu', 'T.CPU'),
                                ('ram', 'RAM'), ('gpu', 'GPU'),
                                ('vram', 'VRAM'), ('tgpu', 'T.GPU')):
            grupo = ttk.Frame(padre)
            grupo.pack(side='left', padx=(0, 8))
            ttk.Label(grupo, text=etiqueta).pack(side='left', padx=(0, 3))
            barra = ttk.Progressbar(grupo, length=58, maximum=100)
            barra.pack(side='left')
            valor = ttk.Label(grupo, text='-', width=7)
            valor.pack(side='left', padx=(3, 0))
            self.etiquetas_recursos[clave] = (barra, valor)

    def _panel_pestanas(self, padre):
        self.pestanas = ttk.Notebook(padre)
        self.pestanas.pack(fill='both', expand=True)

        # El tamano de la figura se calcula a partir del espacio que queda de
        # verdad a la derecha del panel de controles. Si se fija a ojo, el
        # lienzo pide mas espacio del que hay y Tk recorta por el borde.
        ancho_figura = max(6.0, (self.ancho - 400) / 100.0)
        alto_figura = max(4.2, (self.alto - 140) / 100.0)

        # -- graficas de ejecucion --
        marco_graficas = ttk.Frame(self.pestanas)
        self.pestanas.add(marco_graficas, text='  Ejecucion  ')
        self.figura = Figure(figsize=(ancho_figura, alto_figura), dpi=100)
        self.figura.subplots_adjust(hspace=0.55, wspace=0.28,
                                    left=0.09, right=0.97,
                                    top=0.92, bottom=0.14)
        self.ejes = [self.figura.add_subplot(2, 2, i + 1) for i in range(4)]
        self.lienzo = FigureCanvasTkAgg(self.figura, master=marco_graficas)
        self.lienzo.get_tk_widget().pack(fill='both', expand=True)

        # -- detalle por nucleo --
        # Es la pestana que responde al nivel mas exigente de la rubrica: no
        # el uso global de cada plataforma sino el de cada nucleo logico del
        # procesador y el de cada multiprocesador de la tarjeta.
        marco_nucleos = ttk.Frame(self.pestanas)
        self.pestanas.add(marco_nucleos, text='  Nucleos  ')
        self.figura_nuc = Figure(figsize=(ancho_figura, alto_figura), dpi=100)
        self.figura_nuc.subplots_adjust(hspace=0.45, left=0.09, right=0.97,
                                        top=0.92, bottom=0.12)
        self.ejes_nuc = [self.figura_nuc.add_subplot(2, 1, i + 1)
                         for i in range(2)]
        self.lienzo_nuc = FigureCanvasTkAgg(self.figura_nuc,
                                            master=marco_nucleos)
        self.lienzo_nuc.get_tk_widget().pack(fill='both', expand=True)

        # -- matriz de evidencias --
        # Va en un cuadro de texto monoespaciado y no en una tabla de widgets
        # porque las celdas son parrafos, no cifras: una tabla les recorta el
        # texto por el borde de la columna y la evidencia deja de leerse. Asi
        # se muestra exactamente la misma matriz que se exporta al informe.
        marco_evid = ttk.Frame(self.pestanas)
        self.pestanas.add(marco_evid, text='  Evidencias  ')
        self.texto_evid = tk.Text(marco_evid, font=('Consolas', 8),
                                  wrap='none', padx=8, pady=6)
        self.texto_evid.pack(side='left', fill='both', expand=True)
        barra = ttk.Scrollbar(marco_evid, orient='vertical',
                              command=self.texto_evid.yview)
        barra.pack(side='right', fill='y')
        self.texto_evid.configure(yscrollcommand=barra.set, state='disabled')

        # -- tabla comparativa --
        marco_tabla = ttk.Frame(self.pestanas)
        self.pestanas.add(marco_tabla, text='  Tabla  ')
        columnas = ('etiqueta', 'plataforma', 'tiempo_s', 'mb_por_s',
                    'speedup', 'reparto', 'solape_s', 'cpu_medio',
                    'gpu_medio', 'cpu_temp_medio', 'gpu_temp_max',
                    'vram_mb_max', 'conteo_ok')
        titulos = ('Configuracion', 'Plataforma', 'Tiempo (s)', 'MB/s',
                   'Speedup', 'Reparto C/G', 'Solape (s)', 'CPU %', 'GPU %',
                   'T.CPU C', 'T.GPU C', 'VRAM MB', 'Conteo OK')
        anchos = (120, 78, 72, 68, 62, 90, 68, 52, 52, 58, 58, 66, 72)
        self.tabla = ttk.Treeview(marco_tabla, columns=columnas,
                                  show='headings', height=22)
        for columna, titulo, ancho in zip(columnas, titulos, anchos):
            self.tabla.heading(columna, text=titulo)
            self.tabla.column(columna, width=ancho, anchor='center')
        self.tabla.pack(side='left', fill='both', expand=True)
        barra = ttk.Scrollbar(marco_tabla, orient='vertical',
                              command=self.tabla.yview)
        barra.pack(side='right', fill='y')
        self.tabla.configure(yscrollcommand=barra.set)

        # -- registro --
        marco_log = ttk.Frame(self.pestanas)
        self.pestanas.add(marco_log, text='  Registro  ')
        self.texto_log = tk.Text(marco_log, font=('Consolas', 9), wrap='none')
        self.texto_log.pack(fill='both', expand=True)

    def _panel_pie(self, padre):
        marco = ttk.Frame(padre, padding=(10, 4))
        marco.pack(fill='x', side='bottom')

        self.barra = ttk.Progressbar(marco, length=190, maximum=100)
        self.barra.pack(side='left')
        ttk.Label(marco, textvariable=self.velocidad,
                  width=11).pack(side='left', padx=5)

        recursos = ttk.Frame(marco)
        recursos.pack(side='right')
        self._panel_recursos(recursos)

        ttk.Label(marco, textvariable=self.estado).pack(side='left', padx=5)

    # ------------------------------------------------------------------
    # Graficas
    # ------------------------------------------------------------------

    def _marcar_vacio(self, eje, texto='Sin datos todavia'):
        """Deja un aviso en un eje que todavia no tiene nada que mostrar.

        Sin esto, un eje vacio se dibuja con la escala por defecto de 0 a 1 y
        parece un grafico con datos reales pero planos, que es peor que no
        mostrar nada.
        """
        eje.text(0.5, 0.5, texto, ha='center', va='center',
                 transform=eje.transAxes, color='#999999', fontsize=9)
        eje.set_xticks([])
        eje.set_yticks([])

    def _dibujar_graficas(self):
        """Redibuja las cuatro graficas de ejecucion."""
        for eje in self.ejes:
            eje.clear()

        directas = [f for f in self.filas if not f.get('escala')]

        # 1) Tiempo por configuracion.
        eje = self.ejes[0]
        if directas:
            etiquetas = [f['etiqueta'] for f in directas]
            tiempos = [f['tiempo_s'] for f in directas]
            colores = [COLOR_PLATAFORMA.get(f['plataforma'], COLOR_CPU)
                       for f in directas]
            barras = eje.bar(range(len(directas)), tiempos, color=colores)
            eje.set_xticks(range(len(directas)))
            eje.set_xticklabels(etiquetas, rotation=45, ha='right', fontsize=7)
            eje.set_ylabel('segundos', fontsize=8)
            for barra, tiempo in zip(barras, tiempos):
                eje.text(barra.get_x() + barra.get_width() / 2,
                         barra.get_height(), '%.2f' % tiempo,
                         ha='center', va='bottom', fontsize=6)
        else:
            self._marcar_vacio(eje)
        eje.set_title('Tiempo por configuracion', fontsize=10)
        eje.grid(axis='y', alpha=0.3)

        # 2) Reparto del trabajo entre las dos plataformas.
        eje = self.ejes[1]
        datos = [f for f in directas
                 if f.get('reparto_cpu_pct') not in ('', None)]
        if datos:
            posiciones = range(len(datos))
            cpu = [f['reparto_cpu_pct'] for f in datos]
            gpu = [f['reparto_gpu_pct'] for f in datos]
            eje.bar(posiciones, cpu, color=COLOR_CPU, label='CPU')
            eje.bar(posiciones, gpu, bottom=cpu, color=COLOR_GPU, label='GPU')
            eje.set_xticks(list(posiciones))
            eje.set_xticklabels([f['etiqueta'] for f in datos],
                                rotation=45, ha='right', fontsize=7)
            eje.set_ylabel('% del archivo', fontsize=8)
            eje.set_ylim(0, 100)
            eje.legend(fontsize=7, loc='lower right')
        else:
            self._marcar_vacio(eje, 'Se llena al procesar en modo hibrido')
        eje.set_title('Reparto del trabajo', fontsize=10)
        eje.grid(axis='y', alpha=0.3)

        # 3) Linea de tiempo: quien estuvo trabajando y cuando. Es la prueba
        #    directa de que las dos plataformas se solaparon; si el motor
        #    alternase en vez de trabajar en paralelo, las bandas de CPU y de
        #    GPU no compartirian ninguna franja vertical.
        eje = self.ejes[2]
        if self.ultimo_registro:
            trabajadores = []
            for apunte in self.ultimo_registro:
                if apunte['trabajador'] not in trabajadores:
                    trabajadores.append(apunte['trabajador'])
            # La GPU arriba del todo, para que su banda destaque.
            trabajadores.sort(key=lambda t: (t != 'GPU', t))
            posicion = {t: i for i, t in enumerate(trabajadores)}

            for apunte in self.ultimo_registro:
                y = posicion[apunte['trabajador']]
                eje.barh(y, apunte['fin_s'] - apunte['inicio_s'],
                         left=apunte['inicio_s'], height=0.65,
                         color=COLOR_PLATAFORMA.get(apunte['plataforma'],
                                                    COLOR_CPU),
                         edgecolor='white', linewidth=0.4)
            eje.set_yticks(range(len(trabajadores)))
            eje.set_yticklabels(trabajadores, fontsize=7)
            eje.set_xlabel('segundos desde el arranque', fontsize=8)
            eje.invert_yaxis()
        else:
            self._marcar_vacio(eje, 'Se llena al procesar en modo hibrido')
        eje.set_title('Linea de tiempo del reparto', fontsize=10)
        eje.grid(axis='x', alpha=0.3)

        # 4) Escalabilidad por tamano.
        eje = self.ejes[3]
        series = {}
        for fila in self.filas:
            if not fila.get('escala'):
                continue
            series.setdefault(fila['plataforma'], []).append(
                (fila['bytes'] / (1024 * 1024), fila['tiempo_s']))
        if series:
            for nombre, puntos in sorted(series.items()):
                puntos.sort()
                eje.plot([p[0] for p in puntos], [p[1] for p in puntos],
                         marker='o', label=nombre,
                         color=COLOR_PLATAFORMA.get(nombre, COLOR_CPU2))
            eje.set_xlabel('tamano del archivo (MB)', fontsize=8)
            eje.set_ylabel('segundos', fontsize=8)
            eje.legend(fontsize=7)
        else:
            self._marcar_vacio(eje, 'Se llena con "Analisis completo"')
        eje.set_title('Escalabilidad: tiempo por tamano', fontsize=10)
        eje.grid(alpha=0.3)

        self.lienzo.draw()

    def _dibujar_nucleos(self):
        """Dibuja el uso de cada nucleo logico y de cada multiprocesador."""
        for eje in self.ejes_nuc:
            eje.clear()

        # 1) Uso medio de cada nucleo logico del procesador.
        eje = self.ejes_nuc[0]
        nucleos = self.ultimos_recursos.get('cpu_por_nucleo')
        if nucleos:
            posiciones = range(len(nucleos))
            eje.bar(posiciones, nucleos, color=COLOR_CPU)
            eje.axhline(sum(nucleos) / len(nucleos), color='#888888',
                        linestyle='--', linewidth=0.8)
            eje.set_xticks(list(posiciones))
            eje.set_xticklabels(['%d' % i for i in posiciones], fontsize=7)
            eje.set_xlabel('nucleo logico', fontsize=8)
            eje.set_ylabel('uso medio %', fontsize=8)
            eje.set_ylim(0, 100)
        else:
            self._marcar_vacio(eje, 'Se llena al procesar')
        eje.set_title('Uso por nucleo logico de CPU', fontsize=10)
        eje.grid(axis='y', alpha=0.3)

        # 2) Trabajo que se llevo cada multiprocesador de la tarjeta. El dato
        #    no viene de una estimacion: lo contaron los propios hilos dentro
        #    de cada SM leyendo el registro %smid.
        eje = self.ejes_nuc[1]
        censo = self.ultimo_censo
        if censo:
            porcentajes = evidencias.censo_a_porcentajes(censo)
            posiciones = range(len(porcentajes))
            eje.bar(posiciones, porcentajes, color=COLOR_GPU)
            media = sum(porcentajes) / len(porcentajes)
            eje.axhline(media, color='#888888', linestyle='--', linewidth=0.8)
            eje.set_xticks(list(posiciones))
            eje.set_xticklabels(['%d' % i for i in posiciones], fontsize=7)
            eje.set_xlabel('multiprocesador (SM)', fontsize=8)
            eje.set_ylabel('% de los bytes', fontsize=8)
            eje.set_ylim(0, max(porcentajes) * 1.35)
        else:
            self._marcar_vacio(eje, 'Se llena al procesar con GPU')
        eje.set_title('Reparto entre los multiprocesadores de la GPU',
                      fontsize=10)
        eje.grid(axis='y', alpha=0.3)

        self.lienzo_nuc.draw()

    def matriz_evidencias(self):
        """Matriz de la rubrica con todo lo que se ha medido hasta ahora."""
        return evidencias.matriz(
            recursos=self.ultimos_recursos,
            censo_sm=self.ultimo_censo,
            recursos_kernel=self.recursos_kernel,
            info_gpu=self.datos_gpu,
            nucleos_cpu=self.ultimos_recursos.get('cpu_por_nucleo'))

    def _rellenar_evidencias(self):
        """Vuelca la matriz de la rubrica en su pestana."""
        self.texto_evid.configure(state='normal')
        self.texto_evid.delete('1.0', 'end')
        self.texto_evid.insert('end',
                               evidencias.texto(self.matriz_evidencias(), 30))
        self.texto_evid.configure(state='disabled')

    # ------------------------------------------------------------------
    # Acciones de la interfaz
    # ------------------------------------------------------------------

    def elegir_archivo(self):
        ruta = filedialog.askopenfilename(
            title='Selecciona el archivo FASTA',
            filetypes=[('Archivos FASTA', '*.fna *.fa *.fasta *.txt'),
                       ('Todos los archivos', '*.*')])
        if not ruta:
            return
        self.ruta.set(ruta)
        tamano = os.path.getsize(ruta)
        self.etiqueta_tamano.configure(text=benchmark.formato_tamano(tamano))
        self.estado.set('Archivo listo: %s'
                        % benchmark.formato_tamano(tamano))

    def _validar(self):
        ruta = self.ruta.get().strip()
        if not ruta:
            messagebox.showwarning('Falta el archivo',
                                   'Selecciona primero un archivo FASTA.')
            return None
        if not os.path.exists(ruta):
            messagebox.showerror('Archivo no encontrado',
                                 'No existe el archivo:\n%s' % ruta)
            return None
        if self.trabajando:
            messagebox.showinfo('En proceso',
                                'Ya hay un procesamiento en curso.')
            return None
        return ruta

    def procesar_una(self):
        """Ejecuta una sola configuracion con los valores elegidos."""
        ruta = self._validar()
        if ruta:
            self._lanzar(self._tarea_una, ruta)

    def analisis_completo(self):
        """Ejecuta la comparativa de modos y los dos barridos."""
        ruta = self._validar()
        if not ruta:
            return
        tamano = os.path.getsize(ruta)
        if tamano > 1024 ** 3:
            seguir = messagebox.askyesno(
                'Analisis completo',
                'El archivo ocupa %s. El analisis completo lanza varias\n'
                'pasadas sobre el y puede tardar bastantes minutos.\n\n'
                'Continuar?' % benchmark.formato_tamano(tamano))
            if not seguir:
                return
        self._lanzar(self._tarea_completa, ruta)

    def exportar_csv(self):
        if not self.filas:
            messagebox.showinfo('Sin resultados',
                                'Todavia no hay resultados que exportar.')
            return
        ruta = filedialog.asksaveasfilename(
            title='Guardar resultados', defaultextension='.csv',
            initialfile='benchmark_p13.csv', filetypes=[('CSV', '*.csv')])
        if not ruta:
            return

        benchmark.exportar_csv(self.filas, ruta)
        base = os.path.splitext(ruta)[0]
        benchmark.exportar_registro(self.filas, base + '_registro.csv')
        evidencias.guardar(self.matriz_evidencias(),
                           base + '_evidencias.txt')

        self.estado.set('Resultados exportados a %s' % os.path.basename(ruta))
        messagebox.showinfo('Exportado',
                            'Guardados el CSV, la linea de tiempo del\n'
                            'reparto y la matriz de evidencias junto a:\n%s'
                            % ruta)

    # ------------------------------------------------------------------
    # Hilo trabajador
    # ------------------------------------------------------------------

    def _lanzar(self, tarea, ruta):
        """Arranca la tarea en un hilo aparte y bloquea los botones.

        Antes de arrancar se copian los valores de los controles a atributos
        normales. Las variables de Tkinter son objetos de la biblioteca
        grafica y leerlas tambien cuenta como tocar la interfaz, asi que el
        hilo trabajador no debe llamar a su metodo get(): se queda con esta
        fotografia, tomada desde el hilo principal.
        """
        self.config = {
            'modo': self.modo.get(),
            'procesos': max(1, self.procesos.get()),
            'trozo_mb': self.trozo_mb.get(),
        }

        self.trabajando = True
        self.inicio_proceso = time.perf_counter()
        self.boton_procesar.state(['disabled'])
        self.boton_analisis.state(['disabled'])
        self.barra['value'] = 0

        self.hilo = threading.Thread(target=self._envolver, args=(tarea, ruta),
                                     daemon=True)
        self.hilo.start()

    def _envolver(self, tarea, ruta):
        """Ejecuta la tarea capturando cualquier error para la ventana."""
        try:
            tarea(ruta)
        except Exception as error:
            self.cola.put(('error', '%s: %s' % (type(error).__name__, error)))
        finally:
            self.cola.put(('fin', None))

    def _progreso(self, hechos, total):
        self.cola.put(('progreso', (hechos, total)))

    def _estado(self, texto):
        self.cola.put(('estado', texto))

    def _log(self, texto):
        self.cola.put(('log', texto))

    def _tarea_una(self, ruta):
        """Una sola pasada con el modo y los parametros seleccionados."""
        tamano = os.path.getsize(ruta)
        modo = self.config['modo']
        procesos = self.config['procesos']
        trozo = self.config['trozo_mb']

        marca = self.monitor.marcar() if self.monitor else 0

        if modo == 'cpu':
            self._estado('Procesando en CPU con %d proceso(s)...' % procesos)
            etiqueta, plataforma = 'CPU x%d' % procesos, 'CPU'
            if procesos == 1:
                resultado, tiempo = motor_cpu.contar_secuencial(
                    ruta, self._progreso)
            else:
                resultado, tiempo = motor_cpu.contar_paralelo(
                    ruta, procesos, self._progreso)
        elif modo == 'gpu':
            self._estado('Compilando kernels CUDA...')
            self._log('Compilacion JIT: %.3f s (fuera de la medicion)'
                      % self.motor_gpu.precalentar())
            self._estado('Procesando en GPU...')
            etiqueta, plataforma = 'GPU sola', 'GPU'
            resultado, tiempo = motor_hibrido.contar_hibrido(
                ruta, trozo_mb=trozo, usar_cpu=False,
                progreso=self._progreso)
        else:
            self._estado('Compilando kernels CUDA...')
            self._log('Compilacion JIT: %.3f s (fuera de la medicion)'
                      % self.motor_gpu.precalentar())
            self._estado('Procesando con CPU y GPU a la vez...')
            etiqueta = 'CPU x%d + GPU' % procesos
            plataforma = 'HIBRIDO'
            resultado, tiempo = motor_hibrido.contar_hibrido(
                ruta, procesos=procesos, trozo_mb=trozo,
                progreso=self._progreso)

        recursos = self.monitor.resumen(marca) if self.monitor else {}
        if recursos.get('muestras', 0) < mod_monitor.MINIMO_MUESTRAS:
            recursos = {}

        fila = benchmark.fila_base(etiqueta, plataforma, resultado, tiempo,
                                   tamano)
        fila['modo'] = modo
        benchmark._anadir_recursos(fila, recursos)

        datos = resultado.get('hibrido') or {}
        if datos.get('registro'):
            fila['registro_hibrido'] = datos['registro']

        self.cola.put(('recursos', recursos))
        self.cola.put(('resultado', resultado))
        self.cola.put(('filas', [fila]))

        self._log('%-16s %8.3f s  (%.1f MB/s)'
                  % (etiqueta, tiempo, fila['mb_por_s']))
        if datos:
            self._log(motor_hibrido.resumen_reparto(datos))

    def _tarea_completa(self, ruta):
        """Comparativa de modos mas los barridos de procesos y de trozo.

        El muestreo de recursos se hace con el monitor que ya mantiene la
        ventana para su panel en directo. No se crea uno nuevo a proposito:
        dos monitores simultaneos se corromperian las lecturas de psutil.
        """
        avisar = lambda t: self._estado('Analisis completo: ' + t)

        self._estado('Analisis completo: comparativa de modos...')
        filas, referencia = benchmark.comparar_modos(
            ruta, procesos=self.config['procesos'],
            trozo_mb=self.config['trozo_mb'], progreso=avisar,
            monitor=self.monitor)
        self._publicar(filas)
        for fila in filas:
            self._log('%-16s %8.3f s  %8.1f MB/s'
                      % (fila['etiqueta'], fila['tiempo_s'],
                         fila['mb_por_s']))

        base = filas[0]['tiempo_s'] if filas else None

        self._estado('Analisis completo: cuantos procesos sumarle a la GPU...')
        filas_proc, referencia = benchmark.barrido_procesos(
            ruta, trozo_mb=self.config['trozo_mb'], referencia=referencia,
            base_cpu=base, progreso=avisar, monitor=self.monitor)
        self._publicar(filas_proc)
        if filas_proc:
            mejor = min(filas_proc, key=lambda f: f['tiempo_s'])
            self._log('Mejor numero de procesos: %s (%.3f s)'
                      % (mejor['etiqueta'], mejor['tiempo_s']))

        self._estado('Analisis completo: grano del reparto...')
        filas_trozo, referencia = benchmark.barrido_trozo(
            ruta, procesos=self.config['procesos'], referencia=referencia,
            base_cpu=base, progreso=avisar, monitor=self.monitor)
        self._publicar(filas_trozo)
        if filas_trozo:
            mejor = min(filas_trozo, key=lambda f: f['tiempo_s'])
            self._log('Mejor grano de reparto: %s (%.3f s)'
                      % (mejor['etiqueta'], mejor['tiempo_s']))

        self._estado('Analisis completo: escalabilidad...')
        filas_esc = benchmark.escalabilidad(
            ruta, procesos=self.config['procesos'],
            trozo_mb=self.config['trozo_mb'], progreso=avisar)
        for fila in filas_esc:
            fila['escala'] = True
        self._publicar(filas_esc)

        if referencia is not None:
            self.cola.put(('resultado', referencia))

        # Sin esto, la matriz de evidencias se queda sin las lecturas de uso,
        # temperatura y memoria justo despues de un analisis completo, que es
        # cuando mas datos hay que ensenar.
        if self.monitor:
            self.cola.put(('recursos', self.monitor.resumen()))

        malos = [f for f in filas + filas_proc + filas_trozo
                 if not f.get('conteo_ok', True)]
        if malos:
            self._log('ATENCION: %d configuracion(es) con conteo distinto'
                      % len(malos))
        else:
            self._log('Todas las configuraciones dieron el mismo conteo.')

    def _publicar(self, filas):
        """Manda a la ventana un grupo de filas recien medidas."""
        self.cola.put(('filas', filas))

    # ------------------------------------------------------------------
    # Vaciado de la cola desde el hilo principal
    # ------------------------------------------------------------------

    def _revisar_cola(self):
        """Aplica a la ventana los mensajes dejados por el hilo trabajador.

        Es el unico punto donde se tocan los widgets, y siempre se ejecuta en
        el hilo principal. Tkinter no admite otra cosa.
        """
        try:
            while True:
                tipo, dato = self.cola.get_nowait()

                if tipo == 'progreso':
                    hechos, total = dato
                    porcentaje = 100.0 * hechos / total if total else 0.0
                    self.barra['value'] = min(100.0, porcentaje)
                    if self.inicio_proceso:
                        pasado = time.perf_counter() - self.inicio_proceso
                        if pasado > 0:
                            mbs = hechos / (1024 * 1024) / pasado
                            self.velocidad.set('%.1f MB/s' % mbs)

                elif tipo == 'estado':
                    self.estado.set(dato)

                elif tipo == 'log':
                    self.texto_log.insert('end', dato + '\n')
                    self.texto_log.see('end')

                elif tipo == 'recursos':
                    if dato:
                        self.ultimos_recursos = dato
                        self._dibujar_nucleos()
                        self._rellenar_evidencias()

                elif tipo == 'resultado':
                    if dato is not None:
                        self._mostrar_conteo(dato)

                elif tipo == 'filas':
                    self.filas.extend(dato)
                    for fila in dato:
                        if fila.get('registro_hibrido'):
                            self.ultimo_registro = fila['registro_hibrido']
                        if fila.get('censo_sm'):
                            self.ultimo_censo = fila['censo_sm']
                        if fila['plataforma'] == 'HIBRIDO':
                            self._mostrar_reparto(fila)
                    self._rellenar_tabla()
                    self._dibujar_graficas()
                    self._dibujar_nucleos()
                    self._rellenar_evidencias()

                elif tipo == 'error':
                    messagebox.showerror('Error durante el procesamiento',
                                         dato)
                    self.estado.set('Error: %s' % dato)

                elif tipo == 'fin':
                    self._terminar()

        except queue.Empty:
            pass

        if not self.cerrando:
            self.raiz.after(INTERVALO_COLA, self._revisar_cola)

    def cerrar(self):
        """Cierra la ventana dejando el programa en un estado limpio.

        Si hay un analisis en marcha se pide confirmacion, porque el hilo
        trabajador no se puede interrumpir a mitad de una lectura de tres
        gigabytes sin dejar procesos hijos sueltos.
        """
        if self.trabajando:
            seguir = messagebox.askyesno(
                'Analisis en curso',
                'Hay un procesamiento en marcha.\n\n'
                'Si cierras ahora se perderan los resultados que falten.\n'
                'Cerrar de todos modos?')
            if not seguir:
                return

        self.cerrando = True
        if self.monitor:
            self.monitor.detener()
            self.monitor = None
        self.raiz.destroy()

    def _terminar(self):
        self.trabajando = False
        self.boton_procesar.state(['!disabled'])
        self.boton_analisis.state(['!disabled'])
        self.barra['value'] = 100

        # Los recursos del kernel solo se pueden leer despues de que se haya
        # compilado y lanzado al menos una vez.
        if self.hay_gpu and self.recursos_kernel is None:
            try:
                self.recursos_kernel = self.motor_gpu.recursos_kernel()
                self._rellenar_evidencias()
            except Exception:
                pass

        if self.inicio_proceso:
            total = time.perf_counter() - self.inicio_proceso
            self.estado.set('Terminado en %.2f s' % total)
            self.inicio_proceso = None

    def _mostrar_conteo(self, resultado):
        for clave, etiqueta in self.etiquetas_conteo.items():
            etiqueta.configure(text='{:,}'.format(resultado.get(clave, 0)))

        datos = resultado.get('hibrido')
        if datos and datos.get('censo_sm'):
            self.ultimo_censo = datos['censo_sm']

        if datos:
            self._escribir_reparto(motor_hibrido.resumen_reparto(datos))

        self.ultimo_resultado = resultado

    def _escribir_reparto(self, texto):
        """Vuelca en el panel de la izquierda el resumen del reparto."""
        self.texto_reparto.configure(state='normal')
        self.texto_reparto.delete('1.0', 'end')
        self.texto_reparto.insert('end', texto)
        self.texto_reparto.configure(state='disabled')

    def _mostrar_reparto(self, fila):
        """Resumen del reparto a partir de una fila ya medida.

        El analisis completo publica filas, no diccionarios de resultado, asi
        que el panel se alimenta de la fila. Sin esto el cuadro quedaba vacio
        justo despues de un analisis completo, que es cuando mas informacion
        hay que mostrar.
        """
        self._escribir_reparto(
            'Trozos de %s MB: %s en total\n'
            '  CPU x%-2s  %3s trozos  %5.1f %%\n'
            '  GPU      %3s trozos  %5.1f %%\n'
            '  Trabajando a la vez: %.2f s de %.2f s'
            % (fila.get('trozo_mb', '?'),
               (fila.get('trozos_cpu', 0) or 0) + (fila.get('trozos_gpu', 0) or 0),
               fila.get('procesos', '?'), fila.get('trozos_cpu', 0),
               fila.get('reparto_cpu_pct', 0.0),
               fila.get('trozos_gpu', 0), fila.get('reparto_gpu_pct', 0.0),
               fila.get('solape_s', 0.0) or 0.0, fila.get('tiempo_s', 0.0)))

    def _rellenar_tabla(self):
        def numero(fila, clave, formato, sufijo=''):
            """Formatea un valor, o un guion si no se llego a medir.

            Distinguir el cero del no medido importa: en las corridas mas
            cortas que el intervalo de muestreo no hay lecturas de recursos,
            y escribir 0 ahi haria pensar que la tarjeta no se uso.
            """
            valor = fila.get(clave)
            if valor in ('', None):
                return '-'
            return (formato % valor) + sufijo

        for elemento in self.tabla.get_children():
            self.tabla.delete(elemento)

        for fila in self.filas:
            if fila.get('reparto_cpu_pct') not in ('', None):
                reparto = '%.0f%% / %.0f%%' % (fila['reparto_cpu_pct'],
                                               fila['reparto_gpu_pct'])
            else:
                reparto = '-'
            self.tabla.insert('', 'end', values=(
                fila.get('etiqueta', ''),
                fila.get('plataforma', ''),
                '%.3f' % fila.get('tiempo_s', 0),
                '%.1f' % fila.get('mb_por_s', 0),
                numero(fila, 'speedup', '%.2f', 'x'),
                reparto,
                numero(fila, 'solape_s', '%.2f'),
                numero(fila, 'cpu_medio', '%.0f'),
                numero(fila, 'gpu_medio', '%.0f'),
                numero(fila, 'cpu_temp_medio', '%.0f'),
                numero(fila, 'gpu_temp_max', '%.0f'),
                numero(fila, 'vram_mb_max', '%d'),
                'si' if fila.get('conteo_ok', True) else 'NO',
            ))

    # ------------------------------------------------------------------
    # Panel de recursos en vivo
    # ------------------------------------------------------------------

    def _refrescar_recursos(self):
        """Actualiza las barras de uso cada medio segundo."""
        muestra = self.monitor.ultima() if self.monitor else {}

        def poner(clave, valor, texto):
            barra, etiqueta = self.etiquetas_recursos[clave]
            if valor is None:
                barra['value'] = 0
                etiqueta.configure(text='-')
            else:
                barra['value'] = max(0, min(100, valor))
                etiqueta.configure(text=texto)

        poner('cpu', muestra.get('cpu'),
              '%.0f %%' % muestra['cpu'] if 'cpu' in muestra else '-')
        poner('ram', muestra.get('ram'),
              '%.0f %%' % muestra['ram'] if 'ram' in muestra else '-')
        poner('gpu', muestra.get('gpu'),
              '%d %%' % muestra['gpu'] if 'gpu' in muestra else '-')

        # Las dos temperaturas comparten una escala de 0 a 100 grados, que
        # cubre de sobra el rango util de un portatil.
        if 'cpu_temp' in muestra:
            poner('tcpu', muestra['cpu_temp'], '%.0f C' % muestra['cpu_temp'])
        else:
            poner('tcpu', None, '-')

        if 'gpu_temp' in muestra:
            poner('tgpu', muestra['gpu_temp'], '%d C' % muestra['gpu_temp'])
        else:
            poner('tgpu', None, '-')

        if 'vram_mb' in muestra and self.hay_gpu:
            total = max(1, self.datos_gpu.get('vram_total_mb', 1))
            poner('vram', 100.0 * muestra['vram_mb'] / total,
                  '%d MB' % muestra['vram_mb'])
        else:
            poner('vram', None, '-')

        if not self.cerrando:
            self.raiz.after(INTERVALO_RECURSOS, self._refrescar_recursos)


def ejecutar():
    """Crea la ventana y entra en el bucle de eventos."""
    raiz = tk.Tk()
    try:
        # Tema mas cercano al aspecto nativo de Windows.
        ttk.Style().theme_use('vista')
    except Exception:
        pass
    Aplicacion(raiz)
    raiz.mainloop()
    return 0
