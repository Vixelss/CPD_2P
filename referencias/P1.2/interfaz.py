# -*- coding: utf-8 -*-
"""
interfaz.py

Ventana de la aplicacion del P1.2: comparacion CPU contra GPU en el
procesamiento paralelo de cadenas de ADN.

Se arranca desde app_gpu.py, nunca directamente, porque ese archivo es el
que pone el guard de multiprocessing que Windows exige.

ESTRUCTURA
  - Columna izquierda: controles, hardware detectado, conteos y recursos.
  - Columna derecha : pestanas con las graficas y la tabla comparativa.
  - Pie             : barra de progreso con velocidad y estado.

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

import benchmark_gpu
import monitor as mod_monitor
import motor_cpu


# Paleta sobria, pensada para que las graficas se lean igual impresas en el
# informe en blanco y negro.
COLOR_CPU = '#2f6f9f'
COLOR_CPU2 = '#7fb3d5'
COLOR_GPU = '#76b041'
COLOR_AVISO = '#b03030'
COLOR_FONDO = '#f4f4f4'

INTERVALO_COLA = 100        # ms entre revisiones de la cola de mensajes
INTERVALO_RECURSOS = 500    # ms entre refrescos del panel de recursos


def _corto(nombre):
    """Acorta el nombre comercial de una tarjeta para que quepa en el panel."""
    limpio = nombre.replace('NVIDIA GeForce ', '').replace('(R)', '')
    return limpio.replace(' Laptop GPU', '').replace('Graphics', 'Gfx').strip()


class Aplicacion:
    """Ventana principal de la aplicacion."""

    def __init__(self, raiz):
        self.raiz = raiz
        self.raiz.title('P1.2 - Computacion GPU Paralela | CPU vs GPU sobre ADN')

        # La ventana se adapta a la pantalla en lugar de imponer un tamano
        # fijo. En el portatil donde se desarrollo esto la pantalla util son
        # 1536x864 menos la barra de tareas, asi que una ventana de 1320x820
        # dejaba la barra de progreso fuera del borde inferior y recortaba las
        # graficas por la derecha. Se reserva margen para la barra de tareas y
        # para el marco de la ventana.
        ancho_pantalla = self.raiz.winfo_screenwidth()
        alto_pantalla = self.raiz.winfo_screenheight()
        self.ancho = min(1300, ancho_pantalla - 40)
        self.alto = min(800, alto_pantalla - 110)
        self.raiz.geometry('%dx%d+%d+%d'
                           % (self.ancho, self.alto,
                              max(0, (ancho_pantalla - self.ancho) // 2), 10))
        self.raiz.minsize(900, 600)

        # Estado compartido entre el hilo trabajador y la ventana.
        self.cola = queue.Queue()
        self.hilo = None
        self.trabajando = False
        self.monitor = None
        self.inicio_proceso = None

        self.ruta = tk.StringVar(value='')
        self.modo = tk.StringVar(value='ambos')
        self.procesos = tk.IntVar(value=0)
        self.lote_mb = tk.IntVar(value=benchmark_gpu.LOTES_POR_DEFECTO[3])
        self.estado = tk.StringVar(value='Listo. Selecciona un archivo FASTA.')
        self.velocidad = tk.StringVar(value='')

        # Resultados acumulados, para las graficas y el export a CSV.
        self.filas = []
        self.ultimo_resultado = None

        # Fotografia de los controles tomada al pulsar un boton, para que el
        # hilo trabajador no tenga que leer variables de Tkinter.
        self.config = {'modo': 'ambos', 'procesos': 1, 'lote_mb': 64}
        self._eje_vram = None

        # Se pone a True al cerrar la ventana. Los bucles que se reprograman
        # con after lo consultan antes de volver a encolarse: sin esto, una
        # llamada pendiente puede dispararse cuando el interprete de Tcl ya
        # no existe y provocar un error al salir.
        self.cerrando = False

        self._detectar_hardware()
        self._construir()
        self.raiz.protocol('WM_DELETE_WINDOW', self.cerrar)
        # Deja las graficas de recursos con su aviso de "sin datos" en lugar
        # de unos ejes vacios con la escala por defecto de 0 a 1.
        self._dibujar_recursos()
        self._revisar_cola()
        self._refrescar_recursos()

    # ------------------------------------------------------------------
    # Hardware
    # ------------------------------------------------------------------

    def _detectar_hardware(self):
        """Lee las caracteristicas de CPU y GPU una sola vez al arrancar."""
        self.fisicos, self.logicos = motor_cpu.detectar_nucleos()
        self.procesos.set(self.logicos)

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

        # OpenCL es opcional: si falta pyopencl o no hay implementacion en la
        # maquina, la aplicacion funciona igual y solo pierde la comparativa
        # entre tarjetas.
        try:
            import motor_opencl
            self.dispositivos_opencl = motor_opencl.listar_dispositivos()
        except Exception:
            self.dispositivos_opencl = []

    # ------------------------------------------------------------------
    # Construccion de la interfaz
    # ------------------------------------------------------------------

    def _construir(self):
        self.raiz.configure(bg=COLOR_FONDO)

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
        self._panel_pie(self.raiz)

    def _panel_archivo(self, padre):
        marco = ttk.LabelFrame(padre, text=' Archivo ', padding=8)
        marco.pack(fill='x', pady=(0, 6))

        entrada = ttk.Entry(marco, textvariable=self.ruta)
        entrada.pack(fill='x', pady=(0, 6))

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
        for texto, valor in (('CPU', 'cpu'), ('GPU', 'gpu'), ('Ambos', 'ambos')):
            boton = ttk.Radiobutton(fila, text=texto, value=valor,
                                    variable=self.modo)
            boton.pack(side='left', padx=4)
            if valor in ('gpu', 'ambos') and not self.hay_gpu:
                boton.state(['disabled'])

        fila = ttk.Frame(marco)
        fila.pack(fill='x', pady=2)
        ttk.Label(fila, text='Procesos CPU:').pack(side='left')
        ttk.Spinbox(fila, from_=1, to=max(32, self.logicos), width=6,
                    textvariable=self.procesos).pack(side='left', padx=6)
        ttk.Label(fila, text='(logicos: %d)' % self.logicos).pack(side='left')

        fila = ttk.Frame(marco)
        fila.pack(fill='x', pady=2)
        ttk.Label(fila, text='Lote GPU (MB):').pack(side='left')
        combo = ttk.Combobox(fila, width=6, state='readonly',
                             textvariable=self.lote_mb,
                             values=list(benchmark_gpu.LOTES_POR_DEFECTO))
        combo.pack(side='left', padx=6)
        if not self.hay_gpu:
            combo.state(['disabled'])

        botones = ttk.Frame(marco)
        botones.pack(fill='x', pady=(8, 0))
        self.boton_procesar = ttk.Button(botones, text='Procesar',
                                         command=self.procesar_una)
        self.boton_procesar.pack(side='left')
        self.boton_analisis = ttk.Button(botones, text='Analisis completo',
                                         command=self.analisis_completo)
        self.boton_analisis.pack(side='left', padx=6)
        self.boton_csv = ttk.Button(botones, text='Exportar CSV',
                                    command=self.exportar_csv)
        self.boton_csv.pack(side='left')

    def _panel_hardware(self, padre):
        marco = ttk.LabelFrame(padre, text=' Hardware detectado ', padding=8)
        marco.pack(fill='x', pady=(0, 6))

        texto = ('CPU: %d nucleos fisicos / %d logicos'
                 % (self.fisicos, self.logicos))
        ttk.Label(marco, text=texto).pack(anchor='w')

        if self.hay_gpu:
            d = self.datos_gpu
            for linea in (
                'GPU: %s' % d['nombre'],
                '     Compute capability %s, %d SMs, warp %d'
                % (d['compute_capability'], d['sms'], d['warp']),
                '     VRAM %d MB (%d MB libres)'
                % (d['vram_total_mb'], d['vram_libre_mb']),
            ):
                ttk.Label(marco, text=linea).pack(anchor='w')
        else:
            ttk.Label(marco, text='GPU: no disponible (solo modo CPU)',
                      foreground=COLOR_AVISO).pack(anchor='w')

        # Unidades OpenCL. Se listan aparte porque incluyen la grafica
        # integrada del procesador, que con CUDA no es alcanzable y que el
        # analisis completo usa para el punto extra de discreta contra
        # integrada.
        if self.dispositivos_opencl:
            ttk.Label(marco, text='OpenCL:').pack(anchor='w')
            for datos in self.dispositivos_opencl:
                ttk.Label(marco, text='     %s (%s, %d unidades)'
                          % (_corto(datos['nombre']), datos['tipo'],
                             datos['unidades'])).pack(anchor='w')

        disp = mod_monitor.disponibilidad()
        ttk.Label(marco, text='Monitoreo: psutil %s, pynvml %s'
                  % ('si' if disp['psutil'] else 'no',
                     'si' if disp['pynvml'] else 'no')).pack(anchor='w')

    def _panel_conteo(self, padre):
        marco = ttk.LabelFrame(padre, text=' Conteo ', padding=8)
        marco.pack(fill='x', pady=(0, 6))

        # Dos columnas en vez de una lista de ocho filas: a la izquierda las
        # cuatro bases y su total, a la derecha las tres categorias que no
        # son base. Ademas de agrupar mejor la informacion, ahorra la mitad
        # de alto, que es justo lo que hacia falta para que el panel de
        # recursos no quedara fuera del borde inferior de la ventana.
        self.etiquetas_conteo = {}
        rejilla = ttk.Frame(marco)
        rejilla.pack(fill='x')

        columnas = (
            (('A', 'Adenina  (A)'), ('C', 'Citosina (C)'),
             ('G', 'Guanina  (G)'), ('T', 'Timina   (T)'),
             ('bases', 'Total bases')),
            (('N', 'N desconoc.'), ('ambiguos', 'Cod. IUPAC'),
             ('invalidos', 'INVALIDOS')),
        )

        for indice, columna in enumerate(columnas):
            base = indice * 2
            for fila_n, (clave, etiqueta) in enumerate(columna):
                ttk.Label(rejilla, text=etiqueta).grid(
                    row=fila_n, column=base, sticky='w', padx=(0, 4))
                valor = ttk.Label(rejilla, text='-', anchor='e', width=11)
                valor.grid(row=fila_n, column=base + 1, sticky='e',
                           padx=(0, 10))
                self.etiquetas_conteo[clave] = valor

        ttk.Separator(marco, orient='horizontal').pack(fill='x', pady=4)
        ttk.Label(marco, text='Detalle de caracteres:').pack(anchor='w')
        # Tres lineas visibles con barra de desplazamiento: en el genoma real
        # aparecen siete codigos IUPAC distintos y en los archivos de prueba
        # mas de veinte caracteres invalidos, asi que la lista tiene que poder
        # crecer sin empujar fuera de la ventana el panel de recursos.
        caja = ttk.Frame(marco)
        caja.pack(fill='x', pady=(2, 0))
        self.texto_invalidos = tk.Text(caja, height=3, width=34,
                                       font=('Consolas', 8), wrap='none')
        self.texto_invalidos.pack(side='left', fill='x', expand=True)
        desliza = ttk.Scrollbar(caja, orient='vertical',
                                command=self.texto_invalidos.yview)
        desliza.pack(side='right', fill='y')
        self.texto_invalidos.configure(yscrollcommand=desliza.set,
                                       state='disabled')

    def _panel_recursos(self, padre):
        """Construye el panel de recursos como una tira horizontal en el pie.

        Estaba al final de la columna izquierda, pero ahi no cabia: Tk no
        tiene espacio que repartir y comprime los ultimos widgets en vez de
        desbordarlos, de modo que las barras de VRAM y temperatura quedaban
        de 12 y 1 pixel de alto. El pie de la ventana mide lo mismo que la
        ventana entera y estaba practicamente vacio, asi que las cinco
        lecturas caben de sobra en una sola fila.
        """
        self.etiquetas_recursos = {}
        for clave, etiqueta in (('cpu', 'CPU'), ('ram', 'RAM'),
                                ('gpu', 'GPU'), ('vram', 'VRAM'),
                                ('temp', 'Temp')):
            grupo = ttk.Frame(padre)
            grupo.pack(side='left', padx=(0, 10))
            ttk.Label(grupo, text=etiqueta).pack(side='left', padx=(0, 3))
            barra = ttk.Progressbar(grupo, length=70, maximum=100)
            barra.pack(side='left')
            valor = ttk.Label(grupo, text='-', width=7)
            valor.pack(side='left', padx=(3, 0))
            self.etiquetas_recursos[clave] = (barra, valor)

    def _panel_pestanas(self, padre):
        self.pestanas = ttk.Notebook(padre)
        self.pestanas.pack(fill='both', expand=True)

        # -- graficas --
        marco_graficas = ttk.Frame(self.pestanas)
        self.pestanas.add(marco_graficas, text='  Graficas  ')

        # El tamano de la figura se calcula a partir del espacio que queda
        # de verdad a la derecha del panel de controles. Si se fija a ojo, el
        # widget del lienzo pide mas espacio del que hay y Tk recorta las
        # graficas por el borde. El lienzo se expande despues con la ventana.
        ancho_figura = max(6.0, (self.ancho - 380) / 100.0)
        alto_figura = max(4.2, (self.alto - 130) / 100.0)
        self.figura = Figure(figsize=(ancho_figura, alto_figura), dpi=100)
        self.figura.subplots_adjust(hspace=0.48, wspace=0.30,
                                    left=0.09, right=0.97,
                                    top=0.92, bottom=0.13)
        self.ejes = [self.figura.add_subplot(2, 2, i + 1) for i in range(4)]
        self._limpiar_graficas()

        self.lienzo = FigureCanvasTkAgg(self.figura, master=marco_graficas)
        self.lienzo.get_tk_widget().pack(fill='both', expand=True)

        # -- recursos --
        # El enunciado pide comparar CPU y GPU no solo por tiempo sino por
        # "uso de CPU/GPU (memoria, carga)" y por "velocidad de
        # identificacion de errores por segundo". Esas dos magnitudes tienen
        # su propia pestana porque no caben en la escala de las graficas de
        # tiempo: una se mide en porcentaje y la otra en errores por segundo.
        marco_recursos = ttk.Frame(self.pestanas)
        self.pestanas.add(marco_recursos, text='  Recursos  ')

        self.figura_rec = Figure(figsize=(ancho_figura, alto_figura), dpi=100)
        self.figura_rec.subplots_adjust(hspace=0.48, wspace=0.30,
                                        left=0.09, right=0.97,
                                        top=0.92, bottom=0.13)
        self.ejes_rec = [self.figura_rec.add_subplot(2, 2, i + 1)
                         for i in range(4)]
        self.lienzo_rec = FigureCanvasTkAgg(self.figura_rec,
                                            master=marco_recursos)
        self.lienzo_rec.get_tk_widget().pack(fill='both', expand=True)

        # -- tabla --
        marco_tabla = ttk.Frame(self.pestanas)
        self.pestanas.add(marco_tabla, text='  Tabla comparativa  ')

        # Las columnas cubren los cinco criterios de comparacion del
        # enunciado: tiempo total, utilizacion de recursos de hardware,
        # velocidad de identificacion de errores por segundo, escalabilidad
        # (las filas por tamano) y la verificacion de que el conteo coincide.
        columnas = ('etiqueta', 'plataforma', 'tiempo_s', 'mb_por_s',
                    'speedup', 'eficiencia_pct', 'invalidos', 'errores_s',
                    'cpu_medio', 'gpu_medio', 'vram_mb_max', 'gpu_temp_max',
                    'conteo_ok')
        titulos = ('Configuracion', 'Plataforma', 'Tiempo (s)', 'MB/s',
                   'Speedup', 'Efic. %', 'Invalidos', 'Errores/s',
                   'CPU %', 'GPU %', 'VRAM MB', 'Temp C', 'Conteo OK')
        anchos = (130, 75, 75, 70, 65, 60, 75, 80,
                  55, 55, 70, 60, 75)

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

        self.barra = ttk.Progressbar(marco, length=220, maximum=100)
        self.barra.pack(side='left')
        ttk.Label(marco, textvariable=self.velocidad,
                  width=12).pack(side='left', padx=6)

        # Las lecturas de recursos van pegadas al borde derecho del pie.
        recursos = ttk.Frame(marco)
        recursos.pack(side='right')
        self._panel_recursos(recursos)

        ttk.Label(marco, textvariable=self.estado).pack(side='left', padx=6)

    # ------------------------------------------------------------------
    # Graficas
    # ------------------------------------------------------------------

    def _limpiar_graficas(self):
        titulos = ('Tiempo por configuracion',
                   'Speedup frente a CPU secuencial',
                   'Eficiencia por proceso (CPU)',
                   'Escalabilidad: tiempo por tamano')
        for eje, titulo in zip(self.ejes, titulos):
            eje.clear()
            eje.set_title(titulo, fontsize=10)
            eje.text(0.5, 0.5, 'Sin datos todavia', ha='center', va='center',
                     transform=eje.transAxes, color='#999999', fontsize=9)
            eje.set_xticks([])
            eje.set_yticks([])
        if hasattr(self, 'lienzo'):
            self.lienzo.draw()

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
        """Redibuja las cuatro graficas con los resultados acumulados."""
        # Las filas del barrido de escalabilidad se excluyen de las dos
        # primeras graficas: corresponden a archivos recortados de distinto
        # tamano, asi que sus tiempos no son comparables con los del archivo
        # seleccionado y mezclarlos daria barras enganosas. Van solo a la
        # cuarta grafica, que es donde el eje X es precisamente el tamano.
        directas = [f for f in self.filas if not f.get('escala')
                    and not f.get('errores') and not f.get('dispositivos')]
        cpu = [f for f in directas if f['plataforma'] == 'CPU']
        gpu = [f for f in directas if f['plataforma'] == 'GPU']

        for eje in self.ejes:
            eje.clear()

        # 1) Tiempo por configuracion.
        eje = self.ejes[0]
        etiquetas = [f['etiqueta'] for f in cpu + gpu]
        tiempos = [f['tiempo_s'] for f in cpu + gpu]
        colores = [COLOR_CPU] * len(cpu) + [COLOR_GPU] * len(gpu)
        if etiquetas:
            barras = eje.bar(range(len(etiquetas)), tiempos, color=colores)
            eje.set_xticks(range(len(etiquetas)))
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

        # 2) Speedup. La linea base es siempre la CPU con un solo proceso,
        #    de modo que las barras de CPU y de GPU son comparables entre si.
        eje = self.ejes[1]
        datos = [f for f in cpu + gpu if f.get('speedup') not in ('', None)]
        if datos:
            valores = [f['speedup'] for f in datos]
            colores = [COLOR_CPU if f['plataforma'] == 'CPU' else COLOR_GPU
                       for f in datos]
            eje.bar(range(len(datos)), valores, color=colores)
            eje.axhline(1.0, color='#888888', linestyle='--', linewidth=0.8)
            eje.set_xticks(range(len(datos)))
            eje.set_xticklabels([f['etiqueta'] for f in datos],
                                rotation=45, ha='right', fontsize=7)
            eje.set_ylabel('veces mas rapido', fontsize=8)
        else:
            self._marcar_vacio(eje)
        eje.set_title('Speedup frente a CPU secuencial', fontsize=10)
        eje.grid(axis='y', alpha=0.3)

        # 3) Eficiencia de la CPU: que fraccion de cada proceso se aprovecha.
        eje = self.ejes[2]
        datos = [f for f in cpu if f.get('eficiencia_pct') not in ('', None)]
        if datos:
            xs = [f['procesos'] for f in datos]
            ys = [f['eficiencia_pct'] for f in datos]
            eje.plot(xs, ys, marker='o', color=COLOR_CPU)
            eje.axhline(100, color='#888888', linestyle='--', linewidth=0.8)
            eje.set_xlabel('procesos', fontsize=8)
            eje.set_ylabel('eficiencia %', fontsize=8)
            eje.set_ylim(0, max(110, max(ys) * 1.1))
        else:
            self._marcar_vacio(eje)
        eje.set_title('Eficiencia por proceso (CPU)', fontsize=10)
        eje.grid(alpha=0.3)

        # 4) Escalabilidad: donde deja de compensar el coste de la GPU.
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
                color = COLOR_GPU if nombre == 'GPU' else (
                    COLOR_CPU if nombre.endswith('x1') else COLOR_CPU2)
                eje.plot([p[0] for p in puntos], [p[1] for p in puntos],
                         marker='o', label=nombre, color=color)
            eje.set_xlabel('tamano del archivo (MB)', fontsize=8)
            eje.set_ylabel('segundos', fontsize=8)
            eje.legend(fontsize=7)
        else:
            self._marcar_vacio(eje, 'Se llena con "Analisis completo"')
        eje.set_title('Escalabilidad: tiempo por tamano', fontsize=10)
        eje.grid(alpha=0.3)

        self.lienzo.draw()
        self._dibujar_recursos()

    def _dibujar_recursos(self):
        """Dibuja las graficas de uso de recursos y deteccion de errores."""
        for eje in self.ejes_rec:
            eje.clear()

        directas = [f for f in self.filas
                    if not f.get('escala') and not f.get('errores')]
        errores = [f for f in self.filas if f.get('errores')]

        def con_dato(filas, clave):
            return [f for f in filas if f.get(clave) not in ('', None)]

        # 1) Carga de CPU y de GPU por configuracion.
        eje = self.ejes_rec[0]
        datos = [f for f in directas
                 if f.get('cpu_medio') not in ('', None)
                 or f.get('gpu_medio') not in ('', None)]
        if datos:
            posiciones = range(len(datos))
            ancho = 0.4
            eje.bar([p - ancho / 2 for p in posiciones],
                    [f.get('cpu_medio') or 0 for f in datos],
                    ancho, label='CPU %', color=COLOR_CPU)
            eje.bar([p + ancho / 2 for p in posiciones],
                    [f.get('gpu_medio') or 0 for f in datos],
                    ancho, label='GPU %', color=COLOR_GPU)
            eje.set_xticks(list(posiciones))
            eje.set_xticklabels([f['etiqueta'] for f in datos],
                                rotation=45, ha='right', fontsize=7)
            eje.set_ylabel('uso medio %', fontsize=8)
            eje.legend(fontsize=7)
        else:
            self._marcar_vacio(eje, 'Se llena con "Analisis completo"')
        eje.set_title('Carga de CPU y GPU por configuracion', fontsize=10)
        eje.grid(axis='y', alpha=0.3)

        # 2) Memoria: la del propio programa frente a la de la tarjeta.
        #    Se grafica la memoria del proceso y sus hijos, no la del sistema
        #    entero: esta ultima ronda los 16 GB en este equipo y esconderia
        #    por completo los cientos de MB que es lo que realmente cuesta
        #    cada estrategia. Cada serie va en su propio eje porque miden
        #    cosas distintas y sus escalas no tienen por que coincidir.
        eje = self.ejes_rec[1]
        datos = con_dato(directas, 'proc_mb_max')
        if datos:
            posiciones = list(range(len(datos)))
            eje.plot(posiciones, [f['proc_mb_max'] for f in datos],
                     marker='o', label='RAM del programa', color=COLOR_CPU)
            eje.set_ylabel('RAM del programa (MB)', fontsize=8,
                           color=COLOR_CPU)
            eje.tick_params(axis='y', labelcolor=COLOR_CPU)
            eje.set_xticks(posiciones)
            eje.set_xticklabels([f['etiqueta'] for f in datos],
                                rotation=45, ha='right', fontsize=7)

            con_vram = [f.get('vram_mb_max') for f in datos]
            if any(v not in ('', None) for v in con_vram):
                if getattr(self, '_eje_vram', None) is not None:
                    self._eje_vram.remove()
                self._eje_vram = eje.twinx()
                self._eje_vram.plot(posiciones, [v or 0 for v in con_vram],
                                    marker='s', label='VRAM de la tarjeta',
                                    color=COLOR_GPU)
                self._eje_vram.set_ylabel('VRAM (MB)', fontsize=8,
                                          color=COLOR_GPU)
                self._eje_vram.tick_params(axis='y', labelcolor=COLOR_GPU)
        else:
            self._marcar_vacio(eje, 'Se llena con "Analisis completo"')
        eje.set_title('Memoria: programa frente a tarjeta', fontsize=10)
        eje.grid(alpha=0.3)

        # 3) Velocidad de identificacion de errores por segundo. Solo tiene
        #    sentido sobre los archivos con errores inyectados: en el archivo
        #    real no hay ninguno y la velocidad seria cero para todos.
        eje = self.ejes_rec[2]
        datos = [f for f in errores if f.get('invalidos')]
        if datos:
            tipos = []
            for fila in datos:
                if fila['tipo_error'] not in tipos:
                    tipos.append(fila['tipo_error'])
            ancho = 0.4
            posiciones = range(len(tipos))
            for desplazamiento, plataforma, color in (
                    (-ancho / 2, 'CPU', COLOR_CPU),
                    (ancho / 2, 'GPU', COLOR_GPU)):
                valores = []
                for tipo in tipos:
                    coincide = [f for f in datos
                                if f['tipo_error'] == tipo
                                and f['plataforma'] == plataforma]
                    valores.append(coincide[0]['invalidos_por_s']
                                   if coincide else 0)
                eje.bar([p + desplazamiento for p in posiciones], valores,
                        ancho, label=plataforma, color=color)
            eje.set_xticks(list(posiciones))
            eje.set_xticklabels(tipos, rotation=30, ha='right', fontsize=7)
            eje.set_ylabel('errores por segundo', fontsize=8)
            eje.legend(fontsize=7)
        else:
            self._marcar_vacio(eje, 'Se llena con "Analisis completo"')
        eje.set_title('Identificacion de errores por segundo', fontsize=10)
        eje.grid(axis='y', alpha=0.3)

        # 4) GPU discreta frente a integrada. Al medir tambien OpenCL sobre la
        #    tarjeta discreta quedan separadas las dos variables: entre las
        #    barras de OpenCL solo cambia el hardware, y entre las dos de la
        #    NVIDIA solo cambia la tecnologia.
        eje = self.ejes_rec[3]
        datos = [f for f in self.filas if f.get('dispositivos')]
        if datos:
            colores = [COLOR_GPU if f.get('tipo_gpu') == 'discreta'
                       else COLOR_CPU2 for f in datos]
            barras = eje.bar(range(len(datos)),
                             [f['mb_por_s'] for f in datos], color=colores)
            eje.set_xticks(range(len(datos)))
            eje.set_xticklabels([f['etiqueta'] for f in datos],
                                rotation=30, ha='right', fontsize=7)
            eje.set_ylabel('MB/s', fontsize=8)
            for barra, fila in zip(barras, datos):
                eje.text(barra.get_x() + barra.get_width() / 2,
                         barra.get_height(), '%.0f' % fila['mb_por_s'],
                         ha='center', va='bottom', fontsize=6)
        else:
            self._marcar_vacio(eje, 'Se llena con "Analisis completo"')
        eje.set_title('GPU discreta frente a integrada', fontsize=10)
        eje.grid(axis='y', alpha=0.3)

        self.lienzo_rec.draw()

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
        self.etiqueta_tamano.configure(
            text=benchmark_gpu.formato_tamano(tamano))
        self.estado.set('Archivo listo: %s'
                        % benchmark_gpu.formato_tamano(tamano))

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
        """Ejecuta el barrido completo de CPU, GPU y escalabilidad."""
        ruta = self._validar()
        if not ruta:
            return
        tamano = os.path.getsize(ruta)
        if tamano > 1024 ** 3:
            seguir = messagebox.askyesno(
                'Analisis completo',
                'El archivo ocupa %s. El analisis completo lanza varias\n'
                'pasadas sobre el y puede tardar bastantes minutos.\n\n'
                'Continuar?' % benchmark_gpu.formato_tamano(tamano))
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
            initialfile='benchmark_p12.csv',
            filetypes=[('CSV', '*.csv')])
        if not ruta:
            return
        benchmark_gpu.exportar_csv(self.filas, ruta)
        self.estado.set('Resultados exportados a %s' % os.path.basename(ruta))
        messagebox.showinfo('Exportado', 'Resultados guardados en:\n%s' % ruta)

    # ------------------------------------------------------------------
    # Hilo trabajador
    # ------------------------------------------------------------------

    def _lanzar(self, tarea, ruta):
        """Arranca la tarea en un hilo aparte y bloquea los botones.

        Antes de arrancar se copian los valores de los controles a atributos
        normales. Las variables de Tkinter son objetos de la biblioteca
        grafica y leerlas tambien cuenta como tocar la interfaz, asi que el
        hilo trabajador no debe llamar a su metodo get(): se queda con esta
        fotografia, tomada desde el hilo principal. Como efecto secundario
        util, la configuracion queda congelada al pulsar el boton y no cambia
        a mitad de un analisis si alguien mueve un selector.
        """
        self.config = {
            'modo': self.modo.get(),
            'procesos': max(1, self.procesos.get()),
            'lote_mb': self.lote_mb.get(),
        }

        self.trabajando = True
        self.inicio_proceso = time.perf_counter()
        self.boton_procesar.state(['disabled'])
        self.boton_analisis.state(['disabled'])
        self.barra['value'] = 0

        self.monitor = mod_monitor.Monitor()
        self.monitor.iniciar()

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
        lote = self.config['lote_mb']
        filas = []

        if modo in ('cpu', 'ambos'):
            self._estado('Procesando en CPU con %d proceso(s)...' % procesos)
            if procesos == 1:
                resultado, tiempo = motor_cpu.contar_secuencial(
                    ruta, self._progreso)
            else:
                resultado, tiempo = motor_cpu.contar_paralelo(
                    ruta, procesos, self._progreso)
            fila = benchmark_gpu.fila_base('CPU x%d' % procesos, 'CPU',
                                            resultado, tiempo, tamano)
            fila['procesos'] = procesos
            fila['lote_mb'] = ''
            filas.append(fila)
            self.cola.put(('resultado', resultado))
            self._log('CPU x%d: %.3f s (%.1f MB/s)'
                      % (procesos, tiempo, fila['mb_por_s']))

        if modo in ('gpu', 'ambos') and self.hay_gpu:
            self._estado('Compilando kernels CUDA...')
            jit = self.motor_gpu.precalentar()
            self._log('Compilacion JIT: %.3f s (fuera de la medicion)' % jit)

            self._estado('Procesando en GPU con lote de %d MB...' % lote)
            resultado, tiempo = self.motor_gpu.contar_gpu(
                ruta, lote_mb=lote, progreso=self._progreso)
            fila = benchmark_gpu.fila_base('GPU %d MB' % lote, 'GPU',
                                            resultado, tiempo, tamano)
            fila['procesos'] = ''
            fila['lote_mb'] = lote
            filas.append(fila)
            self.cola.put(('resultado', resultado))
            self._log('GPU lote %d MB: %.3f s (%.1f MB/s)'
                      % (lote, tiempo, fila['mb_por_s']))

        # Si se corrieron los dos, se comparan y se calcula el speedup.
        if len(filas) == 2:
            base = filas[0]['tiempo_s']
            # La CPU es la linea base, asi que su speedup es 1 por definicion.
            # Se deja explicito para que la grafica muestre las dos barras y
            # se vea contra que se esta comparando la GPU.
            filas[0]['speedup'] = 1.0
            filas[0]['eficiencia_pct'] = ''
            filas[1]['speedup'] = round(base / filas[1]['tiempo_s'], 3) \
                if filas[1]['tiempo_s'] > 0 else ''
            iguales = (filas[0]['bases'] == filas[1]['bases']
                       and filas[0]['invalidos'] == filas[1]['invalidos'])
            filas[0]['conteo_ok'] = filas[1]['conteo_ok'] = iguales
            self._log('Conteos CPU y GPU coincidentes: %s'
                      % ('si' if iguales else 'NO'))
            if filas[1]['tiempo_s'] > 0:
                self._log('La GPU fue %.2fx mas rapida que la CPU elegida'
                          % (base / filas[1]['tiempo_s']))

        self.cola.put(('filas', filas))

    def _tarea_completa(self, ruta):
        """Barrido completo: CPU, lote de GPU, escalabilidad y tipos de error.

        El muestreo de recursos se hace con el monitor que ya mantiene la
        ventana para su panel en directo. No se crea uno nuevo a proposito:
        dos monitores simultaneos se corromperian las lecturas de psutil, tal
        como se explica en monitor.Monitor.marcar.
        """
        self._estado('Analisis completo: barrido de CPU...')
        filas_cpu, referencia = benchmark_gpu.barrido_cpu(
            ruta, progreso=lambda t: self._estado('Analisis completo: ' + t),
            monitor=self.monitor)
        self.cola.put(('filas', filas_cpu))
        if filas_cpu:
            self.cola.put(('resultado', None))
            for fila in filas_cpu:
                self._log('%-12s %8.3f s  speedup %.2fx  eficiencia %.1f%%'
                          % (fila['etiqueta'], fila['tiempo_s'],
                             fila['speedup'], fila['eficiencia_pct']))

        # La CPU con un solo proceso es la linea base de todos los speedup
        # del analisis. Se calcula aqui, fuera del bloque de GPU, porque las
        # comparativas posteriores la necesitan aunque no haya tarjeta.
        base = filas_cpu[0]['tiempo_s'] if filas_cpu else None

        filas_gpu = []
        if self.hay_gpu:
            self._estado('Analisis completo: barrido de lote en GPU...')
            filas_gpu, referencia = benchmark_gpu.barrido_gpu(
                ruta, referencia=referencia, base_cpu=base,
                progreso=lambda t: self._estado('Analisis completo: ' + t),
                monitor=self.monitor)
            self.cola.put(('filas', filas_gpu))
            for fila in filas_gpu:
                self._log('%-14s %8.3f s  %8.1f MB/s'
                          % (fila['etiqueta'], fila['tiempo_s'],
                             fila['mb_por_s']))
            if filas_gpu:
                mejor = min(filas_gpu, key=lambda f: f['tiempo_s'])
                self._log('Mejor tamano de lote: %d MB' % mejor['lote_mb'])

        self._estado('Analisis completo: escalabilidad por tamano...')
        filas_esc = benchmark_gpu.escalabilidad(
            ruta, progreso=lambda t: self._estado('Analisis completo: ' + t))
        for fila in filas_esc:
            fila['escala'] = True
        self.cola.put(('filas', filas_esc))

        # Comparativa entre las unidades de computo del equipo: CUDA sobre la
        # tarjeta discreta, y OpenCL sobre la discreta y sobre la integrada.
        self._estado('Analisis completo: GPU discreta contra integrada...')
        filas_disp, referencia = benchmark_gpu.barrido_dispositivos(
            ruta, lote_mb=self.config['lote_mb'], referencia=referencia,
            base_cpu=base,
            progreso=lambda t: self._estado('Analisis completo: ' + t),
            monitor=self.monitor)
        self.cola.put(('filas', filas_disp))
        for fila in filas_disp:
            self._log('%-24s %-8s %8.3f s  %8.1f MB/s  (%d unidades)'
                      % (fila['etiqueta'], fila['tipo_gpu'], fila['tiempo_s'],
                         fila['mb_por_s'], fila['unidades']))

        # Escenarios con distintos tipos de error. Es la unica parte del
        # analisis que puede medir la velocidad de identificacion de errores
        # por segundo, porque el archivo real no contiene ningun caracter
        # invalido y sobre el esa velocidad seria siempre cero.
        self._estado('Analisis completo: escenarios de error...')
        filas_err = benchmark_gpu.barrido_tipos_error(
            ruta, progreso=lambda t: self._estado('Analisis completo: ' + t),
            monitor=self.monitor)
        self.cola.put(('filas', filas_err))
        for fila in filas_err:
            self._log('%-18s %8.3f s  %d de %d errores detectados  %s'
                      % (fila['etiqueta'], fila['tiempo_s'],
                         fila['invalidos'], fila['errores_inyectados'],
                         'correcto' if fila['deteccion_ok'] else 'FALLA'))
        fallos = [f for f in filas_err if not f['deteccion_ok']]
        if fallos:
            self._log('ATENCION: %d escenario(s) de error fallaron'
                      % len(fallos))
        elif filas_err:
            self._log('Los %d escenarios de error se detectaron exactamente.'
                      % len(filas_err))

        if referencia is not None:
            self.cola.put(('resultado', referencia))

        malos = [f for f in filas_cpu + filas_gpu
                 if not f.get('conteo_ok', True)]
        if malos:
            self._log('ATENCION: %d configuracion(es) con conteo distinto'
                      % len(malos))
        else:
            self._log('Todas las configuraciones dieron el mismo conteo.')

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

                elif tipo == 'resultado':
                    if dato is not None:
                        self._mostrar_conteo(dato)

                elif tipo == 'filas':
                    self.filas.extend(dato)
                    self._rellenar_tabla()
                    self._dibujar_graficas()

                elif tipo == 'error':
                    messagebox.showerror('Error durante el procesamiento', dato)
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
        gigabytes sin dejar procesos hijos sueltos. Al confirmar se detiene el
        monitor y se marca el cierre para que los bucles de after no se
        vuelvan a encolar.
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
        if self.monitor:
            self.monitor.detener()
            self.monitor = None
        if self.inicio_proceso:
            total = time.perf_counter() - self.inicio_proceso
            self.estado.set('Terminado en %.2f s' % total)
            self.inicio_proceso = None

    def _mostrar_conteo(self, resultado):
        for clave, etiqueta in self.etiquetas_conteo.items():
            etiqueta.configure(text='{:,}'.format(resultado.get(clave, 0)))

        self.texto_invalidos.configure(state='normal')
        self.texto_invalidos.delete('1.0', 'end')

        if resultado.get('detalle_iupac'):
            self.texto_invalidos.insert(
                'end', 'Codigos IUPAC (validos, no son errores):\n')
            for caracter, veces in sorted(resultado['detalle_iupac'].items(),
                                          key=lambda x: -x[1]):
                self.texto_invalidos.insert('end', '  %s = %d\n'
                                            % (caracter, veces))
        if resultado.get('detalle'):
            self.texto_invalidos.insert('end', 'Invalidos:\n')
            for caracter, veces in sorted(resultado['detalle'].items(),
                                          key=lambda x: -x[1]):
                self.texto_invalidos.insert('end', '  %r = %d\n'
                                            % (caracter, veces))
        elif not resultado.get('detalle_iupac'):
            self.texto_invalidos.insert(
                'end', 'No se encontro ningun caracter invalido.')

        self.texto_invalidos.configure(state='disabled')
        self.ultimo_resultado = resultado

    def _rellenar_tabla(self):
        def numero(fila, clave, formato, sufijo=''):
            """Formatea un valor, o un guion si no se llego a medir.

            Distinguir el cero del no medido importa: en las ejecuciones de
            GPU mas cortas que el intervalo de muestreo no hay lecturas de
            recursos, y escribir 0 ahi haria pensar que la tarjeta no se uso.
            """
            valor = fila.get(clave)
            if valor in ('', None):
                return '-'
            return (formato % valor) + sufijo

        for elemento in self.tabla.get_children():
            self.tabla.delete(elemento)

        for fila in self.filas:
            self.tabla.insert('', 'end', values=(
                fila.get('etiqueta', ''),
                fila.get('plataforma', ''),
                '%.3f' % fila.get('tiempo_s', 0),
                '%.1f' % fila.get('mb_por_s', 0),
                numero(fila, 'speedup', '%.2f', 'x'),
                numero(fila, 'eficiencia_pct', '%.1f'),
                '{:,}'.format(fila.get('invalidos', 0)),
                ('{:,.0f}'.format(fila['invalidos_por_s'])
                 if fila.get('invalidos') and
                 fila.get('invalidos_por_s') not in ('', None) else '-'),
                numero(fila, 'cpu_medio', '%.0f'),
                numero(fila, 'gpu_medio', '%.0f'),
                numero(fila, 'vram_mb_max', '%d'),
                numero(fila, 'gpu_temp_max', '%d'),
                'si' if fila.get('conteo_ok', True) else 'NO',
            ))

    # ------------------------------------------------------------------
    # Panel de recursos en vivo
    # ------------------------------------------------------------------

    def _refrescar_recursos(self):
        """Actualiza las barras de uso de CPU, RAM y GPU cada medio segundo."""
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

        if 'vram_mb' in muestra and self.hay_gpu:
            total = max(1, self.datos_gpu.get('vram_total_mb', 1))
            poner('vram', 100.0 * muestra['vram_mb'] / total,
                  '%d MB' % muestra['vram_mb'])
        else:
            poner('vram', None, '-')

        if 'gpu_temp' in muestra:
            # La escala de la barra va de 0 a 100 grados, que cubre de sobra
            # el rango util de una GPU de portatil.
            poner('temp', muestra['gpu_temp'], '%d C' % muestra['gpu_temp'])
        else:
            poner('temp', None, '-')

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
