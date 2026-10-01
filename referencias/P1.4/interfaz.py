# -*- coding: utf-8 -*-
"""
interfaz.py

Ventana de la aplicacion del P1.4: comparacion de dos cadenas de ADN sobre
CPU, GPU y NPU.

Se arranca desde app_p14.py, nunca directamente, porque ese archivo es el que
pone el guard de multiprocessing que Windows exige.

ESTRUCTURA
  - Columna izquierda: archivos, plataforma, parametros, hardware detectado,
                       resultado y recursos en vivo.
  - Columna derecha  : pestanas con las diferencias, las graficas, la matriz
                       de evidencias y el registro.
  - Pie              : barra de progreso y estado.

CONCURRENCIA
Todo el procesamiento ocurre en un hilo aparte. Tkinter no es seguro para
usarse desde varios hilos, asi que el hilo trabajador nunca toca un widget:
deja mensajes en una cola y la ventana la vacia cada 100 ms desde el hilo
principal con after(). Es el unico patron correcto aqui, y ademas es lo que
mantiene la ventana viva mientras se comparan 3 GB.

A PRUEBA DE USUARIO
Esta ventana asume que quien la usa va a equivocarse, y que equivocarse no
debe romper nada ni producir un resultado enganoso. Las defensas son:

  1. Los campos numericos solo aceptan digitos. No se puede teclear una
     letra, un signo menos ni un punto: el propio campo los rechaza segun se
     escriben, de modo que el error no llega ni a existir.
  2. Ademas del filtro de teclado, cada valor se vuelve a validar contra el
     hardware antes de usarlo. Pedir 16 procesos en un equipo de 12 nucleos
     no falla: se ajusta a 12 y se explica por que en el registro.
  3. Los archivos se comprueban antes de leer un solo byte: que existan, que
     no sean carpetas, que no esten vacios, que no sean binarios y que
     parezcan FASTA. Un archivo de 3 GB equivocado costaria minutos en
     descubrirse a mitad de la lectura.
  4. Los botones se desactivan mientras hay trabajo en curso, asi que no se
     puede lanzar dos veces ni cambiar los parametros a mitad.
  5. Cerrar la ventana con un trabajo en marcha pide confirmacion, porque
     hay procesos hijos que hay que terminar de forma ordenada.
  6. Cualquier fallo previsto se muestra en un cuadro de dialogo con texto
     escrito para una persona, no con la traza de la excepcion.

Todo lo que se ajusta solo queda escrito en la pestana de registro: el
programa nunca cambia un parametro en silencio.
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

import comparador
import evidencias
import mezclador
import monitor as mod_monitor
import motor_cpu
import secuencias
from comparador import ErrorEntrada


# ---------------------------------------------------------------------------
# Tokens de diseño
#
# Tkinter no tiene hoja de estilos, asi que la escala tipografica y la paleta
# se declaran aqui y no se repiten por el codigo. La escala sube al menos un
# cuarto de paso a paso, o cambia de peso: es lo que hace que se distinga de
# un vistazo una etiqueta de seccion de la cifra del resultado.
# ---------------------------------------------------------------------------

FUENTE = 'Segoe UI'
F_TITULO = (FUENTE, 13, 'bold')
F_SECCION = (FUENTE, 8, 'bold')
F_CUERPO = (FUENTE, 9)
F_AYUDA = (FUENTE, 8)
F_CIFRA = (FUENTE, 22, 'bold')
F_MONO = ('Consolas', 9)
F_MONO_MIN = ('Consolas', 8)

# Grises con contraste suficiente sobre el fondo: el texto de cuerpo pasa de
# 4.5 a 1 y el secundario tambien. Un gris claro "por elegancia" es la razon
# mas comun de que una interfaz se lea mal.
TINTA = '#1c1c1e'
TINTA_SUAVE = '#55555c'
TINTA_TENUE = '#85858c'
LINEA = '#dcdce1'
FONDO = '#f4f4f6'
PAPEL = '#ffffff'

# Un color por plataforma, usado igual en la ventana y en las graficas. El
# color significa algo: no es decoracion.
COLOR_CPU = '#2f6f9f'
COLOR_GPU = '#3f8f3a'
COLOR_NPU = '#b07d1a'
COLOR_AVISO = '#b03030'
COLOR_BIEN = '#2e7d32'

COLOR_PLATAFORMA = {'CPU': COLOR_CPU, 'GPU': COLOR_GPU, 'NPU': COLOR_NPU}

TITULOS_RENDIMIENTO = (
    'Tiempo por plataforma',
    'Velocidad de comparacion',
    'Preparacion frente a trabajo',
    'Donde caen las diferencias',
)

TITULOS_RECURSOS = (
    'Uso de cada nucleo logico de CPU',
    'Uso de cada plataforma',
    'Temperatura por plataforma',
    'Memoria usada por plataforma',
)

INTERVALO_COLA = 100        # ms entre revisiones de la cola de mensajes
INTERVALO_RECURSOS = 500    # ms entre refrescos del panel de recursos

# Cuantas diferencias se llevan a la tabla. Mostrar mas no aporta y llenar un
# Treeview con cien mil filas congela la ventana varios segundos.
MAX_TABLA = 500

# Tamano del trozo con el que el motor de CPU reparte el conteo. Se usa para
# situar cada barra del mapa de diferencias en su posicion del archivo.
_TROZO = 64 << 20

# Repeticiones de cada medicion cuando se pide medicion justa. Se descarta
# una corrida de calentamiento y se publica la MEDIANA de las demas.
#
# POR QUE HACE FALTA. Las plataformas corren una detras de otra sobre los
# mismos archivos mapeados. La primera paga las fallas de pagina y lee del
# disco; las siguientes encuentran los datos en la cache del sistema. Medido
# en este proyecto: la CPU salia a 429 MB/s en la primera pasada y a 2000
# MB/s en la segunda sobre los mismos datos. Comparar asi no mide el
# hardware, mide el orden en que se ejecutaron.
REPETICIONES = 3


def _corto(nombre):
    """Acorta el nombre comercial de una tarjeta para que quepa en el panel."""
    limpio = (nombre or '').replace('NVIDIA GeForce ', '').replace('(R)', '')
    return limpio.replace(' Laptop GPU', '').replace('Graphics', 'Gfx').strip()


def _miles(numero):
    """Escribe un entero con puntos de millar, como se hace en castellano.

    El formato ',' de Python usa la coma anglosajona, que en un informe en
    espanol se lee como separador decimal y confunde las cifras grandes.
    """
    return '{:,}'.format(int(numero)).replace(',', '.')


def _entero(texto, por_defecto):
    """Convierte a entero lo que venga de un campo, sin romperse nunca.

    El filtro de teclado ya impide escribir cualquier cosa que no sean
    digitos, pero el valor puede llegar vacio (hace falta poder borrar el
    campo para teclear otro numero) y ademas el diccionario de configuracion
    podria venir de otro sitio, como capturas.py. Un int() suelto sobre una
    cadena vacia aborta; esto devuelve el valor por defecto y sigue.

    El rango no se comprueba aqui: de eso se encarga cada motor con su propia
    funcion de validacion, que ademas explica el ajuste.
    """
    try:
        return int(str(texto).strip())
    except (TypeError, ValueError):
        return por_defecto


class Aplicacion:
    """Ventana principal de la aplicacion."""

    def __init__(self, raiz):
        self.raiz = raiz
        raiz.title('P1.4 - Comparacion de cadenas de ADN en CPU, GPU y NPU')

        # La pantalla del equipo es de 1536x864 logicos. Una ventana de mas
        # de 800 px de alto no cabe, asi que se calcula contra la pantalla
        # real en vez de fijarla a ojo.
        self.ancho = min(1450, raiz.winfo_screenwidth() - 60)
        self.alto = min(900, raiz.winfo_screenheight() - 140)
        raiz.geometry('%dx%d' % (self.ancho, self.alto))
        raiz.minsize(1100, 640)
        raiz.protocol('WM_DELETE_WINDOW', self.cerrar)

        self.cola = queue.Queue()
        self.trabajando = False
        self.hilo = None
        self.cerrando = False

        self.ruta_a = tk.StringVar()
        self.ruta_b = tk.StringVar()
        self.modo = tk.StringVar(value='todas')
        self.procesos = tk.StringVar()
        self.bloques_gpu = tk.StringVar(value='1024')
        self.lote_gpu = tk.StringVar(value='64')
        self.capas_npu = tk.StringVar(value='8')
        self.por_secuencia = tk.BooleanVar(value=True)
        self.medicion_justa = tk.BooleanVar(value=True)
        self.mezclar = tk.BooleanVar(value=True)
        self.mezclar_gpu = tk.BooleanVar(value=False)
        self.mezclar_hibrido = tk.BooleanVar(value=False)
        self.estado = tk.StringVar(value='Listo. Elige las dos cadenas.')
        self.avance = tk.DoubleVar(value=0.0)

        self.cadena_a = None
        self.cadena_b = None
        self.resultados = []
        self.ultimo_npu = {}
        self.ultimo_gpu = {}
        self.por_crom = {}          # plataforma -> resumen por secuencia
        self.sin_pareja = ([], [])
        self.parejas = []

        self._detectar_hardware()
        self.procesos.set(str(self.logicos))

        self._construir()

        # Un unico monitor vivo para toda la aplicacion. Dos monitores a la
        # vez se corromperian la referencia interna de psutil.cpu_percent,
        # que es global al proceso.
        self.monitor = mod_monitor.Monitor().iniciar()

        self.raiz.after(INTERVALO_COLA, self._revisar_cola)
        self.raiz.after(INTERVALO_RECURSOS, self._refrescar_recursos)
        self._rellenar_evidencias()

    # -- hardware ---------------------------------------------------------

    def _detectar_hardware(self):
        """Averigua que hay en esta maquina, sin que un fallo tumbe la app."""
        self.fisicos, self.logicos = motor_cpu.detectar_nucleos()

        self.info_gpu = {'disponible': False}
        self.recursos_kernel = None
        self.bloques_gpu_rango = (1, 1024)
        try:
            import motor_gpu
            self.bloques_gpu_rango = (motor_gpu.BLOQUES_MIN,
                                      motor_gpu.BLOQUES_MAX)
            self.info_gpu = motor_gpu.info_gpu()
        except Exception as error:
            self.info_gpu = {'disponible': False, 'motivo': str(error)}
        self.hay_gpu = bool(self.info_gpu.get('disponible'))

        self.info_npu = {'disponible': False}
        try:
            import motor_npu
            self.info_npu = motor_npu.info_npu()
        except Exception as error:
            self.info_npu = {'disponible': False, 'motivo': str(error)}
        self.hay_npu = bool(self.info_npu.get('disponible'))
        self.npu_real = bool(self.info_npu.get('es_npu'))

    # -- construccion de la ventana ---------------------------------------

    def _estilos(self):
        """Define la escala tipografica y los colores de la ventana.

        Tkinter no tiene hoja de estilos, asi que los valores se centralizan
        aqui en vez de repartirse por el codigo. La escala sube de paso en
        paso lo bastante como para que se note: una etiqueta de seccion y la
        cifra del resultado no pueden pesar lo mismo, que es lo que pasaba
        cuando todo estaba a ocho puntos.
        """
        estilo = ttk.Style()
        if 'clam' in estilo.theme_names():
            estilo.theme_use('clam')

        estilo.configure('.', background=FONDO, foreground=TINTA)
        estilo.configure('TFrame', background=FONDO)
        estilo.configure('TLabel', background=FONDO, foreground=TINTA,
                         font=F_CUERPO)
        estilo.configure('Papel.TFrame', background=PAPEL)

        estilo.configure('Titulo.TLabel', font=F_TITULO, foreground=TINTA)
        estilo.configure('Seccion.TLabel', font=F_SECCION,
                         foreground=TINTA_TENUE)
        estilo.configure('Ayuda.TLabel', font=F_AYUDA,
                         foreground=TINTA_SUAVE)
        estilo.configure('Dato.TLabel', font=F_MONO, foreground=TINTA)
        estilo.configure('Cifra.TLabel', font=F_CIFRA, foreground=TINTA,
                         background=PAPEL)
        estilo.configure('CifraPie.TLabel', font=F_CUERPO,
                         foreground=TINTA_SUAVE, background=PAPEL)

        estilo.configure('TButton', font=F_CUERPO, padding=(12, 6))
        estilo.configure('Principal.TButton', font=(FUENTE, 9, 'bold'),
                         padding=(16, 7))
        estilo.configure('TRadiobutton', background=FONDO, font=F_CUERPO)
        estilo.configure('TNotebook', background=FONDO, borderwidth=0)
        estilo.configure('TNotebook.Tab', font=F_CUERPO, padding=(14, 7))
        estilo.configure('Treeview', font=F_MONO, rowheight=22,
                         fieldbackground=PAPEL, background=PAPEL)
        estilo.configure('Treeview.Heading', font=(FUENTE, 8, 'bold'))
        estilo.configure('TProgressbar', background=COLOR_CPU,
                         troughcolor='#e4e4e8', borderwidth=0,
                         lightcolor=COLOR_CPU, darkcolor=COLOR_CPU)
        self.raiz.configure(background=FONDO)

    def _seccion(self, padre, texto, primera=False):
        """Rotulo de seccion con una linea debajo.

        Sustituye a los LabelFrame que habia antes. Cinco cajas apiladas, y
        ademas metidas dentro de otro marco, encerraban el contenido sin
        aportar nada: el rotulo mas una linea separa igual de bien y deja
        respirar la columna.
        """
        marco = ttk.Frame(padre)
        marco.pack(fill='x', pady=((0 if primera else 12), 6))
        ttk.Label(marco, text=texto.upper(),
                  style='Seccion.TLabel').pack(anchor='w')
        tk.Frame(marco, height=1, background=LINEA).pack(fill='x',
                                                         pady=(4, 0))
        return marco

    def _construir(self):
        self._estilos()
        self._cabecera(self.raiz)

        # El pie se coloca ANTES que el cuerpo a proposito. En Tk, pack
        # reparte el espacio por orden de llegada: si el cuerpo pide primero
        # todo lo que sobra (expand=True), al pie no le queda nada y la barra
        # de progreso desaparece por debajo del borde de la ventana.
        self._panel_pie(self.raiz)

        cuerpo = ttk.Frame(self.raiz, padding=(16, 12, 16, 6))
        cuerpo.pack(fill='both', expand=True)

        izquierda = ttk.Frame(cuerpo, width=330)
        izquierda.pack(side='left', fill='y', padx=(0, 18))
        izquierda.pack_propagate(False)

        derecha = ttk.Frame(cuerpo)
        derecha.pack(side='left', fill='both', expand=True)

        self._panel_archivos(izquierda)
        self._panel_configuracion(izquierda)
        self._panel_resultado(izquierda)
        self._panel_recursos(izquierda)

        self._panel_pestanas(derecha)

    def _cabecera(self, padre):
        """Barra superior con el nombre y el hardware en una sola linea.

        El hardware estaba antes en su propia caja, ocupando un tercio de la
        columna izquierda para tres datos que no cambian nunca. Aqui se lee
        de un vistazo y libera ese espacio para lo que si cambia.
        """
        barra = tk.Frame(padre, background=PAPEL)
        barra.pack(fill='x')
        interior = ttk.Frame(barra, style='Papel.TFrame', padding=(16, 11))
        interior.pack(fill='x')

        tk.Label(interior, text='Comparacion de cadenas de ADN',
                 background=PAPEL, foreground=TINTA,
                 font=F_TITULO).pack(side='left')
        tk.Label(interior, text='P1.4  ·  CPU, GPU y NPU', background=PAPEL,
                 foreground=TINTA_TENUE,
                 font=F_AYUDA).pack(side='left', padx=(10, 0), pady=(5, 0))

        # El hardware va a la derecha, cada pieza con el color de su
        # plataforma: asi el color significa algo tambien en la ventana y no
        # solo en las graficas.
        piezas = [('CPU  %d/%d nucleos' % (self.fisicos, self.logicos),
                   COLOR_CPU, True)]
        if self.hay_gpu:
            piezas.append(('GPU  %s' % (_corto(self.info_gpu.get('nombre'))
                                        or 'CUDA'), COLOR_GPU, True))
        else:
            piezas.append(('sin GPU CUDA', TINTA_TENUE, False))
        if self.npu_real:
            piezas.append(('NPU  %s' % self.info_npu.get('descripcion'),
                           COLOR_NPU, True))
        elif self.hay_npu:
            piezas.append(('sin NPU  ·  %s'
                           % self.info_npu.get('descripcion'),
                           TINTA_TENUE, False))
        else:
            piezas.append(('sin NPU', TINTA_TENUE, False))

        for texto, color, activa in reversed(piezas):
            tk.Label(interior, text=texto, background=PAPEL,
                     foreground=color,
                     font=(FUENTE, 8, 'bold' if activa else 'normal')
                     ).pack(side='right', padx=(16, 0))

        tk.Frame(padre, height=1, background=LINEA).pack(fill='x')

    def _panel_archivos(self, padre):
        self._seccion(padre, 'Cadenas a comparar', primera=True)

        self.etiquetas_archivo = {}
        for cual, color in (('A', COLOR_CPU), ('B', COLOR_GPU)):
            fila = ttk.Frame(padre)
            fila.pack(fill='x', pady=3)

            tk.Label(fila, text=cual, width=2, background=color,
                     foreground='white', font=(FUENTE, 9, 'bold')
                     ).pack(side='left', ipady=3)

            ttk.Button(fila, text='Elegir', width=8,
                       command=(lambda c=cual: self._elegir(c))
                       ).pack(side='right', padx=(6, 0))

            nombre = ttk.Label(fila, text='(sin elegir)',
                               style='Ayuda.TLabel', anchor='w')
            nombre.pack(side='left', fill='x', expand=True, padx=(8, 0))
            self.etiquetas_archivo[cual] = nombre

        self.etiqueta_par = ttk.Label(padre, text='', style='Ayuda.TLabel',
                                      wraplength=310, justify='left')
        self.etiqueta_par.pack(fill='x', pady=(7, 0))

    def _panel_configuracion(self, padre):
        self._seccion(padre, 'Plataforma')

        fila = ttk.Frame(padre)
        fila.pack(fill='x', pady=(0, 10))
        for texto, valor in (('CPU', 'cpu'), ('GPU', 'gpu'),
                             ('NPU', 'npu'), ('Las tres', 'todas')):
            boton = ttk.Radiobutton(fila, text=texto, value=valor,
                                    variable=self.modo)
            boton.pack(side='left', padx=(0, 11))
            # Una plataforma que no existe en este equipo no se puede
            # elegir. Desactivar el boton es mejor que dejar que lo pulse y
            # darle un error despues.
            if valor == 'gpu' and not self.hay_gpu:
                boton.state(['disabled'])
            if valor == 'npu' and not self.hay_npu:
                boton.state(['disabled'])

        # Cada campo valida en el teclado su propio rango: ver _fila_numero.
        self.spin_procesos = self._fila_numero(
            padre, 'Procesos de CPU', self.procesos, 1, self.logicos)
        self.spin_bloques = self._fila_numero(
            padre, 'Bloques de GPU', self.bloques_gpu,
            *self.bloques_gpu_rango)
        self.spin_lote = self._fila_numero(
            padre, 'Lote de GPU (MB)', self.lote_gpu, 1, 512)
        if not self.hay_gpu:
            self.spin_bloques.state(['disabled'])
            self.spin_lote.state(['disabled'])
        self.spin_capas = self._fila_numero(
            padre, 'Capas de NPU', self.capas_npu, 1, 22)
        if not self.hay_npu:
            self.spin_capas.state(['disabled'])

        ttk.Checkbutton(
            padre, text='Emparejar por secuencia', variable=self.por_secuencia
        ).pack(anchor='w', pady=(8, 0))
        ttk.Label(padre, text='compara cromosoma con cromosoma',
                  style='Ayuda.TLabel').pack(anchor='w', padx=(20, 0))

        ttk.Checkbutton(
            padre, text='Medicion justa', variable=self.medicion_justa
        ).pack(anchor='w', pady=(6, 0))
        ttk.Label(padre, text='calentamiento y mediana de %d' % REPETICIONES,
                  style='Ayuda.TLabel').pack(anchor='w', padx=(20, 0))

        ttk.Checkbutton(
            padre, text='Generar archivo mezclado', variable=self.mezclar
        ).pack(anchor='w', pady=(6, 0))
        ttk.Label(padre, text='fila de A y fila de B alternadas, con la CPU',
                  style='Ayuda.TLabel').pack(anchor='w', padx=(20, 0))

        casilla_gpu = ttk.Checkbutton(
            padre, text='Generar archivo mezclado con GPU',
            variable=self.mezclar_gpu)
        casilla_gpu.pack(anchor='w', pady=(6, 0))
        if not self.hay_gpu:
            casilla_gpu.state(['disabled'])
        ttk.Label(padre, text='lo mismo, armado en la tarjeta (..._GPU.txt)',
                  style='Ayuda.TLabel').pack(anchor='w', padx=(20, 0))

        casilla_hibrido = ttk.Checkbutton(
            padre, text='Generar archivo mezclado con CPU + GPU',
            variable=self.mezclar_hibrido)
        casilla_hibrido.pack(anchor='w', pady=(6, 0))
        if not self.hay_gpu:
            casilla_hibrido.state(['disabled'])
        ttk.Label(padre, text='las dos a la vez, reparto por lotes '
                              '(..._HIBRIDO.txt)',
                  style='Ayuda.TLabel').pack(anchor='w', padx=(20, 0))

        botones = ttk.Frame(padre)
        botones.pack(fill='x', pady=(12, 0))
        self.boton_comparar = ttk.Button(botones, text='Comparar cadenas',
                                         style='Principal.TButton',
                                         command=self.comparar)
        self.boton_comparar.pack(side='left')
        self.boton_exportar = ttk.Button(botones, text='Exportar',
                                         command=self.exportar,
                                         state='disabled')
        self.boton_exportar.pack(side='left', padx=(8, 0))

    def _fila_numero(self, padre, etiqueta, variable, minimo, maximo):
        """Crea una fila con un selector numerico ya validado.

        Validacion de teclado: se registra una funcion de Tk que recibe el
        texto que quedaria si se aceptara la pulsacion y devuelve False para
        rechazarla. Asi no se puede teclear una letra, un signo ni un numero
        fuera de [minimo, maximo]: con 12 nucleos, escribir 100 procesos se
        queda en 10, porque el segundo cero no entra. El error no llega a
        existir, en vez de detectarse despues.
        """
        fila = ttk.Frame(padre)
        fila.pack(fill='x', pady=3)
        ttk.Label(fila, text=etiqueta).pack(side='left')
        ttk.Label(fila, text='%d a %d' % (minimo, maximo),
                  style='Ayuda.TLabel').pack(side='right', padx=(8, 0))
        validacion = (self.raiz.register(
            lambda texto: self._numero_en_rango(texto, minimo, maximo)), '%P')
        control = ttk.Spinbox(fila, from_=minimo, to=maximo, width=5,
                              textvariable=variable, validate='key',
                              validatecommand=validacion, font=F_CUERPO)
        control.pack(side='right')
        return control

    @staticmethod
    def _numero_en_rango(texto, minimo, maximo):
        """Acepta la edicion solo si deja un entero dentro del rango, o nada.

        Pasarse del maximo se rechaza siempre: anadir cifras solo lo haria
        mas grande. Quedarse por debajo del minimo solo se rechaza cuando el
        numero ya tiene tantas cifras como el minimo; antes puede ser un paso
        intermedio (para escribir 15 con minimo 10 hay que pasar por 1). Con
        minimo 1, que es el caso de todos los campos, rechaza el 0.
        """
        if not Aplicacion._solo_digitos(texto):
            return False
        if texto == '':
            return True
        valor = int(texto)
        if valor > maximo:
            return False
        if valor < minimo and len(texto) >= len(str(minimo)):
            return False
        return True

    @staticmethod
    def _solo_digitos(texto):
        """Acepta la edicion solo si lo que queda son digitos o nada.

        Se permite el campo vacio porque hay que poder borrarlo para
        escribir otro numero; el valor vacio se resuelve al validar, no al
        teclear. Se rechaza todo lo demas: letras, signos, puntos y
        espacios.
        """
        if texto == '':
            return True
        if not texto.isdigit():
            return False
        # Un numero absurdamente largo se rechaza en el teclado para que no
        # llegue a convertirse en un entero gigante.
        return len(texto) <= 6

    def _panel_resultado(self, padre):
        self._seccion(padre, 'Resultado')

        tarjeta = tk.Frame(padre, background=PAPEL, highlightthickness=1,
                           highlightbackground=LINEA)
        tarjeta.pack(fill='x')
        interior = ttk.Frame(tarjeta, style='Papel.TFrame',
                             padding=(13, 9))
        interior.pack(fill='x')

        # La cifra de diferencias es el dato por el que existe el programa,
        # asi que se lleva el peso tipografico. Antes iba en el mismo cuerpo
        # de ocho puntos que las etiquetas y se perdia entre ellas.
        self.cifra = ttk.Label(interior, text='—', style='Cifra.TLabel')
        self.cifra.pack(anchor='w')
        self.cifra_pie = ttk.Label(interior, text='sin comparar todavia',
                                   style='CifraPie.TLabel')
        self.cifra_pie.pack(anchor='w', pady=(0, 6))

        # Una linea por plataforma, no solo la ultima. Antes el panel se
        # sobrescribia con cada resultado y en modo "Las tres" solo se veia
        # la NPU, que es justo cuando mas interesa comparar las tres.
        self.detalle_resultado = tk.Label(
            interior, text='', background=PAPEL, foreground=TINTA_SUAVE,
            font=F_MONO_MIN, justify='left', anchor='w')
        self.detalle_resultado.pack(anchor='w', fill='x')

        self.acuerdo = tk.Label(interior, text='', background=PAPEL,
                                font=(FUENTE, 8, 'bold'), anchor='w')
        self.acuerdo.pack(anchor='w', fill='x', pady=(6, 0))

    def _panel_recursos(self, padre):
        self._seccion(padre, 'Recursos en vivo')

        # Dos columnas: las ocho magnitudes en una sola no caben en una
        # ventana de 780 px de alto, que es el techo de esta pantalla, y las
        # de abajo quedaban recortadas por el borde.
        self.recursos = {}
        magnitudes = (('cpu', 'CPU', COLOR_CPU), ('gpu', 'GPU', COLOR_GPU),
                      ('cpu_temp', 'T. CPU', COLOR_CPU),
                      ('gpu_temp', 'T. GPU', COLOR_GPU),
                      ('ram', 'RAM', TINTA_SUAVE),
                      ('vram_mb', 'VRAM', COLOR_GPU),
                      ('proc_mb', 'Programa', TINTA_SUAVE),
                      ('npu', 'NPU', COLOR_NPU))

        rejilla = ttk.Frame(padre)
        rejilla.pack(fill='x')
        rejilla.columnconfigure(0, weight=1)
        rejilla.columnconfigure(1, weight=1)
        for indice, (clave, etiqueta, color) in enumerate(magnitudes):
            fila, columna = divmod(indice, 2)
            celda = ttk.Frame(rejilla)
            celda.grid(row=fila, column=columna, sticky='ew', pady=2)
            tk.Label(celda, text=etiqueta, background=FONDO,
                     foreground=color, font=(FUENTE, 8), width=8,
                     anchor='w').pack(side='left')
            valor = ttk.Label(celda, text='—', style='Dato.TLabel')
            valor.pack(side='left')
            self.recursos[clave] = valor

    def _panel_pestanas(self, padre):
        self.pestanas = ttk.Notebook(padre)
        self.pestanas.pack(fill='both', expand=True)

        # -- diferencias --
        marco_dif = ttk.Frame(self.pestanas, padding=(0, 8, 0, 0))
        self.pestanas.add(marco_dif, text='Diferencias')
        columnas = ('n', 'posicion', 'fila', 'columna', 'a', 'b')
        titulos = ('#', 'Posicion', 'Fila', 'Columna', 'Cadena A',
                   'Cadena B')
        anchos = (60, 140, 120, 90, 90, 90)
        self.tabla = ttk.Treeview(marco_dif, columns=columnas,
                                  show='headings', height=24)
        for columna, titulo, ancho in zip(columnas, titulos, anchos):
            self.tabla.heading(columna, text=titulo)
            self.tabla.column(columna, width=ancho, anchor='center')
        self.tabla.pack(side='left', fill='both', expand=True)
        barra = ttk.Scrollbar(marco_dif, orient='vertical',
                              command=self.tabla.yview)
        barra.pack(side='right', fill='y')
        self.tabla.configure(yscrollcommand=barra.set)

        # Tamano de partida deliberadamente corto. Un lienzo de Tk no
        # conoce su tamano real hasta que su pestana se muestra por primera
        # vez: hasta entonces mide un pixel. Si la figura naciera con el
        # tamano final, se dibujaria recortada por la derecha y por abajo
        # hasta que alguien cambiase de pestana. Naciendo pequena, el
        # redibujado de _al_cambiar_pestana la estira al hueco real.
        ancho_figura = 7.2
        alto_figura = 4.4

        # -- rendimiento --
        marco_rend = ttk.Frame(self.pestanas)
        self.pestanas.add(marco_rend, text='Rendimiento')
        self.figura = Figure(figsize=(ancho_figura, alto_figura), dpi=100,
                             facecolor=FONDO)
        self.ejes = [self.figura.add_subplot(2, 2, i + 1) for i in range(4)]
        self.lienzo = FigureCanvasTkAgg(self.figura, master=marco_rend)
        self.lienzo.get_tk_widget().pack(fill='both', expand=True)

        # -- nucleos y recursos --
        # Es la pestana que responde a los criterios 4, 5 y 6 de la rubrica,
        # y la que hay que fotografiar para la captura de "uso de recursos"
        # que pide el enunciado.
        marco_nuc = ttk.Frame(self.pestanas)
        self.pestanas.add(marco_nuc, text='Nucleos y recursos')
        self.figura_rec = Figure(figsize=(ancho_figura, alto_figura),
                                 dpi=100, facecolor=FONDO)
        self.ejes_rec = [self.figura_rec.add_subplot(2, 2, i + 1)
                         for i in range(4)]
        self.lienzo_rec = FigureCanvasTkAgg(self.figura_rec,
                                            master=marco_nuc)
        self.lienzo_rec.get_tk_widget().pack(fill='both', expand=True)

        # -- por secuencia --
        # Compara cromosoma con cromosoma en vez de posicion con posicion.
        # Es la pestana que demuestra que los dos ensamblajes son el mismo:
        # de corrido dan mil millones de diferencias, emparejados dan cero.
        marco_sec = ttk.Frame(self.pestanas, padding=(0, 8, 0, 0))
        self.pestanas.add(marco_sec, text='Por secuencia')
        columnas = ('nombre', 'criterio', 'inicio_a', 'inicio_b', 'largo',
                    'dif')
        titulos = ('Secuencia', 'Emparejada por', 'Inicio en A',
                   'Inicio en B', 'Largo', 'Diferencias')
        anchos = (110, 110, 140, 140, 120, 110)
        self.tabla_sec = ttk.Treeview(marco_sec, columns=columnas,
                                      show='headings', height=24)
        for columna, titulo, ancho in zip(columnas, titulos, anchos):
            self.tabla_sec.heading(columna, text=titulo)
            self.tabla_sec.column(columna, width=ancho, anchor='center')
        self.tabla_sec.pack(side='left', fill='both', expand=True)
        barra = ttk.Scrollbar(marco_sec, orient='vertical',
                              command=self.tabla_sec.yview)
        barra.pack(side='right', fill='y')
        self.tabla_sec.configure(yscrollcommand=barra.set)

        # -- matriz de evidencias --
        # Va en un cuadro de texto monoespaciado y no en una tabla de
        # widgets porque las celdas son parrafos, no cifras: una tabla les
        # recorta el texto por el borde de la columna y la evidencia deja de
        # leerse.
        marco_evid = ttk.Frame(self.pestanas)
        self.pestanas.add(marco_evid, text='Evidencias')
        self.texto_evid = tk.Text(marco_evid, font=F_MONO_MIN, wrap='none',
                                  padx=12, pady=10, background=PAPEL,
                                  foreground=TINTA, relief='flat',
                                  borderwidth=0)
        self.texto_evid.pack(side='left', fill='both', expand=True)
        barra = ttk.Scrollbar(marco_evid, orient='vertical',
                              command=self.texto_evid.yview)
        barra.pack(side='right', fill='y')
        self.texto_evid.configure(yscrollcommand=barra.set,
                                  state='disabled')

        # -- registro --
        marco_log = ttk.Frame(self.pestanas)
        self.pestanas.add(marco_log, text='Registro')
        self.texto_log = tk.Text(marco_log, font=F_MONO_MIN, wrap='word',
                                 padx=12, pady=10, background=PAPEL,
                                 foreground=TINTA, relief='flat',
                                 borderwidth=0)
        self.texto_log.pack(side='left', fill='both', expand=True)
        barra = ttk.Scrollbar(marco_log, orient='vertical',
                              command=self.texto_log.yview)
        barra.pack(side='right', fill='y')
        self.texto_log.configure(yscrollcommand=barra.set)

        self.pestanas.bind('<<NotebookTabChanged>>',
                           self._al_cambiar_pestana)
        self._limpiar_graficas()

    def _al_cambiar_pestana(self, _evento=None):
        """Encaja la figura de la pestana que se acaba de mostrar.

        Un lienzo de Tk no conoce su tamano hasta que su pestana es visible:
        hasta entonces mide un pixel. Se le pregunta aqui, ya visible, y se
        le da a la figura exactamente ese tamano en pulgadas.

        Se hace a mano y no confiando en el redimensionado automatico de
        matplotlib porque ese solo se dispara con un evento de
        redimensionado del widget, que no llega cuando la pestana
        simplemente pasa a estar al frente. El resultado era una figura
        dibujada mas grande que su hueco, recortada por la derecha y por
        abajo justo en la captura que va al informe.
        """
        try:
            actual = self.pestanas.index(self.pestanas.select())
        except Exception:
            return

        pareja = {1: (getattr(self, 'figura', None),
                      getattr(self, 'lienzo', None)),
                  2: (getattr(self, 'figura_rec', None),
                      getattr(self, 'lienzo_rec', None))}.get(actual)
        if not pareja or pareja[0] is None:
            return
        figura, lienzo = pareja

        self.raiz.update_idletasks()
        widget = lienzo.get_tk_widget()
        ancho = widget.winfo_width()
        alto = widget.winfo_height()
        if ancho < 50 or alto < 50:
            # Todavia no tiene tamano real; se reintenta en cuanto Tk
            # termine de colocar los widgets.
            self.raiz.after(60, self._al_cambiar_pestana)
            return

        ppp = figura.get_dpi()
        figura.set_size_inches(ancho / ppp, alto / ppp, forward=False)
        self._ajustar_figuras()
        lienzo.draw()

    def _panel_pie(self, padre):
        tk.Frame(padre, height=1, background=LINEA).pack(fill='x',
                                                         side='bottom')
        marco = ttk.Frame(padre, padding=(16, 8))
        marco.pack(fill='x', side='bottom')
        self.barra = ttk.Progressbar(marco, variable=self.avance,
                                     maximum=100.0, length=260)
        self.barra.pack(side='left', padx=(0, 14))
        ttk.Label(marco, textvariable=self.estado,
                  style='Ayuda.TLabel').pack(side='left')

    # -- graficas ---------------------------------------------------------

    def _preparar_eje(self, eje, titulo):
        """Deja un eje limpio y con el aspecto comun a todas las graficas."""
        eje.clear()
        eje.set_facecolor(PAPEL)
        eje.set_title(titulo, fontsize=9.5, color=TINTA, pad=8)
        eje.tick_params(labelsize=7, colors=TINTA_SUAVE)
        for lado in ('top', 'right'):
            eje.spines[lado].set_visible(False)
        for lado in ('left', 'bottom'):
            eje.spines[lado].set_color(LINEA)
        eje.grid(axis='y', alpha=0.25, linewidth=0.6)

    def _marcar_vacio(self, eje, titulo, texto='Sin datos todavia'):
        self._preparar_eje(eje, titulo)
        eje.text(0.5, 0.5, texto, ha='center', va='center',
                 transform=eje.transAxes, color=TINTA_TENUE, fontsize=8.5)
        eje.set_xticks([])
        eje.set_yticks([])
        eje.grid(False)

    def _ajustar_figuras(self):
        for figura in (self.figura, self.figura_rec):
            figura.subplots_adjust(hspace=0.62, wspace=0.30, left=0.11,
                                   right=0.97, top=0.91, bottom=0.20)

    def _limpiar_graficas(self):
        for eje, titulo in zip(self.ejes, TITULOS_RENDIMIENTO):
            self._marcar_vacio(eje, titulo)
        for eje, titulo in zip(self.ejes_rec, TITULOS_RECURSOS):
            self._marcar_vacio(eje, titulo)
        self._ajustar_figuras()
        self.lienzo.draw()
        self.lienzo_rec.draw()

    def _barras(self, eje, titulo, valores, unidad, formato='%.3f',
                colores=None, etiquetas=None, rotacion=22):
        """Dibuja un grupo de barras con su valor escrito encima."""
        self._preparar_eje(eje, titulo)
        if etiquetas is None:
            etiquetas = [r['etiqueta'] for r in self.resultados]
        if colores is None:
            colores = [COLOR_PLATAFORMA.get(r['plataforma'], COLOR_CPU)
                       for r in self.resultados]
        indices = range(len(valores))
        dibujo = eje.bar(indices, valores, color=colores, width=0.62)
        eje.set_xticks(list(indices))
        eje.set_xticklabels(etiquetas, rotation=rotacion, ha='right',
                            fontsize=7)
        eje.set_ylabel(unidad, fontsize=7.5, color=TINTA_SUAVE)
        tope = max(valores) if valores and max(valores) else 1
        for barra, valor in zip(dibujo, valores):
            eje.text(barra.get_x() + barra.get_width() / 2,
                     barra.get_height() + tope * 0.02, formato % valor,
                     ha='center', va='bottom', fontsize=6.5,
                     color=TINTA_SUAVE)
        eje.set_ylim(0, tope * 1.18)

    def _dibujar_graficas(self):
        """Redibuja las cuatro graficas de rendimiento."""
        if not self.resultados:
            self._limpiar_graficas()
            return

        self._barras(self.ejes[0], TITULOS_RENDIMIENTO[0],
                     [r['tiempo'] for r in self.resultados], 'segundos')
        self._barras(self.ejes[1], TITULOS_RENDIMIENTO[1],
                     [r['mb_s'] for r in self.resultados], 'MB/s', '%.0f')

        # Preparacion frente a trabajo: justifica que el arranque de los
        # procesos y la compilacion de los kernels se midan aparte. En
        # entradas pequenas la preparacion es varias veces el trabajo.
        eje = self.ejes[2]
        self._preparar_eje(eje, TITULOS_RENDIMIENTO[2])
        indices = range(len(self.resultados))
        trabajo = [r['tiempo'] for r in self.resultados]
        preparacion = [r.get('preparacion', 0.0) for r in self.resultados]
        colores = [COLOR_PLATAFORMA.get(r['plataforma'], COLOR_CPU)
                   for r in self.resultados]
        eje.bar(indices, trabajo, color=colores, width=0.62,
                label='comparacion')
        eje.bar(indices, preparacion, bottom=trabajo, color='#d6d6da',
                width=0.62, label='preparacion')
        eje.set_xticks(list(indices))
        eje.set_xticklabels([r['etiqueta'] for r in self.resultados],
                            rotation=22, ha='right', fontsize=7)
        eje.set_ylabel('segundos', fontsize=7.5, color=TINTA_SUAVE)
        # Tope explicito: la barra apilada de preparacion, cuando vale cero,
        # impide que matplotlib deje margen arriba y la barra mas alta queda
        # pegada al borde. Mismo margen que el resto de graficas.
        tope = max(t + p for t, p in zip(trabajo, preparacion)) or 1
        eje.set_ylim(0, tope * 1.18)
        eje.legend(fontsize=6.5, frameon=False)

        # Donde caen las diferencias a lo largo del archivo. Es la unica
        # grafica que habla del problema en si y no del hardware: ensena si
        # las dos cadenas divergen de golpe en un punto o poco a poco por
        # todas partes.
        self._dibujar_mapa(self.ejes[3])

        self._ajustar_figuras()
        self.lienzo.draw()

    def _dibujar_mapa(self, eje):
        """Reparto de las diferencias a lo largo del archivo.

        Se dibuja desde el reparto por trozos que devuelve el motor de CPU, y
        NO desde la lista de posiciones del detalle. La lista esta cortada en
        las primeras mil, asi que dibujarla amontonaria todas las barras al
        principio del archivo y daria a entender que las cadenas divergen
        solo ahi, que es falso.
        """
        self._preparar_eje(eje, TITULOS_RENDIMIENTO[3])

        primero = self.resultados[0]['resultado']
        reparto = None
        for fila in self.resultados:
            posible = fila['resultado'].get('reparto')
            if posible:
                reparto = posible
                break

        if not primero['diferencias']:
            eje.text(0.5, 0.5, 'Las dos cadenas son identicas',
                     ha='center', va='center', transform=eje.transAxes,
                     color=COLOR_BIEN, fontsize=8.5)
            eje.set_xticks([])
            eje.set_yticks([])
            eje.grid(False)
            return

        if not reparto:
            self._marcar_vacio(eje, TITULOS_RENDIMIENTO[3],
                               'Solo lo calcula la CPU')
            return

        centros = [(inicio + _TROZO / 2) / 1e6 for inicio, _ in reparto]
        alturas = [cuenta for _, cuenta in reparto]
        ancho = (_TROZO / 1e6) * 0.92
        eje.bar(centros, alturas, width=ancho, color=COLOR_CPU, alpha=0.9)
        eje.set_xlabel('posicion en el archivo (millones de bases)',
                       fontsize=7.5, color=TINTA_SUAVE)
        eje.set_ylabel('diferencias por tramo', fontsize=7.5,
                       color=TINTA_SUAVE)
        eje.set_xlim(0, primero['comparados'] / 1e6)

    def _dibujar_recursos(self):
        """Redibuja las cuatro graficas de nucleos y recursos.

        Son las que responden a los criterios 4, 5 y 6 de la rubrica: no el
        rendimiento sino cuanto se uso cada nucleo, a que temperatura y con
        cuanta memoria.
        """
        if not self.resultados:
            return

        # 1) Uso de cada nucleo logico. Criterio 4, columna CPU.
        eje = self.ejes_rec[0]
        nucleos = None
        for fila in self.resultados:
            serie = (fila.get('recursos') or {}).get('cpu_por_nucleo')
            if serie:
                nucleos = serie
                break
        if nucleos:
            self._barras(eje, TITULOS_RECURSOS[0], nucleos, '%', '%.0f',
                         colores=[COLOR_CPU] * len(nucleos),
                         etiquetas=[str(i) for i in range(len(nucleos))],
                         rotacion=0)
            eje.set_xlabel('nucleo logico', fontsize=7.5,
                           color=TINTA_SUAVE)
        else:
            self._marcar_vacio(eje, TITULOS_RECURSOS[0],
                               'Sin muestras suficientes')

        # 2) Uso de cada plataforma mientras trabajaba. Criterio 4.
        self._serie_recurso(self.ejes_rec[1], TITULOS_RECURSOS[1],
                            {'CPU': 'cpu_medio', 'GPU': 'gpu_medio',
                             'NPU': 'npu_medio'}, '%', '%.0f')

        # 3) Temperatura. Criterio 5. La NPU no tiene sensor propio y
        #    comparte el del paquete con el procesador.
        self._serie_recurso(self.ejes_rec[2], TITULOS_RECURSOS[2],
                            {'CPU': 'cpu_temp_medio',
                             'GPU': 'gpu_temp_medio',
                             'NPU': 'cpu_temp_medio'}, 'grados C', '%.1f')

        # 4) Memoria. Criterio 6. La GPU se mide por su VRAM; la CPU y la
        #    NPU por la memoria residente del programa, porque la NPU no
        #    tiene memoria dedicada.
        self._serie_recurso(self.ejes_rec[3], TITULOS_RECURSOS[3],
                            {'CPU': 'proc_mb_max', 'GPU': 'vram_mb_max',
                             'NPU': 'proc_mb_max'}, 'MB', '%.0f')

        self._ajustar_figuras()
        self.lienzo_rec.draw()

    def _serie_recurso(self, eje, titulo, claves, unidad, formato):
        """Dibuja una magnitud del monitor, una barra por plataforma.

        Las magnitudes que el monitor no llego a medir no se dibujan. Una
        barra a cero se leeria como una medicion que salio cero, cuando lo
        que paso es que no hubo muestras suficientes; esa distincion es la
        regla que gobierna todo el panel de recursos.
        """
        valores = []
        etiquetas = []
        colores = []
        for fila in self.resultados:
            clave = claves.get(fila['plataforma'])
            valor = (fila.get('recursos') or {}).get(clave)
            if valor in (None, ''):
                continue
            valores.append(float(valor))
            etiquetas.append(fila['etiqueta'])
            colores.append(COLOR_PLATAFORMA.get(fila['plataforma'],
                                                COLOR_CPU))

        if not valores:
            self._marcar_vacio(eje, titulo, 'Sin muestras suficientes')
            return
        self._barras(eje, titulo, valores, unidad, formato,
                     colores=colores, etiquetas=etiquetas)
    def matriz_evidencias(self):
        """Construye la matriz con lo que se haya medido hasta ahora."""
        # Cada columna sale de las muestras tomadas mientras trabajaba esa
        # plataforma, que son las mismas que dibujan las graficas. Asi la
        # matriz y las graficas no pueden contradecirse.
        por_plataforma = {}
        for fila in self.resultados:
            por_plataforma[fila['plataforma']] = fila.get('recursos') or {}
        nucleos = (por_plataforma.get('CPU') or {}).get('cpu_por_nucleo')
        return evidencias.matriz(recursos_kernel=self.recursos_kernel,
                                 info_gpu=self.info_gpu,
                                 info_npu=self.info_npu,
                                 nucleos_cpu=nucleos,
                                 datos_npu=self.ultimo_npu,
                                 datos_gpu=self.ultimo_gpu,
                                 por_plataforma=por_plataforma)

    def _rellenar_evidencias(self):
        texto = evidencias.texto(self.matriz_evidencias())
        self.texto_evid.configure(state='normal')
        self.texto_evid.delete('1.0', 'end')
        self.texto_evid.insert('1.0', texto)
        self.texto_evid.configure(state='disabled')

    # -- eleccion de archivos ---------------------------------------------

    def _elegir(self, cual):
        if self.trabajando:
            return
        ruta = filedialog.askopenfilename(
            title='Elige la cadena %s' % cual,
            filetypes=[('Archivos FASTA', '*.fna *.fa *.fasta *.txt'),
                       ('Todos los archivos', '*.*')])
        if not ruta:
            return

        # Se valida en el acto, no al pulsar Comparar. Asi el usuario se
        # entera de que el archivo no sirve mientras todavia tiene el
        # explorador fresco en la cabeza.
        try:
            comparador.validar_fasta(ruta)
        except ErrorEntrada as error:
            messagebox.showerror('Archivo no valido', str(error))
            return

        if cual == 'A':
            self.ruta_a.set(ruta)
        else:
            self.ruta_b.set(ruta)
        self._describir_par()

    def _describir_par(self):
        """Avisa de lo que se ve a simple vista, antes de procesar nada."""
        a, b = self.ruta_a.get(), self.ruta_b.get()

        for cual, ruta in (('A', a), ('B', b)):
            etiqueta = self.etiquetas_archivo[cual]
            if ruta:
                etiqueta.configure(text=os.path.basename(ruta),
                                   foreground=TINTA)
            else:
                etiqueta.configure(text='(sin elegir)',
                                   foreground=TINTA_TENUE)

        if not a or not b:
            self.etiqueta_par.configure(text='', foreground=TINTA_SUAVE)
            return

        if os.path.abspath(a) == os.path.abspath(b):
            self.etiqueta_par.configure(
                text='Son el mismo archivo: el resultado sera cero '
                     'diferencias.',
                foreground=COLOR_AVISO)
            return

        try:
            ta = os.path.getsize(a) / (1024.0 * 1024.0)
            tb = os.path.getsize(b) / (1024.0 * 1024.0)
        except OSError:
            self.etiqueta_par.configure(text='', foreground=TINTA_SUAVE)
            return

        # Se avisa de la diferencia de tamano antes de procesar: es el caso
        # que mas confunde al leer el resultado despues.
        texto = 'A  %.1f MB      B  %.1f MB' % (ta, tb)
        color = TINTA_SUAVE
        if abs(ta - tb) > 0.05:
            texto += '   (miden distinto)'
            color = COLOR_AVISO
        self.etiqueta_par.configure(text=texto, foreground=color)

    # -- lanzamiento ------------------------------------------------------

    def _validar(self):
        """Comprueba todo antes de tocar un solo byte. Devuelve True si vale.

        Todos los mensajes estan escritos para que los lea una persona: dicen
        que pasa y que hacer, no como se llama la excepcion.
        """
        # Los campos numericos se normalizan ANTES de comprobar nada mas.
        # El filtro de teclado permite dejarlos vacios, porque hay que poder
        # borrarlos para teclear otro numero, y un campo vacio no es un error
        # del usuario sino un estado intermedio. Si esto fuera despues de las
        # comprobaciones de archivo, un campo vacio se quedaria vacio cada vez
        # que ademas faltara un archivo, que es justo cuando el usuario esta
        # empezando a rellenar la ventana.
        if not self.procesos.get().strip():
            self.procesos.set(str(self.logicos))
            self._log('El campo de procesos estaba vacio; se usan %d.'
                      % self.logicos)
        if not self.bloques_gpu.get().strip():
            self.bloques_gpu.set('1024')
        if not self.lote_gpu.get().strip():
            self.lote_gpu.set('64')
        if not self.capas_npu.get().strip():
            self.capas_npu.set('8')

        a, b = self.ruta_a.get(), self.ruta_b.get()

        if not a or not b:
            messagebox.showwarning(
                'Faltan archivos',
                'Hay que elegir las dos cadenas antes de comparar.\n\n'
                'Usa los botones "..." para seleccionarlas.')
            return False

        for ruta, nombre in ((a, 'A'), (b, 'B')):
            try:
                comparador.validar_fasta(ruta)
            except ErrorEntrada as error:
                # El archivo pudo borrarse o moverse entre que se eligio y se
                # pulso Comparar, asi que se vuelve a comprobar aqui.
                messagebox.showerror('Cadena %s no valida' % nombre,
                                     str(error))
                return False

        if self.modo.get() == 'gpu' and not self.hay_gpu:
            messagebox.showerror(
                'GPU no disponible',
                self.info_gpu.get('motivo')
                or 'No hay una tarjeta CUDA en este equipo.')
            return False

        if self.modo.get() == 'npu' and not self.hay_npu:
            messagebox.showerror(
                'NPU no disponible',
                self.info_npu.get('motivo')
                or 'No se pudo preparar la ruta de NPU.')
            return False

        return True

    def _recoger_configuracion(self):
        """Copia a un diccionario normal todo lo que el hilo va a necesitar.

        ESTO NO ES UNA COMODIDAD, ES OBLIGATORIO. Las variables de Tkinter
        (StringVar y companyia) no son objetos de Python corrientes: cada
        .get() entra en el interprete Tcl que hay debajo de la ventana, y Tcl
        solo se puede tocar desde el hilo que arranco el mainloop. Leer una
        de ellas desde el hilo trabajador aborta con

            RuntimeError: main thread is not in the main loop

        y ademas lo hace de forma intermitente, porque depende de en que
        momento caiga la llamada.

        Por eso el hilo trabajador NO recibe la ventana ni ninguna de sus
        variables: recibe este diccionario de cadenas y numeros, leido aqui,
        en el hilo principal, antes de arrancar nada. La regla completa es
        que el hilo trabajador no lee ni escribe un solo widget; lo que tenga
        que decir lo deja en la cola.
        """
        return {
            'ruta_a': self.ruta_a.get(),
            'ruta_b': self.ruta_b.get(),
            'modo': self.modo.get(),
            'procesos': self.procesos.get(),
            'bloques_gpu': self.bloques_gpu.get(),
            'lote_gpu': self.lote_gpu.get(),
            'capas_npu': self.capas_npu.get(),
            'por_secuencia': bool(self.por_secuencia.get()),
            'medicion_justa': bool(self.medicion_justa.get()),
            'mezclar': bool(self.mezclar.get()),
            'mezclar_gpu': bool(self.mezclar_gpu.get()) and self.hay_gpu,
            'mezclar_hibrido': (bool(self.mezclar_hibrido.get())
                                and self.hay_gpu),
        }

    def comparar(self):
        if self.trabajando:
            return
        if not self._validar():
            return

        configuracion = self._recoger_configuracion()
        self._empezar()
        self.hilo = threading.Thread(target=self._tarea,
                                     args=(configuracion,), daemon=True)
        self.hilo.start()

    def _empezar(self):
        self.trabajando = True
        self.resultados = []
        self.ultimo_npu = {}
        self.ultimo_gpu = {}
        self.avance.set(0.0)
        self.boton_comparar.state(['disabled'])
        self.boton_exportar.state(['disabled'])
        for control in (self.spin_procesos, self.spin_bloques, self.spin_lote,
                        self.spin_capas):
            control.state(['disabled'])
        self.tabla.delete(*self.tabla.get_children())
        self.tabla_sec.delete(*self.tabla_sec.get_children())
        self.por_crom = {}
        self.cifra.configure(text='—', foreground=TINTA)
        self.cifra_pie.configure(text='comparando...')
        self.detalle_resultado.configure(text='')
        self.acuerdo.configure(text='')

    def _terminar(self):
        self.trabajando = False
        self.boton_comparar.state(['!disabled'])
        if self.resultados:
            self.boton_exportar.state(['!disabled'])
        self.spin_procesos.state(['!disabled'])
        if self.hay_gpu:
            self.spin_bloques.state(['!disabled'])
            self.spin_lote.state(['!disabled'])
        if self.hay_npu:
            self.spin_capas.state(['!disabled'])

    # -- mensajes al hilo principal ---------------------------------------

    def _enviar(self, tipo, dato):
        self.cola.put((tipo, dato))

    def _log(self, texto):
        self._enviar('log', texto)

    def _estado(self, texto):
        self._enviar('estado', texto)

    def _progreso(self, hechos, total):
        self._enviar('progreso', 100.0 * hechos / total if total else 0.0)

    # -- trabajo ----------------------------------------------------------

    def _tarea(self, configuracion):
        """Corre en el hilo trabajador. No toca ni un widget.

        Recibe la configuracion ya leida como diccionario; ver
        _recoger_configuracion() para el motivo. Todo lo que este metodo
        quiera mostrar sale por la cola, que vacia el hilo principal.
        """
        modo = configuracion['modo']
        try:
            self._estado('Preparando las cadenas...')
            self._log('=' * 62)
            self._log('Comparacion iniciada: %s'
                      % time.strftime('%Y-%m-%d %H:%M:%S'))

            ca = comparador.preparar(configuracion['ruta_a'],
                                     progreso=self._progreso)
            self._log('Cadena A: %d bases  (%s)'
                      % (ca.largo, 'cache' if ca.desde_cache
                         else '%.2f s de preparacion' % ca.preparacion_s))
            cb = comparador.preparar(configuracion['ruta_b'],
                                     progreso=self._progreso)
            self._log('Cadena B: %d bases  (%s)'
                      % (cb.largo, 'cache' if cb.desde_cache
                         else '%.2f s de preparacion' % cb.preparacion_s))

            for aviso in comparador.validar_par(ca, cb):
                self._log('AVISO: %s' % aviso)

            self.parejas = []
            if configuracion.get('por_secuencia'):
                self._estado('Indexando las secuencias...')
                ra = secuencias.indexar(configuracion['ruta_a'],
                                        progreso=self._progreso)
                rb = secuencias.indexar(configuracion['ruta_b'],
                                        progreso=self._progreso)
                parejas, solo_a, solo_b = secuencias.emparejar(ra, rb)
                self.parejas = parejas
                self.sin_pareja = (solo_a, solo_b)
                for linea in secuencias.resumen_emparejamiento(
                        parejas, solo_a, solo_b).split(chr(10)):
                    self._log(linea)
                for r in solo_a:
                    self._log('  sin pareja en B: %s (%d bases)'
                              % (r.identificador, r.largo))
                for r in solo_b:
                    self._log('  sin pareja en A: %s (%d bases)'
                              % (r.identificador, r.largo))
                if not parejas:
                    self._log('No se emparejo ninguna secuencia; se compara '
                              'de corrido.')

            self.cadena_a, self.cadena_b = ca, cb
            mb = min(ca.largo, cb.largo) / (1024.0 * 1024.0)

            plataformas = (['cpu', 'gpu', 'npu'] if modo == 'todas'
                           else [modo])
            for plataforma in plataformas:
                if plataforma == 'gpu' and not self.hay_gpu:
                    self._log('GPU: se omite, no disponible en este equipo.')
                    continue
                if plataforma == 'npu' and not self.hay_npu:
                    self._log('NPU: se omite, no disponible en este equipo.')
                    continue
                self._medir(plataforma, ca, cb, mb, configuracion)

            # El archivo mezclado lo genera siempre la CPU, sea cual sea la
            # plataforma elegida para comparar, y con los mismos procesos.
            if configuracion.get('mezclar'):
                self._estado('Generando el archivo mezclado con la CPU...')
                resumen = mezclador.mezclar(
                    ca, cb, procesos=_entero(configuracion['procesos'],
                                             self.logicos),
                    progreso=self._progreso)
                for linea in mezclador.formatear(resumen).split(chr(10)):
                    self._log(linea)

            if configuracion.get('mezclar_gpu'):
                import mezclador_gpu
                self._estado('Generando el archivo mezclado con la GPU...')
                resumen = mezclador_gpu.mezclar_gpu(
                    ca, cb,
                    lote_mb=_entero(configuracion['lote_gpu'], 64),
                    bloques=_entero(configuracion['bloques_gpu'], 1024),
                    progreso=self._progreso)
                for linea in mezclador_gpu.formatear(resumen).split(chr(10)):
                    self._log(linea)

            if configuracion.get('mezclar_hibrido'):
                import mezclador_hibrido
                self._estado('Generando el archivo mezclado con CPU + GPU...')
                resumen = mezclador_hibrido.mezclar_hibrido(
                    ca, cb,
                    procesos=_entero(configuracion['procesos'],
                                     self.logicos),
                    bloques=_entero(configuracion['bloques_gpu'], 1024),
                    progreso=self._progreso)
                for linea in mezclador_hibrido.formatear(resumen).split(
                        chr(10)):
                    self._log(linea)

            self._enviar('fin', None)
        except ErrorEntrada as error:
            self._enviar('error', str(error))
        except Exception as error:                    # pragma: no cover
            self._enviar('error',
                         'Ocurrio un fallo inesperado:\n\n%s: %s'
                         % (type(error).__name__, error))

    def _motor_de(self, plataforma, ca, cb, configuracion):
        """Devuelve (etiqueta, funcion) para la plataforma pedida.

        La funcion toma dos tramos o cadenas y devuelve (resultado, segundos),
        que es la misma firma para las tres. Asi el resto del metodo no
        necesita saber cual esta corriendo.
        """
        if plataforma == 'cpu':
            n = _entero(configuracion['procesos'], self.logicos)
            ajustado, aviso = motor_cpu.procesos_validos(
                n, min(ca.largo, cb.largo))
            if aviso:
                self._log('AJUSTE: %s' % aviso)
            return ('CPU x%d' % ajustado,
                    lambda a, b: motor_cpu.comparar_paralelo(
                        a, b, procesos=ajustado))

        if plataforma == 'gpu':
            import motor_gpu
            lote = _entero(configuracion['lote_gpu'], 64)
            bloques, aviso = motor_gpu.bloques_validos(
                _entero(configuracion.get('bloques_gpu'), motor_gpu.BLOQUES))
            if aviso:
                self._log('AJUSTE: %s' % aviso)
            if self.recursos_kernel is None:
                self.recursos_kernel = motor_gpu.recursos_kernel()
            def gpu(a, b):
                resultado, tiempo = motor_gpu.comparar_gpu(
                    a, b, lote_mb=lote, bloques=bloques)
                self.ultimo_gpu = resultado['gpu']
                return resultado, tiempo
            return 'GPU %d bloques' % bloques, gpu

        import motor_npu
        capas = _entero(configuracion['capas_npu'], 8)
        def npu(a, b):
            resultado, tiempo = motor_npu.comparar_npu(a, b, capas=capas)
            self.ultimo_npu = resultado['npu']
            return resultado, tiempo
        return 'NPU %d capas' % capas, npu

    def _cronometrar(self, funcion, ca, cb, justa):
        """Corre la comparacion y devuelve (resultado, segundos).

        Con medicion justa se descarta una corrida de calentamiento y se
        publica la MEDIANA de las siguientes. La mediana y no la media porque
        basta un pico del sistema para arruinar un promedio de tres.

        Sin ella se corre una sola vez, que es lo rapido pero le regala una
        ventaja a las plataformas que corren despues: encuentran los datos ya
        en la cache del sistema.
        """
        if not justa:
            return funcion(ca, cb)

        self._estado('Calentando...')
        funcion(ca, cb)                       # descartada a proposito

        tiempos = []
        resultado = None
        for vuelta in range(REPETICIONES):
            self._estado('Midiendo %d de %d...' % (vuelta + 1, REPETICIONES))
            resultado, tiempo = funcion(ca, cb)
            tiempos.append(tiempo)
        tiempos.sort()
        return resultado, tiempos[len(tiempos) // 2]

    def _medir(self, plataforma, ca, cb, mb, configuracion):
        """Ejecuta una plataforma y publica su resultado.

        Los parametros salen del diccionario y nunca de las variables de la
        ventana: este metodo corre en el hilo trabajador.
        """
        marca = self.monitor.marcar()
        justa = configuracion.get('medicion_justa', False)

        if configuracion.get('por_secuencia') and self.parejas:
            return self._medir_por_secuencia(plataforma, ca, cb, mb,
                                             configuracion, marca, justa)

        if plataforma == 'cpu':
            n = _entero(configuracion['procesos'], self.logicos)
            ajustado, aviso = motor_cpu.procesos_validos(n, min(ca.largo,
                                                                cb.largo))
            if aviso:
                self._log('AJUSTE: %s' % aviso)
            self._estado('Comparando en CPU con %d procesos...' % ajustado)
            resultado, tiempo = self._cronometrar(
                lambda a, b: motor_cpu.comparar_paralelo(
                    a, b, procesos=ajustado, progreso=self._progreso),
                ca, cb, justa)
            etiqueta = 'CPU x%d' % resultado['cpu']['procesos']
            preparacion = resultado['cpu'].get('preparacion_s', 0.0)

        elif plataforma == 'gpu':
            import motor_gpu
            lote = _entero(configuracion['lote_gpu'], 64)
            bloques = _entero(configuracion.get('bloques_gpu'),
                              motor_gpu.BLOQUES)
            self._estado('Comparando en GPU con %d bloques...' % bloques)
            if self.recursos_kernel is None:
                self.recursos_kernel = motor_gpu.recursos_kernel()
            resultado, tiempo = self._cronometrar(
                lambda a, b: motor_gpu.comparar_gpu(
                    a, b, lote_mb=lote, bloques=bloques,
                    progreso=self._progreso),
                ca, cb, justa)
            if resultado['gpu'].get('aviso'):
                self._log('AJUSTE: %s' % resultado['gpu']['aviso'])
            if resultado['gpu'].get('nota'):
                self._log('NOTA: %s' % resultado['gpu']['nota'])
            etiqueta = 'GPU %d bloques' % resultado['gpu']['bloques']
            self.ultimo_gpu = resultado['gpu']
            preparacion = resultado['gpu'].get('preparacion_s', 0.0)

        else:
            import motor_npu
            capas = _entero(configuracion['capas_npu'], 8)
            self._estado('Comparando en NPU...')
            resultado, tiempo = self._cronometrar(
                lambda a, b: motor_npu.comparar_npu(
                    a, b, capas=capas, progreso=self._progreso),
                ca, cb, justa)
            for aviso in resultado['npu'].get('avisos', []):
                self._log('AVISO: %s' % aviso)
            etiqueta = 'NPU %d capas' % resultado['npu']['capas']
            preparacion = resultado['npu'].get('preparacion_s', 0.0)
            self.ultimo_npu = resultado['npu']

        recursos = self.monitor.resumen(marca)
        fila = {
            'etiqueta': etiqueta,
            'plataforma': plataforma.upper(),
            'tiempo': tiempo,
            'mb_s': mb / tiempo if tiempo else 0.0,
            'diferencias': resultado['diferencias'],
            'similitud': resultado['similitud_pct'],
            'preparacion': preparacion,
            'resultado': resultado,
            'recursos': recursos,
        }
        if not resultado.get('valido', True):
            self._log('RESULTADO INVALIDO en %s: %s'
                      % (etiqueta, resultado.get('motivo_invalido')))
        self._log('%-16s %8.3f s  %8.1f MB/s  %d diferencias'
                  % (etiqueta, tiempo, fila['mb_s'],
                     resultado['diferencias']))
        self._enviar('resultado', fila)

    def _medir_por_secuencia(self, plataforma, ca, cb, mb, configuracion,
                             marca, justa):
        """Compara secuencia con secuencia en vez de posicion con posicion.

        Es lo que hace que la comparacion signifique algo cuando los dos
        archivos guardan las mismas secuencias en distinto orden. Ver
        secuencias.py para el diagnostico completo.
        """
        etiqueta, motor = self._motor_de(plataforma, ca, cb, configuracion)
        total = len(self.parejas)

        def avance(hechas, cuantas):
            self._progreso(hechas, cuantas)
            self._estado('%s: secuencia %d de %d' % (etiqueta, hechas,
                                                     cuantas))

        def corrida(_a, _b):
            resumen = secuencias.comparar_parejas(ca, cb, self.parejas,
                                                  motor, progreso=avance)
            return resumen, resumen['tiempo_s']

        resumen, tiempo = self._cronometrar(corrida, ca, cb, justa)

        recursos = self.monitor.resumen(marca)
        comparadas = resumen['bases']
        fila = {
            'etiqueta': etiqueta,
            'plataforma': plataforma.upper(),
            'tiempo': tiempo,
            'mb_s': (comparadas / (1024.0 * 1024.0) / tiempo) if tiempo
            else 0.0,
            'diferencias': resumen['diferencias'],
            'similitud': resumen['similitud_pct'],
            'preparacion': 0.0,
            'recursos': recursos,
            'por_secuencia': True,
            'resumen_sec': resumen,
            'resultado': {
                'comparados': comparadas,
                'diferencias': resumen['diferencias'],
                'iguales': resumen['iguales'],
                'similitud_pct': resumen['similitud_pct'],
                'desfase': 0,
                'valido': resumen['invalidas'] == 0,
                'motivo_invalido': None,
                'detalle': [d for f in resumen['filas']
                            for d in f.get('detalle', [])][:MAX_TABLA],
                'reparto': None,
            },
        }
        self._log('%-16s %8.3f s  %d secuencias  %d diferencias'
                  % (etiqueta, tiempo, resumen['parejas'],
                     resumen['diferencias']))
        self._enviar('resultado', fila)

    # -- cola -------------------------------------------------------------

    def _revisar_cola(self):
        try:
            while True:
                tipo, dato = self.cola.get_nowait()

                if tipo == 'log':
                    self.texto_log.insert('end', dato + '\n')
                    self.texto_log.see('end')
                elif tipo == 'estado':
                    self.estado.set(dato)
                elif tipo == 'progreso':
                    self.avance.set(dato)
                elif tipo == 'resultado':
                    self.resultados.append(dato)
                    self._mostrar(dato)
                    self._dibujar_graficas()
                    self._dibujar_recursos()
                elif tipo == 'error':
                    self.estado.set('Se detuvo por un error.')
                    self.avance.set(0.0)
                    self._terminar()
                    messagebox.showerror('No se pudo comparar', dato)
                elif tipo == 'fin':
                    self.avance.set(100.0)
                    self.estado.set('Comparacion terminada.')
                    self._terminar()
                    self._rellenar_evidencias()
        except queue.Empty:
            pass

        if not self.cerrando:
            self.raiz.after(INTERVALO_COLA, self._revisar_cola)

    def _mostrar(self, fila):
        """Vuelca un resultado al panel de resumen y a la tabla."""
        resultado = fila['resultado']

        # La cifra manda: es el dato por el que existe el programa. Los
        # separadores de millar se ponen a mano porque el formato con coma de
        # Python no respeta la convencion en castellano.
        self.cifra.configure(
            text=_miles(resultado['diferencias']),
            foreground=COLOR_BIEN if not resultado['diferencias'] else TINTA)
        self.cifra_pie.configure(
            text='diferencias  ·  %.4f %% iguales'
                 % resultado['similitud_pct'])

        # Una linea por plataforma medida hasta ahora, no solo la ultima.
        lineas = ['%-13s %9s %12s' % ('plataforma', 'tiempo', 'difs')]
        for otra in self.resultados:
            marca = '' if otra['resultado'].get('valido', True) else ' !'
            lineas.append('%-13s %8.3fs %12s%s'
                          % (otra['etiqueta'][:13], otra['tiempo'],
                             _miles(otra['diferencias']), marca))
        lineas.append('')
        lineas.append('comparadas   %s' % _miles(resultado['comparados']))
        if resultado.get('desfase'):
            lineas.append('desfase      %+d bases' % resultado['desfase'])
        self.detalle_resultado.configure(text='\n'.join(lineas))

        # Indicador de acuerdo entre plataformas: es la prueba de que el
        # resultado no depende del hardware que lo calculo.
        cuentas = {r['diferencias'] for r in self.resultados}
        invalidas = [r for r in self.resultados
                     if not r['resultado'].get('valido', True)]
        if invalidas:
            self.acuerdo.configure(
                text='%d resultado(s) invalido(s), ver el registro'
                     % len(invalidas), foreground=COLOR_AVISO)
        elif len(self.resultados) < 2:
            self.acuerdo.configure(text='')
        elif len(cuentas) == 1:
            self.acuerdo.configure(
                text='Las %d plataformas coinciden' % len(self.resultados),
                foreground=COLOR_BIEN)
        else:
            self.acuerdo.configure(text='LAS PLATAFORMAS NO COINCIDEN',
                                   foreground=COLOR_AVISO)

        # Tabla por secuencia, con la primera plataforma que la traiga.
        if fila.get('resumen_sec') and not self.tabla_sec.get_children():
            for f in fila['resumen_sec']['filas']:
                self.tabla_sec.insert('', 'end', values=(
                    f['nombre'], f['criterio'], _miles(f['inicio_a']),
                    _miles(f['inicio_b']), _miles(f['largo']),
                    _miles(f['diferencias'])))
            for r in self.sin_pareja[0]:
                self.tabla_sec.insert('', 'end', values=(
                    r.identificador, 'sin pareja en B', '—', '—',
                    _miles(r.largo), 'no comparada'))
            for r in self.sin_pareja[1]:
                self.tabla_sec.insert('', 'end', values=(
                    r.identificador, 'sin pareja en A', '—', '—',
                    _miles(r.largo), 'no comparada'))

        # La tabla se llena con el primer resultado que llegue; los demas
        # coinciden, y verificarlo es justo lo que hace verificar.py.
        if not self.tabla.get_children():
            mostradas = resultado['detalle'][:MAX_TABLA]
            for i, d in enumerate(mostradas, 1):
                self.tabla.insert('', 'end', values=(
                    i, d['posicion'], d['fila'], d['columna'],
                    d['a'], d['b']))
            # Lo que falta se cuenta contra lo que de verdad se mostro, no
            # contra MAX_TABLA: el detalle puede traer menos filas.
            if resultado['diferencias'] > len(mostradas):
                self.tabla.insert('', 'end', values=(
                    '...', 'y %d mas' % (resultado['diferencias']
                                         - len(mostradas)),
                    '', '', '', ''))

    # -- recursos en vivo -------------------------------------------------

    def _refrescar_recursos(self):
        if self.cerrando:
            return
        muestra = self.monitor.ultima() if self.monitor else {}

        def poner(clave, texto):
            self.recursos[clave].configure(text=texto)

        formatos = {
            'cpu': ('%.0f %%', 1), 'cpu_temp': ('%.1f C', 1),
            'ram': ('%.0f %%', 1), 'proc_mb': ('%d MB', 1),
            'gpu': ('%.0f %%', 1), 'gpu_temp': ('%.0f C', 1),
            'vram_mb': ('%d MB', 1), 'npu': ('%.0f %%', 1),
        }
        for clave, (formato, _) in formatos.items():
            valor = muestra.get(clave)
            # Una magnitud que no se pudo leer muestra un guion, nunca un
            # cero: un cero se leeria como una medicion real que salio cero.
            poner(clave, formato % valor if valor is not None else '—')

        self.raiz.after(INTERVALO_RECURSOS, self._refrescar_recursos)

    # -- exportar ---------------------------------------------------------

    def exportar(self):
        if not self.resultados:
            messagebox.showinfo('Nada que exportar',
                                'Primero hay que comparar dos cadenas.')
            return

        carpeta = 'resultados'
        try:
            if not os.path.isdir(carpeta):
                os.makedirs(carpeta)

            ruta_csv = os.path.join(carpeta, 'comparacion.csv')
            with open(ruta_csv, 'w', encoding='utf-8', newline='') as f:
                f.write('plataforma,etiqueta,tiempo_s,mb_por_s,diferencias,'
                        'similitud_pct,preparacion_s\n')
                for r in self.resultados:
                    f.write('%s,%s,%.4f,%.2f,%d,%.6f,%.4f\n'
                            % (r['plataforma'], r['etiqueta'], r['tiempo'],
                               r['mb_s'], r['diferencias'], r['similitud'],
                               r['preparacion']))

            ruta_dif = os.path.join(carpeta, 'diferencias.csv')
            primero = self.resultados[0]['resultado']
            with open(ruta_dif, 'w', encoding='utf-8', newline='') as f:
                f.write('posicion,fila,columna,cadena_a,cadena_b\n')
                for d in primero['detalle']:
                    f.write('%d,%d,%d,%s,%s\n'
                            % (d['posicion'], d['fila'], d['columna'],
                               d['a'], d['b']))

            ruta_evid = evidencias.guardar(self.matriz_evidencias(),
                                           os.path.join(carpeta,
                                                        'evidencias.txt'))
            self.figura.savefig(os.path.join(carpeta, 'graficas.png'),
                                dpi=120)

            messagebox.showinfo(
                'Exportado',
                'Se guardaron en la carpeta "%s":\n\n'
                '  comparacion.csv\n  diferencias.csv\n  evidencias.txt\n'
                '  graficas.png' % carpeta)
            self._log('Exportado a %s' % os.path.abspath(carpeta))
        except OSError as error:
            messagebox.showerror(
                'No se pudo exportar',
                'No se pudieron escribir los archivos en "%s".\n\n%s\n\n'
                'Comprueba que la carpeta no este abierta en otro programa '
                'y que haya espacio en el disco.' % (carpeta, error))

    # -- cierre -----------------------------------------------------------

    def cerrar(self):
        """Cierre ordenado, con confirmacion si hay trabajo en curso.

        Cerrar en mitad de una comparacion deja procesos hijos vivos, asi que
        se pregunta antes en vez de matarlos por sorpresa.
        """
        if self.trabajando:
            if not messagebox.askyesno(
                    'Hay una comparacion en curso',
                    'Se esta comparando ahora mismo.\n\n'
                    'Si cierras, el trabajo se interrumpe y se pierden los '
                    'resultados que no se hayan mostrado todavia.\n\n'
                    '¿Cerrar de todos modos?'):
                return

        self.cerrando = True
        try:
            if self.monitor is not None:
                self.monitor.detener()
        except Exception:
            pass
        for cadena in (self.cadena_a, self.cadena_b):
            if cadena is not None:
                cadena.cerrar()
        self.raiz.destroy()


def ejecutar():
    raiz = tk.Tk()
    try:
        # Escalado por DPI: sin esto la ventana sale borrosa y con los
        # tamanos mal en pantallas al 125 por ciento, que es la de este
        # equipo.
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    Aplicacion(raiz)
    raiz.mainloop()
    return 0
