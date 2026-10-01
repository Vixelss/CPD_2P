# -*- coding: utf-8 -*-
"""
app.py

Interfaz grafica del proyecto P1.1: conteo de nucleotidos A, C, G y T
sobre archivos FASTA, comparando ejecucion secuencial contra ejecucion
paralela con multiprocessing.

La ventana permite:
  - Elegir el archivo FASTA.
  - Elegir el numero de procesos y lanzar una sola ejecucion.
  - Lanzar el analisis completo (barrido con 1, 2, 4, 8 y 12 procesos).
  - Ver el conteo de A, C, G, T y el total de bases.
  - Ver los nucleos fisicos y logicos detectados.
  - Seguir el avance con una barra de progreso y la velocidad en MB/s.
  - Ver cuatro graficas: tiempo, tabla comparativa, speedup y eficiencia.
  - Exportar los resultados a CSV.

Todo el procesamiento corre en un hilo aparte para que la ventana siga
respondiendo mientras se leen archivos de 3 GB.

Ejecutar con:
    python app.py
"""

import multiprocessing
import os
import queue
import threading
import time

import benchmark
import motor_adn

# ---------------------------------------------------------------------------
# Importaciones pesadas
#
# En Windows multiprocessing crea cada proceso hijo volviendo a importar este
# archivo, pero bajo el nombre '__mp_main__' en vez de '__main__'. Cargar
# tkinter y matplotlib cuesta cerca de un segundo por hijo, tiempo que se
# sumaria a cada medicion y hundiria el speedup justo en las configuraciones
# con mas procesos. Como los hijos solo cuentan bases y nunca dibujan nada,
# se saltan estas importaciones.
# ---------------------------------------------------------------------------
if __name__ != '__mp_main__':
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Paleta de la interfaz.
COLOR_FONDO = '#f4f6f8'
COLOR_PANEL = '#ffffff'
COLOR_TITULO = '#1f3a5f'
COLOR_BASE = {'A': '#2e7d32', 'C': '#1565c0', 'G': '#ef6c00', 'T': '#c62828'}

# Configuraciones del barrido pedidas por el proyecto.
BARRIDO = benchmark.PROCESOS_POR_DEFECTO


class Aplicacion(object):
    """Ventana principal de la aplicacion."""

    def __init__(self, raiz):
        self.raiz = raiz
        self.raiz.title('P1.1 - Conteo paralelo de nucleotidos (FASTA)')

        # La ventana se adapta a la pantalla: en monitores con escalado de
        # Windows el alto util es bastante menor de lo que parece, y las
        # graficas necesitan todo el espacio que quede libre.
        ancho = min(1340, self.raiz.winfo_screenwidth() - 60)
        alto = min(980, self.raiz.winfo_screenheight() - 70)
        self.raiz.geometry('%dx%d' % (ancho, alto))
        self.raiz.minsize(1040, 640)
        self.raiz.configure(bg=COLOR_FONDO)

        # Estado interno.
        self.ruta_archivo = None
        self.tamano_archivo = 0
        self.resumen = None          # ultimo barrido completo
        self.trabajando = False
        self.cola = queue.Queue()
        self.inicio_ejecucion = 0.0
        self.fisicos, self.logicos = motor_adn.detectar_nucleos()

        self._configurar_estilos()
        self._construir_interfaz()
        self._dibujar_graficas(None)

        # Bucle de sondeo: el hilo de trabajo nunca toca los widgets, solo
        # deja mensajes en la cola y esta funcion los aplica.
        self.raiz.after(100, self._procesar_cola)

    # -----------------------------------------------------------------
    # Construccion de la interfaz
    # -----------------------------------------------------------------

    def _configurar_estilos(self):
        estilo = ttk.Style()
        try:
            estilo.theme_use('clam')
        except tk.TclError:
            pass

        estilo.configure('TFrame', background=COLOR_FONDO)
        estilo.configure('Panel.TLabelframe', background=COLOR_PANEL)
        estilo.configure('Panel.TLabelframe.Label', background=COLOR_FONDO,
                         foreground=COLOR_TITULO,
                         font=('Segoe UI', 10, 'bold'))
        estilo.configure('TLabel', background=COLOR_FONDO)
        estilo.configure('Panel.TLabel', background=COLOR_PANEL)
        estilo.configure('Dato.TLabel', background=COLOR_PANEL,
                         font=('Consolas', 11))
        estilo.configure('Accion.TButton', font=('Segoe UI', 9, 'bold'))

    def _construir_interfaz(self):
        contenedor = ttk.Frame(self.raiz, padding=8)
        contenedor.pack(fill='both', expand=True)

        self._construir_panel_configuracion(contenedor)
        self._construir_panel_progreso(contenedor)
        self._construir_panel_conteo(contenedor)
        self._construir_panel_graficas(contenedor)

    def _construir_panel_configuracion(self, padre):
        """Archivo, hardware detectado y botones, todo en dos filas."""
        marco = ttk.LabelFrame(padre, text=' Configuracion ',
                               style='Panel.TLabelframe', padding=6)
        marco.pack(fill='x', pady=(0, 4))
        marco.columnconfigure(1, weight=1)

        # --- Fila 0: archivo ---
        ttk.Label(marco, text='Archivo FASTA:',
                  style='Panel.TLabel').grid(row=0, column=0, sticky='w',
                                             padx=(0, 6))

        self.var_ruta = tk.StringVar(value='Ningun archivo seleccionado')
        ttk.Entry(marco, textvariable=self.var_ruta,
                  state='readonly').grid(row=0, column=1, sticky='ew',
                                         padx=(0, 6))

        self.var_tamano = tk.StringVar(value='')
        ttk.Label(marco, textvariable=self.var_tamano, style='Panel.TLabel',
                  width=11, anchor='e',
                  font=('Consolas', 10)).grid(row=0, column=2, padx=(0, 6))

        ttk.Button(marco, text='Seleccionar archivo...',
                   command=self.seleccionar_archivo,
                   style='Accion.TButton').grid(row=0, column=3, sticky='e')

        # --- Fila 1: hardware y acciones ---
        acciones = ttk.Frame(marco)
        acciones.grid(row=1, column=0, columnspan=4, sticky='ew', pady=(8, 0))

        ttk.Label(acciones,
                  text='Nucleos fisicos: %d   |   logicos: %d   |   psutil: %s'
                       % (self.fisicos, self.logicos,
                          'si' if motor_adn._HAY_PSUTIL else 'no (estimado)'),
                  style='Dato.TLabel').pack(side='left', padx=(0, 16))

        ttk.Label(acciones, text='Procesos:',
                  style='Panel.TLabel').pack(side='left', padx=(0, 4))

        opciones = sorted(set(list(BARRIDO) + list(range(1, self.logicos + 1))))
        self.var_procesos = tk.StringVar(value=str(self.logicos))
        self.combo_procesos = ttk.Combobox(
            acciones, textvariable=self.var_procesos, state='readonly',
            width=5, values=[str(n) for n in opciones])
        self.combo_procesos.pack(side='left', padx=(0, 10))

        self.btn_procesar = ttk.Button(
            acciones, text='Procesar', style='Accion.TButton',
            command=self.procesar_una_configuracion)
        self.btn_procesar.pack(side='left', padx=3)

        self.btn_analisis = ttk.Button(
            acciones, text='Analisis completo (1-2-4-8-12)',
            style='Accion.TButton', command=self.analisis_completo)
        self.btn_analisis.pack(side='left', padx=3)

        self.btn_exportar = ttk.Button(
            acciones, text='Exportar CSV', style='Accion.TButton',
            command=self.exportar_csv, state='disabled')
        self.btn_exportar.pack(side='left', padx=3)

        self.var_secuencial = tk.BooleanVar(value=True)
        ttk.Checkbutton(acciones, text='Incluir secuencial',
                        variable=self.var_secuencial).pack(side='left', padx=(12, 0))

    def _construir_panel_progreso(self, padre):
        marco = ttk.Frame(padre)
        marco.pack(fill='x', pady=2)

        self.var_progreso = tk.DoubleVar(value=0.0)
        self.barra = ttk.Progressbar(marco, variable=self.var_progreso,
                                     maximum=100.0)
        self.barra.pack(side='left', fill='x', expand=True, padx=(0, 10))

        self.var_velocidad = tk.StringVar(
            value='  0.0%   |     0.0 MB/s   |    0.0 s')
        ttk.Label(marco, textvariable=self.var_velocidad,
                  font=('Consolas', 10, 'bold')).pack(side='left')

        self.var_estado = tk.StringVar(value='Listo. Selecciona un archivo FASTA.')
        ttk.Label(padre, textvariable=self.var_estado,
                  font=('Segoe UI', 9)).pack(anchor='w', pady=(1, 3))

    def _construir_panel_conteo(self, padre):
        marco = ttk.LabelFrame(padre, text=' Conteo de nucleotidos ',
                               style='Panel.TLabelframe', padding=5)
        marco.pack(fill='x', pady=2)

        self.var_conteo = {}
        for columna, base in enumerate(('A', 'C', 'G', 'T')):
            celda = tk.Frame(marco, bg=COLOR_PANEL, highlightthickness=1,
                             highlightbackground='#d5dbe1')
            celda.grid(row=0, column=columna, padx=4, pady=1, sticky='nsew')
            marco.columnconfigure(columna, weight=1)

            fila = tk.Frame(celda, bg=COLOR_PANEL)
            fila.pack(pady=2)
            tk.Label(fila, text=base, bg=COLOR_PANEL, fg=COLOR_BASE[base],
                     font=('Segoe UI', 14, 'bold')).pack(side='left', padx=(0, 8))

            var = tk.StringVar(value='-')
            tk.Label(fila, textvariable=var, bg=COLOR_PANEL,
                     font=('Consolas', 12)).pack(side='left')
            self.var_conteo[base] = var

            var_pct = tk.StringVar(value='')
            tk.Label(celda, textvariable=var_pct, bg=COLOR_PANEL,
                     fg='#607080', font=('Segoe UI', 8)).pack(pady=(0, 2))
            self.var_conteo[base + '_pct'] = var_pct

        # Celda del total, un poco mas ancha.
        celda = tk.Frame(marco, bg='#eef3f8', highlightthickness=1,
                         highlightbackground='#c3ccd6')
        celda.grid(row=0, column=4, padx=4, pady=1, sticky='nsew')
        marco.columnconfigure(4, weight=2)

        fila = tk.Frame(celda, bg='#eef3f8')
        fila.pack(pady=2)
        tk.Label(fila, text='TOTAL', bg='#eef3f8', fg=COLOR_TITULO,
                 font=('Segoe UI', 11, 'bold')).pack(side='left', padx=(0, 8))
        self.var_total = tk.StringVar(value='-')
        tk.Label(fila, textvariable=self.var_total, bg='#eef3f8',
                 font=('Consolas', 12, 'bold')).pack(side='left')

        self.var_resumen_tiempo = tk.StringVar(value='')
        tk.Label(celda, textvariable=self.var_resumen_tiempo, bg='#eef3f8',
                 fg='#607080', font=('Segoe UI', 8)).pack(pady=(0, 2))

    def _construir_panel_graficas(self, padre):
        # El texto del marco se actualiza con los datos del ultimo barrido,
        # asi la figura no gasta alto en un titulo propio.
        self.marco_graficas = ttk.LabelFrame(
            padre, text=' Analisis de rendimiento ',
            style='Panel.TLabelframe', padding=3)
        self.marco_graficas.pack(fill='both', expand=True, pady=(2, 0))

        self.figura = Figure(figsize=(11, 5.0), dpi=100, facecolor=COLOR_PANEL)
        self.lienzo = FigureCanvasTkAgg(self.figura, master=self.marco_graficas)
        self.lienzo.get_tk_widget().pack(fill='both', expand=True)

    # -----------------------------------------------------------------
    # Acciones de la interfaz
    # -----------------------------------------------------------------

    def seleccionar_archivo(self):
        """Abre el dialogo de seleccion de archivo FASTA."""
        if self.trabajando:
            return

        ruta = filedialog.askopenfilename(
            title='Selecciona el archivo FASTA',
            filetypes=[('Archivos FASTA', '*.fna *.fa *.fasta *.ffn *.faa'),
                       ('Todos los archivos', '*.*')])
        if not ruta:
            return

        self.marco_graficas.configure(text=' Analisis de rendimiento ')
        self.ruta_archivo = ruta
        self.tamano_archivo = os.path.getsize(ruta)
        self.var_ruta.set(ruta)
        self.var_tamano.set(benchmark.formato_tamano(self.tamano_archivo))
        self.var_estado.set('Archivo listo. Pulsa Procesar o Analisis completo.')

        # Un archivo nuevo invalida el barrido anterior.
        self.resumen = None
        self.btn_exportar.configure(state='disabled')
        self._limpiar_conteo()
        self._dibujar_graficas(None)

    def procesar_una_configuracion(self):
        """Lanza una sola ejecucion con el numero de procesos elegido."""
        if not self._validar_listo():
            return

        n = int(self.var_procesos.get())
        self._empezar_trabajo()
        hilo = threading.Thread(target=self._hilo_una_configuracion,
                                args=(self.ruta_archivo, n), daemon=True)
        hilo.start()

    def analisis_completo(self):
        """Lanza el barrido con 1, 2, 4, 8 y 12 procesos."""
        if not self._validar_listo():
            return

        self._empezar_trabajo()
        hilo = threading.Thread(target=self._hilo_barrido,
                                args=(self.ruta_archivo,
                                      self.var_secuencial.get()), daemon=True)
        hilo.start()

    def exportar_csv(self):
        """Guarda el ultimo barrido en un archivo CSV."""
        if not self.resumen:
            messagebox.showinfo('Exportar',
                                'Primero ejecuta el analisis completo.')
            return

        base = os.path.splitext(os.path.basename(self.resumen['archivo']))[0]
        propuesto = 'benchmark_%s_%s.csv' % (
            base, time.strftime('%Y%m%d_%H%M%S'))

        ruta = filedialog.asksaveasfilename(
            title='Guardar resultados', defaultextension='.csv',
            initialfile=propuesto,
            filetypes=[('Archivo CSV', '*.csv')])
        if not ruta:
            return

        try:
            benchmark.exportar_csv(self.resumen, ruta)
        except OSError as error:
            messagebox.showerror('Exportar', 'No se pudo guardar:\n%s' % error)
            return

        self.var_estado.set('Resultados exportados a %s' % ruta)
        messagebox.showinfo('Exportar', 'Resultados guardados en:\n%s' % ruta)

    def _validar_listo(self):
        """Comprueba que hay archivo y que no se esta procesando ya."""
        if self.trabajando:
            return False
        if not self.ruta_archivo or not os.path.isfile(self.ruta_archivo):
            messagebox.showwarning('Archivo',
                                   'Selecciona primero un archivo FASTA.')
            return False
        return True

    # -----------------------------------------------------------------
    # Hilos de trabajo
    #
    # Estas funciones NO tocan ningun widget: solo dejan mensajes en la
    # cola, que el hilo principal recoge en _procesar_cola.
    # -----------------------------------------------------------------

    def _hilo_una_configuracion(self, ruta, n):
        def progreso(leidos, total):
            self.cola.put(('progreso', leidos, total))

        try:
            self.cola.put(('estado', 'Procesando con %d proceso(s)...' % n))
            if n == 1:
                conteo, tiempo = motor_adn.contar_secuencial(ruta, progreso)
                etiqueta = 'Secuencial (1 proceso)'
            else:
                conteo, tiempo = motor_adn.contar_paralelo(ruta, n, progreso)
                etiqueta = 'Paralelo con %d procesos' % n
            self.cola.put(('conteo', conteo, tiempo, etiqueta))
        except Exception as error:
            self.cola.put(('error', str(error)))
        finally:
            self.cola.put(('fin',))

    def _hilo_barrido(self, ruta, incluir_secuencial):
        def progreso(leidos, total):
            self.cola.put(('progreso', leidos, total))

        def aviso(texto):
            # Cada configuracion reinicia la barra y el cronometro.
            self.cola.put(('nueva_fase', texto))

        try:
            resumen = benchmark.ejecutar_barrido(
                ruta, procesos=BARRIDO,
                incluir_secuencial=incluir_secuencial,
                progreso=progreso, aviso=aviso)
            self.cola.put(('barrido', resumen))
        except Exception as error:
            self.cola.put(('error', str(error)))
        finally:
            self.cola.put(('fin',))

    # -----------------------------------------------------------------
    # Puente entre el hilo de trabajo y la ventana
    # -----------------------------------------------------------------

    def _procesar_cola(self):
        """Vacia la cola de mensajes y actualiza la ventana."""
        ultimo_progreso = None

        try:
            while True:
                mensaje = self.cola.get_nowait()
                tipo = mensaje[0]

                if tipo == 'progreso':
                    # Solo interesa el ultimo valor de cada ciclo de sondeo.
                    ultimo_progreso = mensaje
                elif tipo == 'estado':
                    self.var_estado.set(mensaje[1])
                elif tipo == 'nueva_fase':
                    self.var_estado.set(mensaje[1])
                    self.inicio_ejecucion = time.perf_counter()
                    self.var_progreso.set(0.0)
                    # Se descarta el progreso de la fase anterior: aplicarlo
                    # con el cronometro recien reiniciado daria una velocidad
                    # absurda.
                    ultimo_progreso = None
                elif tipo == 'conteo':
                    self._mostrar_conteo(mensaje[1], mensaje[2], mensaje[3])
                    ultimo_progreso = None
                elif tipo == 'barrido':
                    self._mostrar_barrido(mensaje[1])
                    ultimo_progreso = None
                elif tipo == 'error':
                    messagebox.showerror('Error durante el proceso', mensaje[1])
                    self.var_estado.set('Error: %s' % mensaje[1])
                elif tipo == 'fin':
                    self._terminar_trabajo()
        except queue.Empty:
            pass

        if ultimo_progreso is not None:
            self._actualizar_progreso(ultimo_progreso[1], ultimo_progreso[2])

        self.raiz.after(100, self._procesar_cola)

    def _actualizar_progreso(self, leidos, total):
        """Refresca la barra, el porcentaje y la velocidad en MB/s."""
        porcentaje = (leidos / total * 100.0) if total else 0.0
        porcentaje = max(0.0, min(100.0, porcentaje))
        self.var_progreso.set(porcentaje)

        transcurrido = time.perf_counter() - self.inicio_ejecucion
        megas = leidos / (1024.0 * 1024.0)
        velocidad = (megas / transcurrido) if transcurrido > 0 else 0.0

        self.var_velocidad.set('%5.1f%%   |   %7.1f MB/s   |   %6.1f s'
                               % (porcentaje, velocidad, transcurrido))

    def _empezar_trabajo(self):
        """Bloquea los botones y reinicia los indicadores."""
        self.trabajando = True
        self.inicio_ejecucion = time.perf_counter()
        self.var_progreso.set(0.0)
        self.var_velocidad.set('  0.0%   |     0.0 MB/s   |    0.0 s')
        self.btn_procesar.configure(state='disabled')
        self.btn_analisis.configure(state='disabled')
        self.btn_exportar.configure(state='disabled')
        self.combo_procesos.configure(state='disabled')

    def _terminar_trabajo(self):
        """Vuelve a habilitar los botones al acabar."""
        self.trabajando = False
        self.var_progreso.set(100.0)
        self.btn_procesar.configure(state='normal')
        self.btn_analisis.configure(state='normal')
        self.combo_procesos.configure(state='readonly')
        if self.resumen:
            self.btn_exportar.configure(state='normal')

    # -----------------------------------------------------------------
    # Presentacion de resultados
    # -----------------------------------------------------------------

    def _limpiar_conteo(self):
        for base in ('A', 'C', 'G', 'T'):
            self.var_conteo[base].set('-')
            self.var_conteo[base + '_pct'].set('')
        self.var_total.set('-')
        self.var_resumen_tiempo.set('')

    def _mostrar_conteo(self, conteo, tiempo, etiqueta):
        """Vuelca el conteo de bases en el panel superior."""
        total = sum(conteo.values())

        for base in ('A', 'C', 'G', 'T'):
            self.var_conteo[base].set('{:,}'.format(conteo[base]).replace(',', '.'))
            porcentaje = (conteo[base] / total * 100.0) if total else 0.0
            self.var_conteo[base + '_pct'].set('%.2f %%' % porcentaje)

        self.var_total.set('{:,}'.format(total).replace(',', '.'))

        megas = self.tamano_archivo / (1024.0 * 1024.0)
        velocidad = (megas / tiempo) if tiempo > 0 else 0.0
        self.var_resumen_tiempo.set('%s  -  %.3f s  -  %.1f MB/s'
                                    % (etiqueta, tiempo, velocidad))
        self.var_estado.set('Listo. %s en %.3f s.' % (etiqueta, tiempo))

        # Cifra definitiva del panel de progreso, ya con el tiempo real.
        self.var_progreso.set(100.0)
        self.var_velocidad.set('%5.1f%%   |   %7.1f MB/s   |   %6.1f s'
                               % (100.0, velocidad, tiempo))

    def _mostrar_barrido(self, resumen):
        """Guarda el barrido, actualiza el conteo y redibuja las graficas."""
        self.resumen = resumen
        # Se muestra la configuracion mas rapida del barrido.
        mejor = min(resumen['resultados'], key=lambda f: f['tiempo'])
        self._mostrar_conteo(resumen['conteo_base'], mejor['tiempo'],
                             'Mejor: %s' % mejor['etiqueta'])
        self._dibujar_graficas(resumen)

        if resumen['coinciden']:
            self.var_estado.set(
                'Analisis completo terminado. Todas las configuraciones '
                'dieron el mismo conteo.')
        else:
            aviso = ('Estas configuraciones NO coinciden con el conteo de '
                     'referencia:\n%s' % ', '.join(resumen['discrepancias']))
            self.var_estado.set('ATENCION: hay conteos que no coinciden.')
            messagebox.showwarning('Verificacion de resultados', aviso)

    def _dibujar_graficas(self, resumen):
        """Dibuja las cuatro graficas del analisis de rendimiento."""
        self.figura.clear()

        if not resumen:
            ejes = self.figura.add_subplot(111)
            ejes.axis('off')
            ejes.text(0.5, 0.5,
                      'Selecciona un archivo FASTA y pulsa\n'
                      '"Analisis completo (1-2-4-8-12)"\n'
                      'para generar las graficas de rendimiento.',
                      ha='center', va='center', fontsize=12, color='#7a8794')
            self.lienzo.draw()
            return

        # Solo las configuraciones paralelas forman las curvas; la medicion
        # secuencial se marca aparte porque no pertenece al barrido.
        filas = [f for f in resumen['resultados'] if f['etiqueta'] != 'Secuencial']
        procesos = [f['procesos'] for f in filas]
        tiempos = [f['tiempo'] for f in filas]
        speedups = [f['speedup'] for f in filas]
        eficiencias = [f['eficiencia'] for f in filas]

        fila_secuencial = next(
            (f for f in resumen['resultados'] if f['etiqueta'] == 'Secuencial'),
            None)

        ejes_tiempo = self.figura.add_subplot(2, 2, 1)
        ejes_tabla = self.figura.add_subplot(2, 2, 2)
        ejes_speedup = self.figura.add_subplot(2, 2, 3)
        ejes_eficiencia = self.figura.add_subplot(2, 2, 4)

        # --- 1) Tiempo contra numero de procesos ---
        ejes_tiempo.plot(procesos, tiempos, marker='o', color='#1565c0',
                         linewidth=2, label='Paralelo')
        if fila_secuencial:
            ejes_tiempo.axhline(fila_secuencial['tiempo'], color='#c62828',
                                linestyle='--', linewidth=1.2,
                                label='Secuencial')
        ejes_tiempo.set_title('Tiempo de ejecucion', fontsize=10, fontweight='bold')
        ejes_tiempo.set_xlabel('Procesos', fontsize=8)
        ejes_tiempo.set_ylabel('Segundos', fontsize=8)
        ejes_tiempo.set_xticks(procesos)
        ejes_tiempo.grid(True, alpha=0.3)
        ejes_tiempo.legend(fontsize=7)
        ejes_tiempo.tick_params(labelsize=8)

        # --- 2) Tabla comparativa ---
        ejes_tabla.axis('off')
        ejes_tabla.set_title('Tabla comparativa', fontsize=10,
                             fontweight='bold', pad=10)

        datos = [['Config.', 'Tiempo', 'Speedup', 'Eficien.', 'MB/s', 'Ok']]
        for f in resumen['resultados']:
            datos.append([
                f['etiqueta'].replace('Paralelo ', 'P-'),
                '%.2f s' % f['tiempo'],
                '%.2fx' % f['speedup'],
                '%.0f%%' % f['eficiencia'],
                '%.0f' % f['mb_por_segundo'],
                'si' if f['coincide'] else 'NO',
            ])

        # Con bbox la tabla ocupa exactamente el area de los ejes, asi no se
        # solapa con el titulo ni se queda diminuta al cambiar el tamano.
        tabla = ejes_tabla.table(cellText=datos[1:], colLabels=datos[0],
                                 cellLoc='center', bbox=[0.0, 0.0, 1.0, 1.0])
        tabla.auto_set_font_size(False)
        tabla.set_fontsize(8)
        for (fila_i, columna), celda in tabla.get_celld().items():
            celda.set_edgecolor('#c3ccd6')
            if fila_i == 0:
                celda.set_facecolor('#1f3a5f')
                celda.set_text_props(color='white', fontweight='bold')
            elif fila_i % 2 == 0:
                celda.set_facecolor('#f4f6f8')

        # Se resalta en rojo cualquier configuracion cuyo conteo no coincida.
        for indice, f in enumerate(resumen['resultados'], start=1):
            if not f['coincide']:
                for columna in range(len(datos[0])):
                    tabla[indice, columna].set_facecolor('#ffd9d9')

        # --- 3) Speedup contra numero de procesos ---
        ejes_speedup.plot(procesos, speedups, marker='o', color='#2e7d32',
                          linewidth=2, label='Speedup real')
        ejes_speedup.plot(procesos, procesos, linestyle=':', color='#9aa5b1',
                          linewidth=1.2, label='Speedup ideal')
        ejes_speedup.set_title('Speedup', fontsize=10, fontweight='bold')
        ejes_speedup.set_xlabel('Procesos', fontsize=8)
        ejes_speedup.set_ylabel('t(1) / t(n)', fontsize=8)
        ejes_speedup.set_xticks(procesos)
        ejes_speedup.grid(True, alpha=0.3)
        ejes_speedup.legend(fontsize=7)
        ejes_speedup.tick_params(labelsize=8)

        # --- 4) Eficiencia contra numero de procesos ---
        barras = ejes_eficiencia.bar([str(n) for n in procesos], eficiencias,
                                     color='#ef6c00', width=0.55)
        ejes_eficiencia.axhline(100, color='#9aa5b1', linestyle=':', linewidth=1.2)
        ejes_eficiencia.set_title('Eficiencia', fontsize=10, fontweight='bold')
        ejes_eficiencia.set_xlabel('Procesos', fontsize=8)
        ejes_eficiencia.set_ylabel('Porcentaje', fontsize=8)
        ejes_eficiencia.set_ylim(0, max(110.0, max(eficiencias) * 1.15))
        ejes_eficiencia.grid(True, axis='y', alpha=0.3)
        ejes_eficiencia.tick_params(labelsize=8)

        for barra, valor in zip(barras, eficiencias):
            ejes_eficiencia.text(barra.get_x() + barra.get_width() / 2,
                                 barra.get_height() + 2, '%.0f%%' % valor,
                                 ha='center', fontsize=7)

        self.marco_graficas.configure(
            text=' Analisis de rendimiento - %s - %s - %d nucleos fisicos '
                 '/ %d logicos '
                 % (os.path.basename(resumen['archivo']),
                    benchmark.formato_tamano(resumen['tamano']),
                    resumen['fisicos'], resumen['logicos']))

        # Margenes fijos en fraccion de figura: se mantienen bien cuando el
        # usuario cambia el tamano de la ventana. tight_layout no se usa
        # porque la tabla con bbox lo descuadra.
        self.figura.subplots_adjust(left=0.075, right=0.975, top=0.92,
                                    bottom=0.10, wspace=0.20, hspace=0.45)
        self.lienzo.draw()


def main():
    raiz = tk.Tk()
    Aplicacion(raiz)
    raiz.mainloop()


if __name__ == '__main__':
    # multiprocessing en Windows exige este guard: sin el, cada proceso hijo
    # volveria a abrir la ventana en un bucle infinito.
    multiprocessing.freeze_support()
    main()
