# -*- coding: utf-8 -*-
"""
generar_informe_docx.py

Construye el informe del P1.2 en formato Word, siguiendo la misma estructura
del informe entregado en el P1.1. Toma las cifras de los resultados medidos y
las capturas de la aplicacion desde la carpeta resultados.

Uso:
    python generar_informe_docx.py
"""

import os

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

SALIDA = 'resultados'
DESTINO = 'P1.2_Colaborativo_VivancoAlex.docx'

# Mediciones sobre el genoma humano completo GRCh38.p14 (3.11 GB).
GENOMA = {
    'A': 923117203, 'C': 642552917, 'G': 645231996, 'T': 925917038,
    'N': 161611379, 'iupac': 103, 'invalidos': 0,
    'cpu_seq': 51.463, 'cpu_par': 17.469, 'gpu': 3.443,
    'mbs_seq': 61.9, 'mbs_par': 182.3, 'mbs_gpu': 925.2,
}
GENOMA['bases'] = GENOMA['A'] + GENOMA['C'] + GENOMA['G'] + GENOMA['T']

# Barrido sobre el recorte de 500 MB: (etiqueta, tiempo, mb/s, speedup, efic.)
BARRIDO_CPU = [
    ('CPU x1 (secuencial)', 7.305, 68.4, 1.00, 100.0),
    ('CPU x2', 5.432, 92.0, 1.34, 67.2),
    ('CPU x4', 4.444, 112.5, 1.64, 41.1),
    ('CPU x8', 4.282, 116.8, 1.71, 21.3),
    ('CPU x12', 4.779, 104.6, 1.53, 12.7),
]
BARRIDO_GPU = [
    ('GPU lote 8 MB', 0.596, 839.6, 12.27),
    ('GPU lote 16 MB', 0.531, 941.1, 13.75),
    ('GPU lote 32 MB', 0.518, 965.8, 14.11),
    ('GPU lote 64 MB', 0.447, 1117.9, 16.33),
    ('GPU lote 128 MB', 0.482, 1037.6, 15.16),
    ('GPU lote 256 MB', 0.495, 1009.2, 14.74),
]
ESCALABILIDAD = [
    ('50 MB', 0.859, 2.510, 0.090),
    ('200 MB', 2.678, 3.659, 0.334),
    ('500 MB', 7.283, 5.005, 0.717),
]
IUPAC = [
    ('Y', 36, 'C o T (pirimidina)'),
    ('R', 29, 'A o G (purina)'),
    ('W', 15, 'A o T (enlace debil)'),
    ('K', 8, 'G o T (ceto)'),
    ('M', 8, 'A o C (amino)'),
    ('S', 5, 'G o C (enlace fuerte)'),
    ('B', 2, 'C, G o T (no A)'),
]


def miles(n):
    return '{:,}'.format(n).replace(',', '.')


class Informe:
    def __init__(self):
        self.doc = Document()
        estilo = self.doc.styles['Normal']
        estilo.font.name = 'Calibri'
        estilo.font.size = Pt(11)
        self.figura = 0

    # -- utilidades de escritura ----------------------------------------

    def titulo(self, texto, tamano=14, espacio_antes=14):
        p = self.doc.add_paragraph()
        p.paragraph_format.space_before = Pt(espacio_antes)
        p.paragraph_format.space_after = Pt(6)
        r = p.add_run(texto)
        r.bold = True
        r.font.size = Pt(tamano)
        return p

    def parrafo(self, texto, justificado=True):
        p = self.doc.add_paragraph(texto)
        if justificado:
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_after = Pt(8)
        return p

    def vineta(self, texto):
        p = self.doc.add_paragraph(texto, style='List Bullet')
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_after = Pt(4)
        return p

    def imagen(self, ruta, pie, ancho=6.3):
        if not os.path.exists(ruta):
            print('  AVISO: falta la imagen %s' % ruta)
            return
        self.doc.add_picture(ruta, width=Inches(ancho))
        self.doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        self.figura += 1
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run('Figura %d. %s' % (self.figura, pie))
        r.italic = True
        r.font.size = Pt(9)
        p.paragraph_format.space_after = Pt(12)

    def tabla(self, cabecera, filas, anchos=None):
        t = self.doc.add_table(rows=1, cols=len(cabecera))
        t.style = 'Light Grid Accent 1'
        for celda, texto in zip(t.rows[0].cells, cabecera):
            celda.text = ''
            r = celda.paragraphs[0].add_run(texto)
            r.bold = True
            r.font.size = Pt(9)
        for fila in filas:
            celdas = t.add_row().cells
            for celda, texto in zip(celdas, fila):
                celda.text = ''
                r = celda.paragraphs[0].add_run(str(texto))
                r.font.size = Pt(9)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(6)
        return t

    # -- contenido -------------------------------------------------------

    def portada(self):
        for texto, tam, negrita in (
            ('PONTIFICIA UNIVERSIDAD CATOLICA DEL ECUADOR', 13, True),
            ('Facultad de Ingenieria', 12, False),
            ('Ingenieria en Sistemas y Computacion', 12, False),
        ):
            p = self.doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(texto)
            r.bold = negrita
            r.font.size = Pt(tam)

        self.doc.add_paragraph()
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run('PROYECTO P1.2')
        r.bold = True
        r.font.size = Pt(20)

        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run('Comparacion del Rendimiento entre CPU y GPU en el '
                      'Procesamiento Paralelo de Cadenas de ADN')
        r.bold = True
        r.font.size = Pt(14)

        self.doc.add_paragraph()
        for texto in ('Asignatura: Computacion Paralela, Distribuida y en la Nube',
                      'Docente: Ing. JW Condor',
                      'Autor: Alex Vivanco'):
            p = self.doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.add_run(texto).font.size = Pt(11)

    def resumen(self):
        self.titulo('1. Resumen')
        self.parrafo(
            'Este informe presenta el desarrollo de una aplicacion en Python que '
            'procesa el archivo FASTA del genoma humano GRCh38.p14, de 3,11 GB, '
            'comparando el rendimiento de dos plataformas de computo paralelo: la '
            'CPU multinucleo mediante multiprocessing y la GPU mediante kernels '
            'CUDA escritos con Numba. Ademas del conteo de los nucleotidos A, C, G '
            'y T, la aplicacion identifica los caracteres invalidos presentes en la '
            'secuencia y los distingue de los codigos IUPAC de ambiguedad, que son '
            'notacion valida del formato.')
        self.parrafo(
            'Sobre el archivo completo, la ejecucion secuencial en CPU tomo %.3f '
            'segundos, la ejecucion paralela con doce procesos tomo %.3f segundos '
            'y la ejecucion en GPU tomo %.3f segundos, lo que representa un speedup '
            'de %.2f veces frente a la version secuencial y de %.2f veces frente a '
            'la mejor configuracion de CPU. Las tres ejecuciones produjeron conteos '
            'identicos, verificados sobre el histograma completo de 256 valores de '
            'byte y no unicamente sobre las cuatro bases.'
            % (GENOMA['cpu_seq'], GENOMA['cpu_par'], GENOMA['gpu'],
               GENOMA['cpu_seq'] / GENOMA['gpu'],
               GENOMA['cpu_par'] / GENOMA['gpu']))

    def problema(self):
        self.titulo('2. Planteamiento del problema')
        self.parrafo(
            'El ADN se representa computacionalmente como una cadena de caracteres '
            'construida sobre un alfabeto de cuatro simbolos que corresponden a las '
            'bases nitrogenadas adenina, citosina, guanina y timina. El conteo de '
            'frecuencias sobre esa cadena es una operacion algoritmicamente trivial, '
            'pero deja de serlo cuando el volumen de datos alcanza varios gigabytes, '
            'porque el costo deja de estar en el calculo y pasa a estar en el acceso '
            'a los datos.')
        self.parrafo(
            'El proyecto P1.1 resolvio este problema repartiendo el trabajo entre los '
            'nucleos de la CPU. El presente proyecto plantea una pregunta distinta: '
            'si una GPU, disenada para ejecutar miles de hilos simples en paralelo, '
            'resulta mas adecuada que la CPU para este tipo de carga, y bajo que '
            'condiciones. La respuesta no es evidente de antemano, porque la GPU '
            'introduce un costo que la CPU no tiene: los datos deben viajar por el bus '
            'PCIe desde la memoria principal hasta la memoria de la tarjeta antes de '
            'poder ser procesados.')
        self.parrafo(
            'El archivo utilizado sigue el formato FASTA, en el que las lineas que '
            'comienzan con el simbolo mayor que corresponden a cabeceras descriptivas '
            'y deben excluirse del conteo. Las lineas de secuencia contienen ademas la '
            'letra N para bases desconocidas y emplean minusculas como marcado de '
            'regiones repetitivas, que deben contarse igual que las mayusculas.')

    def objetivos(self):
        self.titulo('3. Objetivos')
        for texto in (
            'Desarrollar una aplicacion que procese un archivo FASTA de varios '
            'gigabytes empleando paralelismo sobre GPU mediante CUDA.',
            'Implementar una version equivalente sobre CPU multinucleo que sirva '
            'de termino de comparacion valido.',
            'Identificar los caracteres invalidos presentes en la secuencia y '
            'distinguirlos de los codigos validos del formato.',
            'Medir y comparar el desempeno de ambas plataformas en terminos de '
            'tiempo de ejecucion, velocidad de procesamiento y uso de recursos.',
            'Determinar el tamano de lote optimo para el balance de carga de la GPU.',
            'Analizar la escalabilidad de ambas plataformas frente a volumenes '
            'crecientes de datos e identificar los cuellos de botella.',
        ):
            self.vineta(texto)

    def ambiente(self):
        self.titulo('4. Ambiente de pruebas')
        self.parrafo(
            'Todas las mediciones se realizaron sobre el mismo equipo y bajo las '
            'mismas condiciones. Las especificaciones son las siguientes:')
        for texto in (
            'Procesador: Intel Core i5-13420H, 13.a generacion, arquitectura hibrida.',
            'Nucleos fisicos: 8. Procesadores logicos: 12.',
            'Tarjeta grafica: NVIDIA GeForce RTX 4050 Laptop GPU, arquitectura Ada '
            'Lovelace, compute capability 8.9, 20 multiprocesadores de flujo (SM), '
            '6140 MB de memoria VRAM, tamano de warp de 32 hilos.',
            'Memoria RAM: 32 GB. Almacenamiento: unidad de estado solido NVMe.',
            'Sistema operativo: Windows 11.',
            'Lenguaje: Python 3.13, con las librerias numba 0.67 y numba-cuda 0.30 '
            'para la GPU, multiprocessing para la CPU, y psutil, pynvml, matplotlib '
            'y tkinter para el monitoreo y la interfaz.',
            'Archivo de entrada: GCF_000001405.40_GRCh38.p14_genomic.fna, de 3,11 GB.',
        ):
            self.vineta(texto)
        self.imagen(os.path.join(SALIDA, 'ui_graficas.png'),
                    'Aplicacion desarrollada, con el hardware detectado en el '
                    'panel izquierdo y los resultados del analisis completo.')

    def diseno(self):
        self.titulo('5. Diseno de la solucion')

        self.titulo('5.1 Un mismo algoritmo sobre dos plataformas', 12, 10)
        self.parrafo(
            'La decision de diseno mas importante del proyecto fue que ambos motores '
            'ejecutaran exactamente el mismo algoritmo: el calculo de un histograma '
            'completo de 256 posiciones, una por cada valor posible de byte, del que '
            'se derivan despues todos los conteos. Si la CPU emplease un atajo '
            'distinto al de la GPU, los tiempos medidos no serian comparables y la '
            'comparativa perderia todo su valor. De este modo la unica variable entre '
            'una medicion y otra es el hardware que ejecuta, no el algoritmo.')

        self.titulo('5.2 Descarte de cabeceras dentro de la GPU', 12, 10)
        self.parrafo(
            'Las lineas de cabecera deben excluirse del conteo. La solucion sencilla '
            'seria limpiarlas en el procesador antes de enviar el lote a la tarjeta, '
            'pero eso significaria que la CPU realiza parte del trabajo de la GPU y '
            'falsearia la comparacion. Por ello el descarte se resuelve integramente '
            'dentro de la tarjeta, mediante tres kernels encadenados:')
        for texto in (
            'El primer kernel asigna un hilo a cada byte del lote y marca las '
            'posiciones donde comienza una cabecera, es decir, aquellos simbolos '
            'mayor que precedidos de un salto de linea. Las posiciones se acumulan '
            'mediante una operacion atomica sobre un contador compartido.',
            'El segundo kernel asigna un bloque a cada cabecera encontrada. El primer '
            'hilo del bloque localiza el final de la linea y el resto de hilos la '
            'sobreescribe en paralelo con saltos de linea, que el histograma ya '
            'ignora. Las cabeceras son pocas y cortas, de modo que el costo es '
            'despreciable.',
            'El tercer kernel calcula el histograma. Cada bloque acumula primero en '
            'un vector de 256 posiciones alojado en memoria compartida, que reside '
            'dentro del multiprocesador y es mucho mas rapida que la memoria global, '
            'y solo al terminar vuelca su resultado parcial al histograma global. Sin '
            'esta acumulacion intermedia, millones de hilos competirian atomicamente '
            'por las mismas cuatro posiciones y el kernel se serializaria.',
        ):
            self.vineta(texto)
        self.parrafo(
            'El segundo kernel se lanza con una malla de tamano fijo y cada bloque '
            'consulta en memoria de dispositivo cuantas cabeceras hay que procesar. '
            'Esto evita tener que copiar ese numero al procesador entre un kernel y '
            'el siguiente, operacion que obligaria a sincronizar y romperia el '
            'solapamiento entre transferencia y computo.')

        self.titulo('5.3 Procesamiento por lotes y solapamiento', 12, 10)
        self.parrafo(
            'El archivo ocupa 3,11 GB y la tarjeta dispone de 6140 MB de memoria, de '
            'los cuales el sistema operativo ya reserva una parte. Cargar el archivo '
            'completo no es viable. La aplicacion lo procesa por lotes, empleando dos '
            'buferes de memoria de pagina bloqueada, requisito para que las '
            'transferencias sean asincronas, y dos flujos CUDA que se alternan. De '
            'este modo la transferencia de un lote se solapa con el computo del lote '
            'anterior. El tamano de lote es configurable, ya que constituye el '
            'parametro de balance de carga evaluado en la seccion de resultados.')

        self.titulo('5.4 Separacion de la compilacion del tiempo medido', 12, 10)
        self.parrafo(
            'Numba compila cada kernel la primera vez que se invoca, operacion que en '
            'este equipo tarda entre 1,9 y 2,8 segundos. Si ese tiempo se incluyera en '
            'la medicion, se le estaria atribuyendo a la GPU un costo que no pertenece '
            'al algoritmo, y en archivos pequenos la compilacion pesaria mas que el '
            'procesamiento. La aplicacion compila los kernels de forma explicita antes '
            'de iniciar cualquier cronometro.')

    def implementacion(self):
        self.titulo('6. Implementacion')
        self.parrafo(
            'La solucion se estructuro en modulos independientes. El motor de CPU y el '
            'motor de GPU exponen la misma interfaz y devuelven el mismo diccionario '
            'de resultados. Un modulo de monitoreo muestrea el uso de recursos en un '
            'hilo de fondo; un generador produce archivos de prueba con errores '
            'inyectados; un banco de pruebas ejecuta los barridos comparativos; y la '
            'interfaz grafica reune todo lo anterior.')
        self.parrafo(
            'El procesamiento se ejecuta siempre en un hilo separado del hilo de la '
            'interfaz. Dado que Tkinter no admite el acceso a sus componentes desde '
            'varios hilos, el hilo trabajador nunca modifica un componente de forma '
            'directa: deposita mensajes en una cola que la ventana vacia '
            'periodicamente. Este esquema es el que permite que la aplicacion siga '
            'respondiendo mientras procesa un archivo de tres gigabytes.')
        self.parrafo(
            'Un detalle relevante del comportamiento en Windows condiciono la '
            'arquitectura de archivos. En este sistema multiprocessing emplea el '
            'metodo spawn, de modo que cada proceso hijo vuelve a importar el modulo '
            'principal. Durante el desarrollo se midio que la simple presencia de la '
            'importacion del motor de GPU en el modulo principal elevaba el tiempo de '
            'una ejecucion paralela sobre un archivo de 2 MB de 1,34 a 5,51 segundos, '
            'porque los doce procesos hijos cargaban la infraestructura de CUDA sin '
            'llegar a utilizarla. Ese costo se le habria atribuido erroneamente a la '
            'CPU. Por ello el archivo de arranque de la aplicacion contiene unicamente '
            'el lanzador, y tanto la ventana como los motores se importan de forma '
            'diferida.')
        self.imagen(os.path.join(SALIDA, 'ui_tabla.png'),
                    'Tabla comparativa de la aplicacion con todas las '
                    'configuraciones medidas y la verificacion de conteos.')

    def resultados(self):
        self.titulo('7. Resultados')

        self.titulo('7.1 Conteo de nucleotidos', 12, 10)
        self.parrafo(
            'El procesamiento del archivo completo arrojo los siguientes resultados, '
            'identicos en las tres modalidades de ejecucion:')
        total = GENOMA['bases']
        self.tabla(
            ['Base', 'Ocurrencias', 'Porcentaje', 'Par complementario'],
            [['Adenina (A)', miles(GENOMA['A']),
              '%.2f %%' % (100.0 * GENOMA['A'] / total), 'Timina'],
             ['Citosina (C)', miles(GENOMA['C']),
              '%.2f %%' % (100.0 * GENOMA['C'] / total), 'Guanina'],
             ['Guanina (G)', miles(GENOMA['G']),
              '%.2f %%' % (100.0 * GENOMA['G'] / total), 'Citosina'],
             ['Timina (T)', miles(GENOMA['T']),
              '%.2f %%' % (100.0 * GENOMA['T'] / total), 'Adenina'],
             ['Total de bases', miles(total), '100,00 %', '-'],
             ['N (desconocidas)', miles(GENOMA['N']), '-', 'No aplica']])
        self.parrafo(
            'Los valores obtenidos permiten una validacion adicional del resultado. '
            'Dado que el ADN es una molecula de doble hebra en la que la adenina se '
            'empareja siempre con la timina y la citosina con la guanina, en un genoma '
            'completo ambas cantidades deben ser practicamente iguales. La diferencia '
            'observada entre adenina y timina es de apenas el %.2f por ciento, y entre '
            'citosina y guanina del %.2f por ciento, lo que confirma la correctitud del '
            'conteo por una via independiente del propio programa.'
            % (100.0 * abs(GENOMA['A'] - GENOMA['T']) / GENOMA['T'],
               100.0 * abs(GENOMA['C'] - GENOMA['G']) / GENOMA['G']))

        self.titulo('7.2 Identificacion de caracteres invalidos', 12, 10)
        self.parrafo(
            'El analisis del archivo completo revelo 103 caracteres que no son ni las '
            'cuatro bases ni la letra N. Un examen de su naturaleza mostro que no se '
            'trata de errores sino de codigos IUPAC de ambiguedad, notacion estandar y '
            'valida del formato FASTA para posiciones en las que la secuenciacion no '
            'resolvio una base concreta pero si acoto las posibilidades:')
        self.tabla(['Codigo', 'Ocurrencias', 'Significado'],
                   [[c, miles(n), s] for c, n, s in IUPAC])
        self.parrafo(
            'Clasificar estos codigos como caracteres invalidos habria constituido un '
            'error conceptual, ya que no son basura sino informacion parcial. Por ello '
            'la aplicacion los contabiliza en una categoria propia, separada tanto de '
            'las bases como de los errores.')
        self.parrafo(
            'Ahora bien, el archivo real no contiene ningun caracter invalido en '
            'sentido estricto, por lo que no permite demostrar que el detector '
            'funciona. Para obtener una prueba verificable se desarrollo un generador '
            'que produce copias del genoma con una cantidad y un tipo de error '
            'conocidos de antemano, sustituyendo caracteres de las lineas de secuencia '
            'sin alterar el tamano ni la estructura del archivo. Se contemplan cuatro '
            'familias de error: letras ajenas al alfabeto del ADN, digitos, simbolos y '
            'espacios en blanco, siendo estos ultimos los mas dificiles de advertir '
            'porque no son visibles al abrir el archivo. Sobre un archivo de prueba con '
            '1094 errores inyectados de 24 tipos distintos, las tres modalidades de '
            'ejecucion reprodujeron exactamente la cantidad y el desglose esperados.')

        self.titulo('7.3 Desempeno sobre el archivo completo', 12, 10)
        self.parrafo(
            'La tabla siguiente resume las mediciones sobre el genoma completo de '
            '3,11 GB. El speedup se calcula respecto de la ejecucion secuencial.')
        self.tabla(
            ['Configuracion', 'Tiempo (s)', 'MB/s', 'Speedup', 'Conteo identico'],
            [['CPU secuencial', '%.3f' % GENOMA['cpu_seq'],
              '%.1f' % GENOMA['mbs_seq'], '1,00x', 'Si'],
             ['CPU paralelo x12', '%.3f' % GENOMA['cpu_par'],
              '%.1f' % GENOMA['mbs_par'],
              '%.2fx' % (GENOMA['cpu_seq'] / GENOMA['cpu_par']), 'Si'],
             ['GPU (lote 128 MB)', '%.3f' % GENOMA['gpu'],
              '%.1f' % GENOMA['mbs_gpu'],
              '%.2fx' % (GENOMA['cpu_seq'] / GENOMA['gpu']), 'Si']])

        self.titulo('7.4 Barrido de configuraciones', 12, 10)
        self.parrafo(
            'Para caracterizar el comportamiento de ambas plataformas se realizo un '
            'barrido detallado sobre un recorte de 500 MB del mismo archivo, que '
            'permite repetir muchas configuraciones en un tiempo razonable.')
        self.tabla(['Configuracion', 'Tiempo (s)', 'MB/s', 'Speedup', 'Eficiencia'],
                   [[e, '%.3f' % t, '%.1f' % m, '%.2fx' % s, '%.1f %%' % ef]
                    for e, t, m, s, ef in BARRIDO_CPU])
        self.tabla(['Configuracion', 'Tiempo (s)', 'MB/s', 'Speedup'],
                   [[e, '%.3f' % t, '%.1f' % m, '%.2fx' % s]
                    for e, t, m, s in BARRIDO_GPU])
        self.imagen(os.path.join(SALIDA, 'graficas.png'),
                    'Tiempo por configuracion, speedup, eficiencia de la CPU y '
                    'escalabilidad de ambas plataformas.')

        self.titulo('7.5 Escalabilidad', 12, 10)
        self.parrafo(
            'El ultimo barrido midio el comportamiento de ambas plataformas frente a '
            'volumenes crecientes de datos, con el proposito de determinar a partir de '
            'que tamano compensa el costo fijo de la GPU.')
        self.tabla(['Tamano', 'CPU x1 (s)', 'CPU x12 (s)', 'GPU (s)',
                    'GPU frente a CPU x1'],
                   [[e, '%.3f' % c1, '%.3f' % c12, '%.3f' % g, '%.2fx' % (c1 / g)]
                    for e, c1, c12, g in ESCALABILIDAD])

    def analisis(self):
        self.titulo('8. Analisis critico')

        self.titulo('8.1 Origen de la ventaja de la GPU', 12, 10)
        self.parrafo(
            'La GPU resulto %.2f veces mas rapida que la ejecucion secuencial y %.2f '
            'veces mas rapida que la mejor configuracion de CPU sobre el archivo '
            'completo. La razon de esta diferencia esta en la naturaleza del problema. '
            'El conteo de frecuencias es una operacion sin dependencias entre '
            'elementos: el resultado de un caracter no condiciona el de ningun otro. '
            'Esta caracteristica se ajusta con precision al modelo de ejecucion de una '
            'GPU, que dispone de veinte multiprocesadores capaces de mantener miles de '
            'hilos simultaneos ejecutando la misma instruccion sobre datos distintos. '
            'La CPU, en cambio, dispone de doce procesadores logicos que ademas deben '
            'atender al sistema operativo.'
            % (GENOMA['cpu_seq'] / GENOMA['gpu'],
               GENOMA['cpu_par'] / GENOMA['gpu']))

        self.titulo('8.2 El verdadero cuello de botella', 12, 10)
        self.parrafo(
            'La velocidad maxima alcanzada por la GPU fue de 1117,9 MB/s sobre el '
            'recorte de 500 MB y de 925,2 MB/s sobre el archivo completo. Estas cifras '
            'no reflejan la capacidad de calculo de la tarjeta sino la velocidad de '
            'lectura del disco de estado solido. El kernel de histograma procesa los '
            'datos mucho mas rapido de lo que el subsistema de almacenamiento consigue '
            'suministrarlos, de modo que durante la mayor parte de la ejecucion la GPU '
            'permanece a la espera.')
        self.parrafo(
            'Esta observacion tiene una consecuencia importante para la '
            'interpretacion de los resultados: el speedup medido no expresa la '
            'superioridad computacional de la GPU sobre la CPU, sino unicamente que la '
            'GPU alcanza antes el limite impuesto por el almacenamiento. Si el archivo '
            'residiera integramente en memoria, la diferencia entre ambas plataformas '
            'seria considerablemente mayor.')

        self.titulo('8.3 Tamano de lote y balance de carga', 12, 10)
        self.parrafo(
            'El barrido del tamano de lote mostro un optimo claro en 64 MB, con 1117,9 '
            'MB/s, frente a los 839,6 MB/s obtenidos con lotes de 8 MB y los 1009,2 '
            'MB/s con lotes de 256 MB. El comportamiento responde a dos efectos '
            'opuestos. Con lotes pequenos se desaprovecha el ancho de banda del bus '
            'PCIe y se multiplica el numero de lanzamientos de kernel, cada uno con su '
            'propia latencia de inicio. Con lotes grandes, en cambio, el procesador '
            'debe llenar por completo el bufer antes de que la tarjeta pueda comenzar a '
            'trabajar, lo que retrasa el inicio del computo y reduce el solapamiento '
            'efectivo entre transferencia y calculo. El valor de 64 MB equilibra ambos '
            'efectos en este equipo.')

        self.titulo('8.4 Caida de la eficiencia en la CPU', 12, 10)
        self.parrafo(
            'La eficiencia de la CPU descendio de forma sostenida desde el 100 por '
            'ciento con un proceso hasta el 12,7 por ciento con doce. El motivo es que '
            'el problema esta limitado por la entrada y salida de datos, no por el '
            'calculo: anadir procesos incorpora unidades de ejecucion, pero no anade '
            'ancho de banda de disco. A partir de ocho procesos el tiempo dejo incluso '
            'de mejorar y con doce empeoro, pasando de 4,282 a 4,779 segundos, porque '
            'los cuatro procesos adicionales compiten por los mismos recursos fisicos.')
        self.parrafo(
            'A ello se suma la naturaleza hibrida del procesador empleado. Los ocho '
            'nucleos fisicos no son equivalentes entre si: parte de ellos son nucleos '
            'de rendimiento y el resto nucleos de eficiencia, de menor capacidad. Al '
            'pasar de ocho a doce procesos no se incorporan nucleos adicionales sino '
            'hilos logicos que comparten recursos con los ya activos.')

        self.titulo('8.5 Cuando la paralelizacion resulta contraproducente', 12, 10)
        self.parrafo(
            'El barrido de escalabilidad dejo al descubierto un resultado que conviene '
            'destacar. Sobre el recorte de 50 MB, la ejecucion paralela con doce '
            'procesos tardo 2,510 segundos frente a los 0,859 segundos de la ejecucion '
            'secuencial, es decir, resulto casi tres veces mas lenta. La causa es el '
            'costo de creacion de los procesos: en Windows cada proceso hijo arranca un '
            'interprete nuevo y vuelve a importar los modulos necesarios, operacion '
            'que sobre un archivo pequeno consume mas tiempo que el propio '
            'procesamiento. Solo a partir de los 500 MB la version paralela recupera la '
            'ventaja. La GPU, por el contrario, se mantuvo por delante en los tres '
            'tamanos evaluados, porque su costo fijo se limita a la transferencia y no '
            'incluye la creacion de procesos del sistema operativo.')

        self.titulo('8.6 Consideracion metodologica', 12, 10)
        self.parrafo(
            'El equipo dispone de 32 GB de memoria RAM y el archivo procesado ocupa '
            '3,11 GB, por lo que el sistema operativo puede mantener una fraccion '
            'considerable del archivo en su cache de disco entre una ejecucion y la '
            'siguiente. Las mediciones sucesivas de un mismo barrido se benefician por '
            'tanto de lecturas mas rapidas que la primera. Dado que este efecto '
            'favorece por igual a ambas plataformas, no invalida la comparacion, pero '
            'si implica que las cifras absolutas de velocidad deben interpretarse como '
            'un limite superior optimista.')

    def aplicaciones(self):
        self.titulo('9. Aplicaciones reales')
        self.parrafo(
            'El escenario planteado en este proyecto constituye una version '
            'simplificada de problemas que se presentan de forma cotidiana en '
            'bioinformatica. El alineamiento de secuencias contra un genoma de '
            'referencia, la deteccion de variantes geneticas, el calculo del contenido '
            'de guanina y citosina por regiones o el control de calidad de los datos '
            'producidos por un secuenciador operan sobre volumenes comparables o '
            'superiores.')
        self.parrafo(
            'Estas operaciones comparten la caracteristica observada aqui: son '
            'algoritmicamente simples pero se aplican sobre cantidades de datos muy '
            'grandes, lo que las convierte en candidatas naturales para la '
            'paralelizacion en GPU. De hecho, herramientas de uso extendido en el '
            'sector, como los alineadores acelerados por hardware, se fundamentan '
            'precisamente en este principio. El patron trasciende ademas el ambito '
            'biologico: el mismo esquema de histograma paralelo sobre memoria '
            'compartida es el que sustenta el analisis de registros de servidores, el '
            'procesamiento de imagenes y la construccion de indices sobre grandes '
            'colecciones de texto.')

    def conclusiones(self):
        self.titulo('10. Conclusiones')
        for texto in (
            'La ejecucion en GPU redujo el tiempo de procesamiento del genoma '
            'completo de %.3f a %.3f segundos frente a la version secuencial, lo que '
            'representa un speedup de %.2f veces, y de %.3f a %.3f segundos frente a '
            'la mejor configuracion de CPU, con un speedup de %.2f veces.'
            % (GENOMA['cpu_seq'], GENOMA['gpu'], GENOMA['cpu_seq'] / GENOMA['gpu'],
               GENOMA['cpu_par'], GENOMA['gpu'], GENOMA['cpu_par'] / GENOMA['gpu']),
            'La correctitud se mantuvo en todas las configuraciones evaluadas. La '
            'verificacion se realizo sobre el histograma completo de 256 valores de '
            'byte, criterio mas exigente que la simple comparacion de las cuatro '
            'bases, y todas las ejecuciones coincidieron.',
            'La proporcion obtenida entre bases complementarias, con diferencias '
            'inferiores al 0,31 por ciento entre adenina y timina y al 0,42 por ciento '
            'entre citosina y guanina, valida el resultado por una via biologica '
            'independiente del programa.',
            'El cuello de botella del sistema no reside en el calculo sino en el '
            'acceso a los datos. La GPU alcanza el limite del almacenamiento con '
            'holgura, mientras que la CPU necesita ocho procesos para aproximarse a el.',
            'El tamano de lote optimo para esta tarjeta se situo en 64 MB, valor que '
            'equilibra el aprovechamiento del bus PCIe con el solapamiento entre '
            'transferencia y computo.',
            'La paralelizacion no resulta ventajosa de forma incondicional. Sobre '
            'archivos pequenos el costo de creacion de procesos supera al beneficio, '
            'hasta el punto de que la version paralela puede ser tres veces mas lenta '
            'que la secuencial.',
            'Los 103 caracteres no convencionales hallados en el genoma resultaron ser '
            'codigos IUPAC de ambiguedad y no errores, por lo que se clasificaron en '
            'una categoria propia. La capacidad de deteccion de errores reales se '
            'valido mediante archivos generados con errores conocidos.',
        ):
            self.vineta(texto)

    def recomendaciones(self):
        self.titulo('11. Recomendaciones')
        for texto in (
            'Evaluar la lectura asincrona del archivo mediante varios hilos de '
            'entrada y salida, con el fin de determinar si es posible superar el '
            'limite de velocidad impuesto por el acceso secuencial al disco.',
            'Repetir cada configuracion varias veces y reportar el promedio, de modo '
            'que se reduzca el efecto de la cache del sistema operativo y de la '
            'variabilidad propia de la planificacion.',
            'Extender la comparacion a la tarjeta grafica integrada del procesador '
            'mediante OpenCL, con el proposito de contrastar el rendimiento de una '
            'unidad discreta frente a una integrada.',
            'Explorar el uso de memoria de textura o de instrucciones vectorizadas de '
            'carga en el kernel de histograma, para el caso en que los datos '
            'estuvieran ya residentes en memoria y el disco dejara de ser el factor '
            'limitante.',
            'Considerar la implementacion del motor de CPU en un lenguaje compilado, '
            'con el objeto de establecer el limite superior real de la plataforma y '
            'obtener una comparacion mas estricta frente a la GPU.',
        ):
            self.vineta(texto)

    def referencias(self):
        self.titulo('12. Referencias')
        for texto in (
            'NVIDIA Corporation. CUDA C++ Programming Guide. Disponible en: '
            'https://docs.nvidia.com/cuda/cuda-c-programming-guide/',
            'Anaconda Inc. Numba Documentation: CUDA Python. Disponible en: '
            'https://numba.readthedocs.io/en/stable/cuda/index.html',
            'Python Software Foundation. Documentacion oficial del modulo '
            'multiprocessing. Disponible en: '
            'https://docs.python.org/3/library/multiprocessing.html',
            'National Center for Biotechnology Information. Genome Reference '
            'Consortium Human Build 38 (GRCh38). Disponible en: '
            'https://www.ncbi.nlm.nih.gov/assembly/GCF_000001405.40/',
            'Cornish-Bowden, A. Nomenclature for incompletely specified bases in '
            'nucleic acid sequences. Nucleic Acids Research, vol. 13, num. 9, 1985.',
            'NVIDIA Corporation. Especificaciones de la arquitectura Ada Lovelace y '
            'de la serie GeForce RTX 40. Disponible en: https://www.nvidia.com/',
        ):
            self.vineta(texto)

    def generar(self):
        self.portada()
        self.doc.add_page_break()
        self.resumen()
        self.problema()
        self.objetivos()
        self.ambiente()
        self.diseno()
        self.implementacion()
        self.resultados()
        self.analisis()
        self.aplicaciones()
        self.conclusiones()
        self.recomendaciones()
        self.referencias()
        self.doc.save(DESTINO)
        print('Informe generado: %s' % os.path.abspath(DESTINO))
        print('Figuras incluidas: %d' % self.figura)


if __name__ == '__main__':
    Informe().generar()
