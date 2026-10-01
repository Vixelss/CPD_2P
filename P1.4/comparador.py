# -*- coding: utf-8 -*-
r"""
comparador.py

Base comun de los tres motores del P1.4. Aqui vive todo lo que NO depende de
la plataforma: como se lee un FASTA, como se convierte una posicion en fila y
columna, y como se presenta el resultado.

QUE PROBLEMA RESUELVE EL P1.4
Comparar dos cadenas de ADN y decir en que posiciones difieren, indicando la
fila y la columna de cada diferencia. Es un problema distinto al del P1.3:
alli se contaba un solo archivo, aqui se enfrentan dos.

POR QUE LA SECUENCIA SE EXTRAE ANTES Y FUERA DEL CRONOMETRO
Un archivo FASTA no es secuencia pura: lleva lineas de cabecera que empiezan
con '>' y un salto de linea cada 80 caracteres. Ninguna de esas dos cosas es
ADN, y por tanto no se comparan.

La extraccion se hace UNA VEZ, antes de medir, y los tres motores reciben
exactamente el mismo vector de bytes ya limpio. Se hace asi por tres razones:

  1. Comparabilidad. Si cada motor limpiara el archivo por su cuenta se
     estaria midiendo tambien la limpieza, que es trabajo de entrada y salida
     y no de comparacion. Los tiempos dejarian de hablar del hardware. Es el
     mismo criterio con el que en el P1.2 y el P1.3 se dejaba la compilacion
     de los kernels fuera del reloj.

  2. La NPU no tiene otra opcion. Una NPU no lee archivos: recibe un tensor
     ya formado. Si la CPU y la GPU leyeran del disco y la NPU no, la
     comparacion entre las tres plataformas seria una trampa.

  3. Los dos genomas del proyecto ocupan 3.3 GB cada uno. Extraerlos en cada
     medicion costaria mas que todas las mediciones juntas.

POR QUE UN ARCHIVO .seq Y NO MEMORIA
La primera version cargaba las dos secuencias en memoria y las copiaba a un
bloque de memoria compartida para que los procesos hijos las vieran. Con los
recortes de prueba funcionaba, pero con los genomas completos son 6.7 GB de
secuencia mas otros 6.7 GB de copia compartida: mas de 13 GB solo para poder
empezar.

Ahora la secuencia limpia se escribe una vez en un archivo .seq, que es ADN
crudo sin cabeceras ni saltos, y todos los procesos lo abren con numpy.memmap.
Un archivo mapeado no se copia a cada proceso: el sistema operativo hace que
las mismas paginas fisicas aparezcan en todos, y ademas las descarta solo si
hace falta memoria. Resultado: los doce procesos comparten un unico ejemplar
de cada cadena y el consumo baja de 13 GB a los 6.7 GB de la cache de disco,
que ni siquiera es memoria reservada por el programa.

El .seq se cachea: si ya existe y es mas nuevo que el FASTA del que salio, se
reutiliza y la preparacion pasa a costar milisegundos.

FILA Y COLUMNA
El enunciado pide senalar la fila y la columna de cada diferencia. Se usa la
disposicion del propio formato FASTA, que escribe la secuencia en lineas de
80 caracteres:

    fila    = posicion // 80 + 1
    columna = posicion  % 80 + 1

Ambas empiezan en 1, no en 0, porque es un dato para leer en un informe y no
un indice de programacion. La posicion lineal tambien se publica, que es la
que usa cualquier herramienta posterior.

API publica:
    preparar(ruta, cache='datos')       -> Cadena (secuencia mapeada)
    posicion_a_fila_columna(posicion)   -> (fila, columna)
    filas_columnas(posiciones)          -> (filas, columnas) vectorizado
    resumir(...)                        -> dict de resultados comun
    formatear_diferencias(resultado)    -> texto para consola e informe
"""

import os
import time

import numpy as np


# Ancho de linea del FASTA estandar. Es lo que define la rejilla de filas y
# columnas sobre la que se informan las diferencias.
ANCHO_LINEA = 80

# Tamano de bloque de lectura. El mismo valor que usan los motores del P1.2 y
# del P1.3, por coherencia entre proyectos.
_BLOQUE = 8 * 1024 * 1024

# Bytes con significado propio dentro del formato.
COD_MAYOR = 62          # '>' abre una linea de cabecera
COD_SALTO = 10          # '\n'
COD_RETORNO = 13        # '\r' de los finales de linea de Windows

# Cuantas diferencias se guardan con su detalle completo. El conteo total
# siempre es exacto; lo que se acota es la lista que se muestra, porque dos
# cadenas muy distintas pueden dar millones de diferencias y ni el informe ni
# la ventana pueden con eso.
MAX_DETALLE = 1000

# Extension del archivo de secuencia limpia.
EXT_SEQ = '.seq'

# Extension de la huella que acompana a cada archivo de cache (.seq, .idx) y
# dice de que FASTA exacto salio. Ver cache_al_dia().
EXT_ORIGEN = '.origen'


# ---------------------------------------------------------------------------
# Errores propios
# ---------------------------------------------------------------------------

class ErrorEntrada(Exception):
    """Problema con un archivo de entrada que el usuario puede corregir.

    Se distingue de cualquier otra excepcion a proposito: la ventana la
    muestra tal cual en un cuadro de dialogo, porque su texto esta escrito
    para que lo lea una persona y no para depurar.
    """


# ---------------------------------------------------------------------------
# Cadena preparada
# ---------------------------------------------------------------------------

class Cadena:
    """Una secuencia de ADN lista para comparar, respaldada por un archivo.

    Envuelve el mapeo de memoria para que el resto del programa no tenga que
    saber si la secuencia esta en RAM o en disco. Se comporta como un vector
    de numpy: se puede indexar, rebanar y pasar a cualquiera de los tres
    motores.
    """

    def __init__(self, ruta_origen, ruta_seq, largo, preparacion_s,
                 desde_cache):
        self.ruta_origen = ruta_origen
        self.ruta_seq = ruta_seq
        self.largo = int(largo)
        self.preparacion_s = float(preparacion_s)
        self.desde_cache = bool(desde_cache)
        self._datos = None

    @property
    def datos(self):
        """Vector uint8 con la secuencia, mapeado desde el archivo .seq."""
        if self._datos is None:
            if self.largo == 0:
                self._datos = np.empty(0, dtype=np.uint8)
            else:
                self._datos = np.memmap(self.ruta_seq, dtype=np.uint8,
                                        mode='r', shape=(self.largo,))
        return self._datos

    @property
    def mb(self):
        return self.largo / (1024.0 * 1024.0)

    @property
    def filas(self):
        """Cuantas lineas de 80 caracteres ocupa esta secuencia."""
        if self.largo == 0:
            return 0
        return (self.largo + ANCHO_LINEA - 1) // ANCHO_LINEA

    @property
    def desplazamiento(self):
        """Donde empieza dentro de su .seq. Una Cadena entera empieza en 0.

        Existe para que los motores traten igual a una Cadena completa y a un
        Tramo suyo: los dos dicen su archivo, su desplazamiento y su largo.
        """
        return 0

    def tramo(self, inicio, largo):
        """Devuelve un Tramo de esta cadena, sin copiar nada."""
        return Tramo(self, inicio, largo)

    def cerrar(self):
        """Suelta el mapeo. El archivo .seq se conserva para la proxima vez."""
        self._datos = None

    def __len__(self):
        return self.largo

    def __getitem__(self, indice):
        return self.datos[indice]

    def __repr__(self):
        return '<Cadena %s: %d bases>' % (os.path.basename(self.ruta_origen),
                                          self.largo)


class Tramo:
    """Una porcion de una Cadena, lista para comparar.

    POR QUE HACE FALTA
    Los dos genomas del proyecto tienen las mismas secuencias en DISTINTO
    ORDEN, asi que el cromosoma 12 empieza en la posicion 1.945.677.585 de
    uno y en la 1.945.577.269 del otro. Compararlos exige enfrentar dos
    porciones que arrancan en sitios diferentes, y hasta ahora los motores
    solo sabian comparar dos cadenas enteras desde el byte cero.

    Un Tramo expone lo mismo que una Cadena (datos, largo, ruta_seq y
    desplazamiento), de modo que los tres motores funcionan con cualquiera de
    los dos sin enterarse de la diferencia. No copia memoria: 'datos' es una
    vista del mismo archivo mapeado.
    """

    def __init__(self, cadena, inicio, largo):
        if inicio < 0 or largo < 0 or inicio + largo > cadena.largo:
            raise ErrorEntrada(
                'El tramo [%d, %d) no cabe en una cadena de %d bases.'
                % (inicio, inicio + largo, cadena.largo))
        self.cadena = cadena
        self.inicio = int(inicio)
        self.largo = int(largo)

    @property
    def ruta_origen(self):
        return self.cadena.ruta_origen

    @property
    def ruta_seq(self):
        return self.cadena.ruta_seq

    @property
    def desplazamiento(self):
        return self.inicio

    @property
    def datos(self):
        return self.cadena.datos[self.inicio:self.inicio + self.largo]

    @property
    def mb(self):
        return self.largo / (1024.0 * 1024.0)

    @property
    def filas(self):
        if self.largo == 0:
            return 0
        return (self.largo + ANCHO_LINEA - 1) // ANCHO_LINEA

    def cerrar(self):
        """No hace nada: el mapeo pertenece a la Cadena, no al Tramo."""

    def __len__(self):
        return self.largo

    def __getitem__(self, indice):
        return self.datos[indice]

    def __repr__(self):
        return '<Tramo %d+%d de %s>' % (
            self.inicio, self.largo,
            os.path.basename(self.cadena.ruta_origen))


# ---------------------------------------------------------------------------
# Preparacion
# ---------------------------------------------------------------------------

def preparar(ruta, cache='datos', progreso=None, forzar=False):
    """Extrae la secuencia limpia de un FASTA y la deja lista para comparar.

    Escribe (o reutiliza) un archivo .seq con el ADN sin cabeceras ni saltos
    de linea, y devuelve un objeto Cadena que lo mapea en memoria.

    Parametros:
        ruta     : archivo FASTA de entrada.
        cache    : carpeta donde dejar el .seq.
        progreso : funcion opcional progreso(bytes_hechos, bytes_totales).
        forzar   : rehace el .seq aunque ya exista y este al dia.

    Lanza ErrorEntrada con un mensaje legible si el archivo no sirve.
    """
    validar_fasta(ruta)

    if not os.path.isdir(cache):
        os.makedirs(cache)

    base = os.path.splitext(os.path.basename(ruta))[0]
    ruta_seq = os.path.join(cache, base + EXT_SEQ)

    # Cache valida: el .seq existe y salio de este mismo FASTA.
    if not forzar and cache_al_dia(ruta, ruta_seq):
        largo = os.path.getsize(ruta_seq)
        if progreso is not None:
            progreso(1, 1)
        return Cadena(ruta, ruta_seq, largo, 0.0, True)

    inicio = time.perf_counter()
    tamano = os.path.getsize(ruta)
    leidos = 0
    escritos = 0
    # La huella se toma ANTES de leer: si el FASTA cambiara mientras se
    # extrae, la huella guardada ya no coincidiria y se rehace la proxima vez.
    huella = huella_origen(ruta)

    # El .seq se escribe primero con nombre temporal y se renombra al final.
    # Asi una extraccion interrumpida (cierre de la ventana, corte de luz) no
    # deja una cache a medias que la proxima ejecucion daria por buena.
    temporal = ruta_seq + '.parcial'
    try:
        with open(ruta, 'rb') as entrada, open(temporal, 'wb') as salida:
            for bloque in _bloques_alineados(entrada):
                limpio = _limpiar_bloque(bloque)
                if limpio:
                    salida.write(limpio)
                    escritos += len(limpio)
                leidos += len(bloque)
                if progreso is not None:
                    progreso(leidos, tamano)

        # La huella vieja se borra antes de tocar el .seq: si algo se corta a
        # partir de aqui, queda un .seq sin huella, que no se da por bueno.
        borrar_origen(ruta_seq)
        if os.path.exists(ruta_seq):
            os.remove(ruta_seq)
        os.rename(temporal, ruta_seq)
    except Exception:
        if os.path.exists(temporal):
            try:
                os.remove(temporal)
            except OSError:
                pass
        raise

    if escritos == 0:
        raise ErrorEntrada(
            'El archivo "%s" no contiene ninguna secuencia de ADN.\n'
            'Solo se encontraron lineas de cabecera o el archivo esta vacio.'
            % os.path.basename(ruta))

    anotar_origen(ruta_seq, huella)

    if progreso is not None:
        progreso(tamano, tamano)

    return Cadena(ruta, ruta_seq, escritos, time.perf_counter() - inicio,
                  False)


def huella_origen(ruta):
    """Tamano y fecha exacta (en nanosegundos) de un archivo, como texto."""
    datos = os.stat(ruta)
    return '%d %d' % (datos.st_size, datos.st_mtime_ns)


def cache_al_dia(ruta_origen, ruta_cache):
    """Dice si un archivo de cache (.seq, .idx) salio de este FASTA exacto.

    TRAMPA YA PISADA. La cache se guarda por nombre de archivo, y antes se
    daba por buena si era mas nueva que el FASTA. Eso falla cuando dos FASTA
    distintos se llaman igual: el segundo reutiliza la cache del primero y se
    compara otra cadena sin que nada avise. Windows lo pone facil, porque al
    copiar un archivo conserva su fecha de modificacion. Paso de verdad: el
    indice de par_200MB_B quedo vacio y el emparejamiento por secuencia se
    desactivaba en silencio.

    Por eso junto a cada cache se guarda la huella del FASTA del que salio
    (tamano y fecha exacta) y se exige que coincida con la del FASTA actual.
    Una cache sin huella, o con otra, se rehace.
    """
    if not os.path.exists(ruta_cache) or os.path.getsize(ruta_cache) == 0:
        return False
    try:
        with open(ruta_cache + EXT_ORIGEN, encoding='utf-8') as f:
            guardada = f.read().strip()
        return guardada == huella_origen(ruta_origen)
    except OSError:
        return False


def anotar_origen(ruta_cache, huella):
    """Guarda junto a la cache la huella del FASTA del que salio."""
    with open(ruta_cache + EXT_ORIGEN, 'w', encoding='utf-8') as f:
        f.write(huella)


def borrar_origen(ruta_cache):
    """Invalida una cache borrando su huella."""
    try:
        os.remove(ruta_cache + EXT_ORIGEN)
    except OSError:
        pass


def _bloques_alineados(f):
    """Va entregando bloques del archivo que terminan en linea completa.

    Que cada bloque acabe justo despues de un salto de linea es lo que permite
    decidir sin ambiguedad si una linea es cabecera: basta con mirar su primer
    byte. Es la misma regla de alineacion que usan los motores de los
    proyectos anteriores.
    """
    while True:
        bloque = f.read(_BLOQUE)
        if not bloque:
            return
        if not bloque.endswith(b'\n'):
            resto = f.readline()
            if resto:
                bloque += resto
        yield bloque


def _limpiar_bloque(bloque):
    """Devuelve solo los bytes de secuencia de un bloque alineado a lineas.

    Se descartan las lineas de cabecera enteras y, de las lineas de secuencia,
    los separadores de linea. Lo que sale es ADN puro.
    """
    if b'>' in bloque:
        # Ruta lenta, poco frecuente: hay cabeceras en este bloque.
        lineas = bloque.split(b'\n')
        bloque = b''.join(l for l in lineas if not l.startswith(b'>'))
    else:
        # Ruta rapida: no hay cabeceras, basta con quitar los separadores.
        bloque = bloque.replace(b'\n', b'')

    if b'\r' in bloque:
        bloque = bloque.replace(b'\r', b'')

    return bloque


# ---------------------------------------------------------------------------
# Validacion de entrada
# ---------------------------------------------------------------------------

def validar_fasta(ruta):
    """Comprueba que la ruta apunta a un FASTA que se puede leer.

    Todas las comprobaciones lanzan ErrorEntrada con un mensaje que explica
    que pasa y que hacer, porque el destinatario es quien usa el programa y no
    quien lo escribio.
    """
    if not ruta:
        raise ErrorEntrada('No se ha elegido ningun archivo.')

    if not os.path.exists(ruta):
        raise ErrorEntrada('No existe el archivo:\n%s' % ruta)

    if os.path.isdir(ruta):
        raise ErrorEntrada('"%s" es una carpeta, no un archivo.'
                           % os.path.basename(ruta))

    if os.path.getsize(ruta) == 0:
        raise ErrorEntrada('El archivo "%s" esta vacio.'
                           % os.path.basename(ruta))

    try:
        with open(ruta, 'rb') as f:
            muestra = f.read(4096)
    except PermissionError:
        raise ErrorEntrada(
            'No hay permiso para leer "%s".\n'
            'Puede estar abierto en otro programa.' % os.path.basename(ruta))
    except OSError as error:
        raise ErrorEntrada('No se pudo leer "%s":\n%s'
                           % (os.path.basename(ruta), error))

    if b'\x00' in muestra:
        raise ErrorEntrada(
            'El archivo "%s" parece ser binario, no un FASTA de texto.'
            % os.path.basename(ruta))

    # Un FASTA valido empieza por una cabecera. Se admite tambien un archivo
    # de secuencia suelta, que es lo que genera este mismo programa, pero se
    # rechaza cualquier otra cosa antes de gastar minutos leyendo 3 GB.
    primero = muestra.lstrip()[:1]
    if primero and primero != b'>' and not _parece_secuencia(muestra):
        raise ErrorEntrada(
            'El archivo "%s" no parece un FASTA.\n'
            'Deberia empezar por una linea de cabecera con ">".'
            % os.path.basename(ruta))

    return True


def _parece_secuencia(muestra):
    """Dice si una muestra de bytes parece ADN suelto sin cabecera."""
    utiles = muestra.replace(b'\n', b'').replace(b'\r', b'')
    if not utiles:
        return False
    validos = sum(1 for b in utiles.upper()
                  if b in b'ACGTNRYSWKMBDHV')
    return validos / len(utiles) > 0.9


def validar_par(cadena_a, cadena_b):
    """Comprueba que dos cadenas se pueden comparar entre si.

    Devuelve una lista de avisos: cosas que no impiden comparar pero que hay
    que decirle al usuario para que no interprete mal el resultado. Si la
    comparacion es imposible, lanza ErrorEntrada.
    """
    avisos = []

    if cadena_a.largo == 0 or cadena_b.largo == 0:
        raise ErrorEntrada('Una de las dos cadenas esta vacia.')

    if cadena_a.ruta_origen == cadena_b.ruta_origen:
        avisos.append('Las dos cadenas son el mismo archivo: el resultado '
                      'sera cero diferencias.')

    desfase = cadena_a.largo - cadena_b.largo
    if desfase:
        comun = min(cadena_a.largo, cadena_b.largo)
        avisos.append(
            'Las cadenas miden distinto: %d y %d bases (%s%d). Se compararan '
            'las %d posiciones que tienen en comun; el sobrante de la mas '
            'larga no tiene pareja contra la que compararse.'
            % (cadena_a.largo, cadena_b.largo,
               '+' if desfase > 0 else '', desfase, comun))

    return avisos


# ---------------------------------------------------------------------------
# Rejilla de filas y columnas
# ---------------------------------------------------------------------------

def posicion_a_fila_columna(posicion, ancho=ANCHO_LINEA):
    """Convierte una posicion lineal en (fila, columna), ambas desde 1."""
    return int(posicion) // ancho + 1, int(posicion) % ancho + 1


def filas_columnas(posiciones, ancho=ANCHO_LINEA):
    """Version vectorizada: convierte un array de posiciones de una vez.

    Se hace con numpy y no con un bucle de Python porque el vector de
    posiciones puede tener millones de elementos, y recorrerlo en Python
    costaria mas que la propia comparacion que acaba de hacer la GPU.
    """
    posiciones = np.asarray(posiciones, dtype=np.int64)
    return posiciones // ancho + 1, posiciones % ancho + 1


# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------

def resumir(total, posiciones, cadena_a, cadena_b, comparados, plataforma,
            detalle=None):
    """Construye el diccionario de resultados que devuelven los tres motores.

    'total' es cuantas diferencias hay en realidad y 'posiciones' son las
    primeras de ellas, que pueden ser muchas menos. Se separan los dos datos a
    proposito: comparar dos genomas completos da mil millones de diferencias,
    y guardarlas todas serian 8.6 GB de indices. El total siempre es exacto;
    lo que se acota es la lista que se muestra.

    Los tres motores devuelven exactamente esta misma estructura, de modo que
    el resto del programa no necesita saber quien hizo el trabajo.
    """
    posiciones = np.asarray(posiciones, dtype=np.int64)
    total = int(total)

    if detalle is None:
        detalle = MAX_DETALLE
    recorte = posiciones if detalle < 0 else posiciones[:detalle]

    diferencias = []
    if recorte.size:
        filas, columnas = filas_columnas(recorte)
        # Se leen de golpe los caracteres implicados en vez de uno a uno: con
        # una cadena mapeada en disco, mil accesos sueltos son mil saltos.
        letras_a = np.asarray(cadena_a.datos)[recorte]
        letras_b = np.asarray(cadena_b.datos)[recorte]
        for p, fi, co, la, lb in zip(recorte, filas, columnas,
                                     letras_a, letras_b):
            diferencias.append({
                'posicion': int(p),
                'fila': int(fi),
                'columna': int(co),
                'a': chr(int(la)),
                'b': chr(int(lb)),
            })

    # Validacion de sanidad. No puede haber mas diferencias que posiciones
    # comparadas, ni un numero negativo de ellas: si el conteo sale de ahi es
    # que el motor devolvio basura y el resultado NO se puede publicar.
    #
    # No es una precaucion teorica. El motor de NPU devolvia veinte digitos
    # por un fallo de DirectML con tensores vacios (ver motor_npu, el
    # comentario del bucle de lotes), y el numero se mostraba tan campante en
    # la ventana y en la matriz de evidencias. Esta comprobacion es la red
    # que impide que vuelva a pasar con cualquier otro motor.
    valido = 0 <= total <= comparados
    motivo = None
    if not valido:
        motivo = ('El motor devolvio %d diferencias sobre %d posiciones '
                  'comparadas, lo que es imposible. El resultado esta mal y '
                  'no se publica.' % (total, comparados))

    iguales = comparados - total
    return {
        'plataforma': plataforma,
        'comparados': int(comparados),
        'iguales': int(iguales) if valido else 0,
        'diferencias': total,
        'valido': valido,
        'motivo_invalido': motivo,
        'similitud_pct': ((100.0 * iguales / comparados)
                          if (comparados and valido) else 0.0),
        'detalle': diferencias,
        'detalle_truncado': total > len(diferencias),
        'largo_a': int(cadena_a.largo),
        'largo_b': int(cadena_b.largo),
        'desfase': int(cadena_a.largo - cadena_b.largo),
    }


def formatear_diferencias(resultado, maximo=20):
    """Texto corto con las primeras diferencias, para consola e informe."""
    if not resultado['diferencias']:
        return ('Las dos cadenas son identicas en las %d posiciones '
                'comparadas.' % resultado['comparados'])

    lineas = [
        'Diferencias: %d de %d posiciones comparadas  (%.6f %% iguales)'
        % (resultado['diferencias'], resultado['comparados'],
           resultado['similitud_pct']),
        '',
        '  %12s %10s %8s   %s' % ('POSICION', 'FILA', 'COLUMNA', 'A -> B'),
        '  ' + '-' * 50,
    ]
    for d in resultado['detalle'][:maximo]:
        lineas.append('  %12d %10d %8d   %s -> %s'
                      % (d['posicion'], d['fila'], d['columna'],
                         d['a'], d['b']))

    mostradas = min(maximo, len(resultado['detalle']))
    restantes = resultado['diferencias'] - mostradas
    if restantes > 0:
        lineas.append('  ... y %d diferencias mas' % restantes)

    if resultado['desfase']:
        lineas.append('')
        lineas.append('  AVISO: las cadenas miden distinto (%d frente a %d).'
                      % (resultado['largo_a'], resultado['largo_b']))
        lineas.append('  Se comparo hasta donde alcanzan las dos.')

    return '\n'.join(lineas)


# ---------------------------------------------------------------------------
# Prueba rapida por linea de comandos
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print('Uso: python comparador.py <archivo.fna> [carpeta_cache]')
        print('')
        print('Extrae la secuencia limpia y muestra sus caracteristicas.')
        sys.exit(0)

    destino = sys.argv[2] if len(sys.argv) > 2 else 'datos'

    def avance(hechos, total):
        sys.stdout.write('\r  preparando... %5.1f %%'
                         % (100.0 * hechos / total if total else 100.0))
        sys.stdout.flush()

    try:
        cadena = preparar(sys.argv[1], cache=destino, progreso=avance)
    except ErrorEntrada as error:
        print('')
        print('ERROR: %s' % error)
        sys.exit(1)

    print('')
    print('Archivo     : %s' % cadena.ruta_origen)
    print('Tamano      : %.1f MB'
          % (os.path.getsize(cadena.ruta_origen) / (1024 * 1024)))
    print('Secuencia   : %d bases (%.1f MB)' % (cadena.largo, cadena.mb))
    print('Archivo .seq: %s' % cadena.ruta_seq)
    print('Filas de %d : %d' % (ANCHO_LINEA, cadena.filas))
    print('Preparacion : %s'
          % ('reutilizada de la cache'
             if cadena.desde_cache else '%.3f s' % cadena.preparacion_s))
