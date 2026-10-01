# -*- coding: utf-8 -*-
r"""
secuencias.py

Indice de los registros de un FASTA y emparejamiento entre dos archivos.

POR QUE HACE FALTA ESTO
Comparar dos archivos posicion a posicion solo tiene sentido si las dos
secuencias estan alineadas. Con los dos genomas del proyecto NO lo estan, y
la razon se midio exactamente:

  Los archivos son GCA (GenBank) y GCF (RefSeq) del MISMO ensamblaje
  GRCh38.p14. Sus primeros 29 registros coinciden al byte: los cromosomas 1
  al 11 estan en el mismo sitio y miden lo mismo. El registro numero 30 es
  donde se rompe:

    A (GenBank)  KI270721.1  chromosome 11 unlocalized genomic contig
                 100.316 bases
    B (RefSeq)   NC_000012.12  chromosome 12, Primary Assembly
                 133.275.309 bases

  GenBank intercala los contigs no ubicados detras de su cromosoma; RefSeq
  los agrupa en otro sitio. A partir de ahi todo queda desplazado.

  Se ve en los offsets del cromosoma 12: A lo tiene en 1.945.677.585 y B en
  1.945.577.269. La diferencia es 100.316, exactamente la longitud del
  contig que GenBank metio en medio.

CONSECUENCIA MEDIDA: comparar los dos archivos de corrido daba 1.072.801.765
diferencias y un 67 % de similitud. Emparejando las secuencias por nombre,
CADA CROMOSOMA SALE IDENTICO, cero diferencias. Los mil millones eran
artefacto del desalineamiento, no diferencias geneticas.

COMO SE EMPAREJA
GenBank y RefSeq usan identificadores distintos (CM000663.2 frente a
NC_000001.11), asi que no se puede emparejar por el codigo de la cabecera.
Se usan dos claves, en este orden:

  1. Por cromosoma. La descripcion de los dos formatos contiene el texto
     "chromosome N, ... primary assembly" (o "mitochondrion"). Son 25
     registros, cubren el 93,6 % de las bases y los 25 tienen exactamente la
     misma longitud en los dos archivos.

  2. Por longitud unica. Para el resto (contigs, scaffolds y parches), una
     longitud que aparece UNA sola vez en cada archivo identifica el registro
     sin ambiguedad. Medido: empareja 701 de los 709 registros y cubre el
     99,99 % de las bases.

Lo que no empareja por ninguna de las dos vias se informa aparte y NO se
compara, que es lo unico honesto: un registro sin pareja no tiene contra que
compararse.

API publica:
    indexar(ruta, cache='datos')     -> lista de Registro
    emparejar(regs_a, regs_b)        -> (parejas, solo_a, solo_b)
    nombre_corto(cabecera)           -> etiqueta legible
"""

import os
import re

import comparador

# Bloque de lectura para recorrer el archivo buscando cabeceras.
_BLOQUE = 8 * 1024 * 1024

# Extension del archivo de indice.
EXT_INDICE = '.idx'

# Cromosoma dentro de una descripcion de ensamblaje principal. Vale para los
# dos formatos:
#   >CM000663.2 Homo sapiens chromosome 1, GRCh38 reference primary assembly
#   >NC_000001.11 Homo sapiens chromosome 1, GRCh38.p14 Primary Assembly
_CROMOSOMA = re.compile(r'chromosome\s+([0-9]{1,2}|[XY])\s*,'
                        r'[^,]*primary\s+assembly', re.I)

# El ADN mitocondrial no lleva numero de cromosoma y se nombra aparte.
_MITOCONDRIA = re.compile(r'mitochondrion', re.I)


class Registro:
    """Un registro de un FASTA: su cabecera y donde cae en la secuencia.

    'inicio' y 'largo' estan en coordenadas de la SECUENCIA LIMPIA, la del
    archivo .seq, no del archivo original. Es decir, ya sin cabeceras ni
    saltos de linea, que es donde comparan los motores.
    """

    def __init__(self, cabecera, inicio, largo):
        self.cabecera = cabecera
        self.inicio = int(inicio)
        self.largo = int(largo)

    @property
    def identificador(self):
        """El codigo de acceso, lo primero de la cabecera tras el '>'."""
        return self.cabecera[1:].split(None, 1)[0] if self.cabecera else ''

    @property
    def descripcion(self):
        partes = self.cabecera[1:].split(None, 1)
        return partes[1] if len(partes) > 1 else ''

    @property
    def cromosoma(self):
        """Clave de cromosoma, o None si el registro no es uno principal."""
        encontrado = _CROMOSOMA.search(self.descripcion)
        if encontrado:
            return encontrado.group(1).upper()
        if _MITOCONDRIA.search(self.descripcion):
            return 'MT'
        return None

    def __repr__(self):
        return '<Registro %s %d+%d>' % (self.identificador, self.inicio,
                                        self.largo)


def nombre_corto(registro):
    """Etiqueta legible para tablas y graficas."""
    clave = registro.cromosoma
    if clave:
        return 'MT' if clave == 'MT' else 'Cr %s' % clave
    return registro.identificador


# ---------------------------------------------------------------------------
# Indice
# ---------------------------------------------------------------------------

def indexar(ruta, cache='datos', progreso=None, forzar=False):
    """Devuelve la lista de Registro de un FASTA, cacheando el resultado.

    Recorrer 3,3 GB buscando cabeceras cuesta unos segundos, y el indice no
    cambia mientras no cambie el archivo. Se guarda junto al .seq, con la
    misma regla de validez: se reutiliza solo si su huella coincide con la
    del FASTA (ver comparador.cache_al_dia).
    """
    if not os.path.isdir(cache):
        os.makedirs(cache)

    base = os.path.splitext(os.path.basename(ruta))[0]
    ruta_idx = os.path.join(cache, base + EXT_INDICE)

    if not forzar and comparador.cache_al_dia(ruta, ruta_idx):
        return _leer_indice(ruta_idx)

    huella = comparador.huella_origen(ruta)
    registros = _recorrer(ruta, progreso)
    comparador.borrar_origen(ruta_idx)
    _escribir_indice(registros, ruta_idx)
    comparador.anotar_origen(ruta_idx, huella)
    return registros


def _recorrer(ruta, progreso=None):
    """Recorre el FASTA anotando donde empieza y cuanto mide cada registro.

    Se lleva la cuenta de la secuencia LIMPIA: cada linea de datos suma su
    longitud sin el salto, y las cabeceras no suman nada. Asi los offsets que
    salen de aqui son exactamente los del archivo .seq que usan los motores.
    """
    registros = []
    cabecera = None
    inicio = 0
    acumulado = 0
    resto = b''
    leidos = 0
    tamano = os.path.getsize(ruta)

    with open(ruta, 'rb') as f:
        while True:
            bloque = f.read(_BLOQUE)
            if not bloque:
                break
            leidos += len(bloque)

            datos = resto + bloque
            corte = datos.rfind(b'\n')
            if corte < 0:
                # Linea absurdamente larga: se acumula y se sigue leyendo.
                resto = datos
                continue
            resto = datos[corte + 1:]
            datos = datos[:corte + 1]

            for linea in datos.split(b'\n'):
                if not linea:
                    continue
                if linea.startswith(b'>'):
                    if cabecera is not None:
                        registros.append(Registro(cabecera, inicio,
                                                  acumulado - inicio))
                    cabecera = linea.decode('ascii', 'replace').strip()
                    inicio = acumulado
                else:
                    acumulado += len(linea.rstrip(b'\r'))

            if progreso is not None:
                progreso(leidos, tamano)

    # La cola que quedo sin salto de linea al final del archivo.
    if resto:
        linea = resto.rstrip(b'\r\n')
        if linea and not linea.startswith(b'>'):
            acumulado += len(linea)
    if cabecera is not None:
        registros.append(Registro(cabecera, inicio, acumulado - inicio))

    if progreso is not None:
        progreso(tamano, tamano)
    return registros


def _escribir_indice(registros, destino):
    temporal = destino + '.parcial'
    with open(temporal, 'w', encoding='utf-8') as f:
        f.write('# inicio\tlargo\tcabecera\n')
        for r in registros:
            f.write('%d\t%d\t%s\n' % (r.inicio, r.largo, r.cabecera))
    if os.path.exists(destino):
        os.remove(destino)
    os.rename(temporal, destino)


def _leer_indice(ruta):
    registros = []
    with open(ruta, encoding='utf-8') as f:
        for linea in f:
            if linea.startswith('#'):
                continue
            partes = linea.rstrip('\n').split('\t', 2)
            if len(partes) == 3:
                registros.append(Registro(partes[2], int(partes[0]),
                                          int(partes[1])))
    return registros


# ---------------------------------------------------------------------------
# Emparejamiento
# ---------------------------------------------------------------------------

class Pareja:
    """Dos registros que se corresponden entre los dos archivos."""

    def __init__(self, a, b, clave, criterio):
        self.a = a
        self.b = b
        self.clave = clave
        self.criterio = criterio      # 'cromosoma' o 'longitud'

    @property
    def largo(self):
        """Posiciones comparables: hasta donde alcanzan las dos."""
        return min(self.a.largo, self.b.largo)

    @property
    def desfase(self):
        return self.a.largo - self.b.largo

    def __repr__(self):
        return '<Pareja %s por %s>' % (self.clave, self.criterio)


def emparejar(regs_a, regs_b):
    """Empareja los registros de dos archivos.

    Devuelve (parejas, solo_a, solo_b). Las parejas salen ordenadas: primero
    los cromosomas principales, en orden natural, y luego el resto por
    posicion en el archivo A.
    """
    pendientes_a = list(regs_a)
    pendientes_b = list(regs_b)
    parejas = []

    # -- 1) por cromosoma -------------------------------------------------
    por_crom_b = {}
    for r in pendientes_b:
        clave = r.cromosoma
        if clave:
            # Si un cromosoma apareciera dos veces no se puede decidir, asi
            # que se marca como ambiguo y se deja para la clave de longitud.
            por_crom_b.setdefault(clave, []).append(r)

    usados_b = set()
    for r in pendientes_a:
        clave = r.cromosoma
        if not clave:
            continue
        candidatos = por_crom_b.get(clave) or []
        if len(candidatos) != 1:
            continue
        pareja_b = candidatos[0]
        if id(pareja_b) in usados_b:
            continue
        usados_b.add(id(pareja_b))
        parejas.append(Pareja(r, pareja_b, clave, 'cromosoma'))

    emparejados_a = {id(p.a) for p in parejas}
    pendientes_a = [r for r in pendientes_a if id(r) not in emparejados_a]
    pendientes_b = [r for r in pendientes_b if id(r) not in usados_b]

    # -- 2) por longitud unica --------------------------------------------
    conteo_a = {}
    conteo_b = {}
    for r in pendientes_a:
        conteo_a[r.largo] = conteo_a.get(r.largo, 0) + 1
    for r in pendientes_b:
        conteo_b[r.largo] = conteo_b.get(r.largo, 0) + 1

    indice_b = {r.largo: r for r in pendientes_b if conteo_b[r.largo] == 1}

    restantes_a = []
    usados_b = set()
    for r in pendientes_a:
        if conteo_a[r.largo] == 1 and r.largo in indice_b:
            pareja_b = indice_b[r.largo]
            if id(pareja_b) in usados_b:
                restantes_a.append(r)
                continue
            usados_b.add(id(pareja_b))
            parejas.append(Pareja(r, pareja_b, r.identificador, 'longitud'))
        else:
            restantes_a.append(r)

    solo_a = restantes_a
    solo_b = [r for r in pendientes_b if id(r) not in usados_b]

    parejas.sort(key=_orden_pareja)
    return parejas, solo_a, solo_b


def _orden_pareja(pareja):
    """Los cromosomas primero y en orden natural; el resto por posicion."""
    if pareja.criterio != 'cromosoma':
        return (2, pareja.a.inicio, '')
    clave = pareja.clave
    if clave.isdigit():
        return (0, int(clave), '')
    return (1, {'X': 0, 'Y': 1, 'MT': 2}.get(clave, 9), clave)


def comparar_parejas(cadena_a, cadena_b, parejas, motor, progreso=None,
                     detalle=None):
    """Compara pareja a pareja y devuelve una fila por secuencia.

    'motor' es una funcion motor(tramo_a, tramo_b) -> (resultado, segundos).
    Se recibe como parametro y no se importa aqui para que este modulo no
    dependa de los motores: asi la misma orquestacion sirve para CPU, GPU y
    NPU sin duplicar nada.

    Cada fila trae la clave, el criterio con que se emparejo, los offsets de
    cada archivo, el largo comparado, las diferencias y el tiempo. El total
    de diferencias es la suma de las filas.

    'detalle' es cuantas posiciones se guardan EN TOTAL, repartidas entre las
    secuencias en orden: con 701 parejas no se pueden guardar mil por cada
    una, pero con una sola pareja tampoco tiene sentido quedarse con veinte.
    """
    filas = []
    total_dif = 0
    total_bases = 0
    total_tiempo = 0.0
    invalidas = 0
    quedan = 500 if detalle is None else max(0, int(detalle))

    for indice, pareja in enumerate(parejas):
        largo = pareja.largo
        if largo <= 0:
            continue
        tramo_a = cadena_a.tramo(pareja.a.inicio, largo)
        tramo_b = cadena_b.tramo(pareja.b.inicio, largo)

        resultado, tiempo = motor(tramo_a, tramo_b)
        valido = resultado.get('valido', True)
        if not valido:
            invalidas += 1

        guardar = resultado.get('detalle', [])[:quedan]
        quedan -= len(guardar)

        filas.append({
            'clave': pareja.clave,
            'nombre': nombre_corto(pareja.a),
            'criterio': pareja.criterio,
            'inicio_a': pareja.a.inicio,
            'inicio_b': pareja.b.inicio,
            'largo': largo,
            'desfase': pareja.desfase,
            'diferencias': resultado['diferencias'],
            'similitud_pct': resultado['similitud_pct'],
            'valido': valido,
            'tiempo_s': tiempo,
            'detalle': guardar,
        })
        if valido:
            total_dif += resultado['diferencias']
        total_bases += largo
        total_tiempo += tiempo

        if progreso is not None:
            progreso(indice + 1, len(parejas))

    iguales = total_bases - total_dif
    return {
        'filas': filas,
        'parejas': len(filas),
        'bases': total_bases,
        'diferencias': total_dif,
        'iguales': iguales,
        'similitud_pct': (100.0 * iguales / total_bases) if total_bases
        else 0.0,
        'tiempo_s': total_tiempo,
        'invalidas': invalidas,
    }


def tabla_por_secuencia(resumen, maximo=30):
    """Texto tabulado con el resultado de comparar pareja a pareja."""
    lineas = [
        '%-12s %-10s %14s %14s %13s %12s %9s'
        % ('SECUENCIA', 'CRITERIO', 'INICIO A', 'INICIO B', 'LARGO',
           'DIFERENCIAS', 'TIEMPO'),
        '-' * 92,
    ]
    for fila in resumen['filas'][:maximo]:
        marca = '' if fila['valido'] else '  INVALIDO'
        lineas.append('%-12s %-10s %14d %14d %13d %12d %8.3fs%s'
                      % (fila['nombre'][:12], fila['criterio'],
                         fila['inicio_a'], fila['inicio_b'], fila['largo'],
                         fila['diferencias'], fila['tiempo_s'], marca))
    if len(resumen['filas']) > maximo:
        lineas.append('... y %d secuencias mas'
                      % (len(resumen['filas']) - maximo))
    lineas.append('-' * 92)
    lineas.append('TOTAL: %d secuencias, %d bases comparadas, %d diferencias '
                  '(%.6f %% iguales)'
                  % (resumen['parejas'], resumen['bases'],
                     resumen['diferencias'], resumen['similitud_pct']))
    return '\n'.join(lineas)


def resumen_emparejamiento(parejas, solo_a, solo_b):
    """Texto corto con el resultado del emparejamiento."""
    por_crom = sum(1 for p in parejas if p.criterio == 'cromosoma')
    por_largo = len(parejas) - por_crom
    bases = sum(p.largo for p in parejas)
    lineas = [
        'Emparejadas %d secuencias: %d por cromosoma y %d por longitud unica.'
        % (len(parejas), por_crom, por_largo),
        'Cubren %d bases comparables.' % bases,
    ]
    if solo_a or solo_b:
        lineas.append('Sin pareja: %d solo en A y %d solo en B. No se '
                      'comparan.' % (len(solo_a), len(solo_b)))
    return '\n'.join(lineas)


# ---------------------------------------------------------------------------
# Prueba rapida por linea de comandos
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print('Uso: python secuencias.py <a.fna> <b.fna>')
        sys.exit(0)

    def avance(hechos, total):
        sys.stdout.write('\r  indexando... %5.1f %%'
                         % (100.0 * hechos / total if total else 100.0))
        sys.stdout.flush()

    ra = indexar(sys.argv[1], progreso=avance)
    rb = indexar(sys.argv[2], progreso=avance)
    print('')
    print('A: %d registros   B: %d registros' % (len(ra), len(rb)))

    parejas, solo_a, solo_b = emparejar(ra, rb)
    print('')
    print(resumen_emparejamiento(parejas, solo_a, solo_b))
    print('')
    print('%-10s %-10s %14s %14s %12s' % ('clave', 'criterio', 'inicio A',
                                          'inicio B', 'largo'))
    for p in parejas[:30]:
        print('%-10s %-10s %14d %14d %12d'
              % (p.clave[:10], p.criterio, p.a.inicio, p.b.inicio, p.largo))
    if len(parejas) > 30:
        print('... y %d parejas mas' % (len(parejas) - 30))
