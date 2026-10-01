# -*- coding: utf-8 -*-
"""
ensuciar.py

Generador de archivos FASTA con caracteres invalidos inyectados, a partir
del genoma real entregado por el profesor.

POR QUE HACE FALTA ESTE PROGRAMA
El enunciado del P1.2 pide un archivo "con caracteres validos y no validos"
y la rubrica reserva 10 puntos a la correcta identificacion de los
invalidos. Sin embargo, el archivo real GCF_000001405.40_GRCh38.p14 no tiene
ningun caracter invalido: se verifico con un muestreo de 320 MB repartido
por todo el archivo y su alfabeto de secuencia es unicamente A C G T en
mayuscula y minuscula, la N de base desconocida y el salto de linea. Los
demas caracteres que aparecen en el archivo pertenecen a las lineas de
cabecera, que no son secuencia.

Contra ese archivo el detector siempre daria cero y no se podria demostrar
que funciona. Por eso se derivan del genoma real copias "sucias" con una
cantidad y un tipo de error conocidos de antemano, que sirven de prueba
verificable: el motor acierta si reproduce exactamente el .esperado.txt.

Los errores se inyectan SUSTITUYENDO caracteres de las lineas de secuencia,
nunca insertando. Asi el archivo resultante conserva el tamano y la
estructura de lineas del original, y la comparacion de rendimiento contra el
archivo limpio sigue siendo valida.

Las cabeceras no se tocan jamas: un error dentro de una cabecera no seria
detectable, porque las cabeceras se descartan antes de contar.

Uso:
    python ensuciar.py <origen.fna> <destino.fna> [opciones]

Opciones:
    --mb N          tomar solo los primeros N MB del origen
    --tasa N        errores por cada millon de caracteres de secuencia
    --tipo T        letras | digitos | simbolos | espacios | mixto
    --semilla N     semilla del generador aleatorio (reproducibilidad)
"""

import argparse
import os
import sys

import numpy as np

from motor_cpu import BINS, nombre_byte, resumir_histograma


# Catalogo de errores por tipo. Ninguno de estos caracteres es una base
# valida ni un codigo IUPAC de ambiguedad, de modo que todos deben acabar
# contados como invalidos.
#   - letras   : letras que no existen en el alfabeto del ADN.
#   - digitos  : contaminacion tipica al mezclar datos con numeros de linea.
#   - simbolos : basura de codificacion o de copiado entre formatos.
#   - espacios : el caso mas traicionero, porque no se ve al abrir el
#                archivo pero rompe igual cualquier parser.
CATALOGO = {
    # Ojo con las letras elegidas: no puede aparecer ninguna base (ACGTN),
    # ningun codigo IUPAC de ambiguedad (RYSWKMBDHV) ni la U del uracilo,
    # que es una base valida en ARN. Cualquiera de esas seria clasificada
    # como valida por el motor y la prueba no demostraria nada.
    'letras':   b'XZJOQE',
    'digitos':  b'0123456789',
    'simbolos': b'@#$%*?',
    'espacios': b' \t',
}

# Tipos que se mezclan cuando se pide el modo 'mixto'.
TIPOS = ('letras', 'digitos', 'simbolos', 'espacios')

# Tasa por defecto: errores por cada millon de caracteres de secuencia.
TASA_POR_DEFECTO = 100

# Tamano de bloque de escritura.
_BLOQUE = 8 * 1024 * 1024

# Bytes con significado propio dentro del formato.
COD_MAYOR = 62          # '>' abre una linea de cabecera
COD_SALTO = 10          # '\n'
COD_RETORNO = 13        # '\r' de los finales de linea de Windows


def construir_alfabeto(tipo):
    """Devuelve la lista de bytes de error correspondiente al tipo pedido."""
    if tipo == 'mixto':
        alfabeto = b''.join(CATALOGO[t] for t in TIPOS)
    elif tipo in CATALOGO:
        alfabeto = CATALOGO[tipo]
    else:
        raise ValueError('Tipo de error desconocido: %s' % tipo)
    return list(alfabeto)


def _marcar_cabeceras(arr):
    """Devuelve una mascara booleana con los bytes que son de cabecera.

    El bloque siempre empieza en principio de linea, de modo que una cabecera
    es un '>' en la posicion cero o precedido de un salto de linea. Se marca
    desde ahi hasta el final de su linea. Las cabeceras son poquisimas frente
    al tamano del bloque, asi que recorrerlas una a una en Python no cuesta
    nada.
    """
    es_cabecera = np.zeros(arr.size, dtype=bool)
    candidatas = np.flatnonzero(arr == COD_MAYOR)
    if candidatas.size == 0:
        return es_cabecera

    # Un '>' solo abre cabecera si esta al principio del bloque o justo
    # despues de un salto de linea; en medio de una linea es un caracter mas.
    inicios = candidatas[(candidatas == 0)
                         | (arr[np.maximum(candidatas - 1, 0)] == COD_SALTO)]
    if inicios.size == 0:
        return es_cabecera

    # Los saltos de linea se localizan una sola vez y el final de cada
    # cabecera se busca con una busqueda binaria sobre ellos. Hacerlo
    # recorriendo el resto del bloque por cada cabecera costaria el tamano
    # del bloque multiplicado por el numero de cabeceras, y en la zona de
    # scaffolds del genoma hay muchas cabeceras muy juntas.
    saltos = np.flatnonzero(arr == COD_SALTO)
    posiciones = np.searchsorted(saltos, inicios)
    for inicio, posicion in zip(inicios, posiciones):
        fin = saltos[posicion] if posicion < saltos.size else arr.size
        es_cabecera[inicio:fin] = True
    return es_cabecera


def ensuciar(origen, destino, mb=None, tasa=TASA_POR_DEFECTO,
             tipo='mixto', semilla=1234, progreso=None):
    """Crea una copia del FASTA con caracteres invalidos inyectados.

    Las cabeceras se copian intactas. En las lineas de secuencia, cada
    caracter tiene una probabilidad tasa/1e6 de ser sustituido por un
    caracter invalido del alfabeto elegido.

    Mientras escribe va acumulando el histograma real del resultado, de modo
    que los conteos esperados no son una estimacion sino la cuenta exacta de
    lo que se acaba de escribir.

    POR QUE ESTA VECTORIZADO
    La primera version recorria el archivo byte a byte en Python, sorteando
    un numero aleatorio por cada caracter. Sobre 50 MB ya tardaba casi un
    minuto, y sobre el genoma completo habria sido inviable, lo que dejaba
    sin poder probar la deteccion de errores al tamano real. Ahora se trabaja
    por bloques con numpy: en lugar de un sorteo por caracter se saca de una
    binomial cuantos errores caen en el bloque y se eligen sus posiciones de
    golpe, que es estadisticamente equivalente y varios ordenes de magnitud
    mas rapido.

    Devuelve el diccionario de resultados esperados.
    """
    alfabeto = np.array(construir_alfabeto(tipo), dtype=np.uint8)
    azar = np.random.default_rng(semilla)
    probabilidad = tasa / 1e6

    limite = None if mb is None else int(mb) * 1024 * 1024
    tamano_origen = os.path.getsize(origen)
    total_previsto = tamano_origen if limite is None else min(limite,
                                                              tamano_origen)

    histograma = np.zeros(BINS, dtype=np.int64)
    inyectados = 0
    escritos = 0

    with open(origen, 'rb') as entrada, open(destino, 'wb') as salida:
        while True:
            if limite is not None and escritos >= limite:
                break

            por_leer = _BLOQUE
            if limite is not None:
                por_leer = min(por_leer, limite - escritos)

            bloque = entrada.read(por_leer)
            if not bloque:
                break

            # El bloque debe terminar en linea completa para que ninguna
            # cabecera quede partida entre dos bloques.
            if not bloque.endswith(b'\n'):
                resto = entrada.readline()
                if resto:
                    bloque += resto

            arr = np.frombuffer(bloque, dtype=np.uint8).copy()
            es_cabecera = _marcar_cabeceras(arr)

            # Candidatos a ensuciarse: todo lo que sea secuencia de verdad.
            # Ni cabeceras, ni saltos de linea, ni retornos de carro.
            ensuciables = (~es_cabecera & (arr != COD_SALTO)
                           & (arr != COD_RETORNO))
            candidatos = np.flatnonzero(ensuciables)

            if probabilidad > 0 and candidatos.size:
                cuantos = azar.binomial(candidatos.size, probabilidad)
                if cuantos:
                    # Se sortean posiciones y se eliminan las repetidas. El
                    # numero final puede quedar un pelo por debajo del
                    # sorteado, pero da igual: lo que se publica como
                    # esperado es el histograma de lo que se escribio de
                    # verdad, no la cifra que se pretendia inyectar.
                    sorteo = azar.integers(0, candidatos.size, size=cuantos)
                    elegidas = candidatos[np.unique(sorteo)]
                    arr[elegidas] = azar.choice(alfabeto, size=elegidas.size)
                    inyectados += int(elegidas.size)

            # El histograma esperado excluye las cabeceras, igual que hacen
            # los motores de conteo.
            histograma += np.bincount(arr[~es_cabecera], minlength=BINS)

            salida.write(arr.tobytes())
            escritos += arr.size

            if progreso is not None:
                progreso(escritos, total_previsto)

    if progreso is not None:
        progreso(total_previsto, total_previsto)

    esperado = resumir_histograma(histograma)
    esperado['inyectados'] = inyectados
    esperado['bytes'] = escritos
    return esperado


def escribir_esperado(ruta_destino, esperado, tipo, tasa, semilla):
    """Guarda el archivo .esperado.txt con los conteos correctos.

    Este archivo es el que convierte la prueba en verificable: cualquiera
    puede correr el motor y comprobar que reproduce estos numeros.
    """
    ruta = ruta_destino + '.esperado.txt'
    with open(ruta, 'w', encoding='utf-8') as f:
        f.write('# Conteo correcto de %s\n' % os.path.basename(ruta_destino))
        f.write('# Generado por ensuciar.py  tipo=%s  tasa=%s/millon  '
                'semilla=%s\n' % (tipo, tasa, semilla))
        f.write('#\n')
        f.write('A\t%d\n' % esperado['A'])
        f.write('C\t%d\n' % esperado['C'])
        f.write('G\t%d\n' % esperado['G'])
        f.write('T\t%d\n' % esperado['T'])
        f.write('N\t%d\n' % esperado['N'])
        f.write('BASES\t%d\n' % esperado['bases'])
        f.write('AMBIGUOS\t%d\n' % esperado['ambiguos'])
        f.write('INVALIDOS\t%d\n' % esperado['invalidos'])
        f.write('#\n')
        f.write('# Codigos IUPAC de ambiguedad presentes (validos, no error)\n')
        for caracter, veces in sorted(esperado['detalle_iupac'].items(),
                                      key=lambda x: -x[1]):
            f.write('AMB\t%s\t%d\n' % (caracter, veces))
        f.write('#\n')
        f.write('# Detalle de caracteres invalidos (caracter, veces)\n')
        for caracter, veces in sorted(esperado['detalle'].items(),
                                      key=lambda x: -x[1]):
            f.write('INV\t%s\t%d\n' % (caracter, veces))
    return ruta


def main():
    analizador = argparse.ArgumentParser(
        description='Genera un FASTA con caracteres invalidos inyectados.')
    analizador.add_argument('origen', help='archivo FASTA de origen')
    analizador.add_argument('destino', help='archivo FASTA a generar')
    analizador.add_argument('--mb', type=int, default=None,
                            help='tomar solo los primeros N MB del origen')
    analizador.add_argument('--tasa', type=float, default=TASA_POR_DEFECTO,
                            help='errores por millon de caracteres')
    analizador.add_argument('--tipo', default='mixto',
                            choices=list(TIPOS) + ['mixto'],
                            help='tipo de error a inyectar')
    analizador.add_argument('--semilla', type=int, default=1234,
                            help='semilla del generador aleatorio')
    args = analizador.parse_args()

    if not os.path.exists(args.origen):
        print('No existe el archivo de origen: %s' % args.origen)
        return 1

    def progreso(hechos, total):
        porcentaje = 100.0 * hechos / total if total else 100.0
        sys.stdout.write('\r  generando... %5.1f%%  (%d MB)'
                         % (porcentaje, hechos // (1024 * 1024)))
        sys.stdout.flush()

    print('Origen : %s' % args.origen)
    print('Destino: %s' % args.destino)
    print('Tipo de error: %s   Tasa: %s por millon   Semilla: %d'
          % (args.tipo, args.tasa, args.semilla))
    print('')

    esperado = ensuciar(args.origen, args.destino, mb=args.mb,
                        tasa=args.tasa, tipo=args.tipo,
                        semilla=args.semilla, progreso=progreso)
    print('')
    print('')
    print('Generado: %.1f MB' % (esperado['bytes'] / (1024 * 1024)))
    print('  A=%d C=%d G=%d T=%d N=%d'
          % (esperado['A'], esperado['C'], esperado['G'],
             esperado['T'], esperado['N']))
    print('  Caracteres invalidos inyectados: %d' % esperado['invalidos'])
    print('  Detalle: %s' % esperado['detalle'])

    ruta = escribir_esperado(args.destino, esperado, args.tipo,
                             args.tasa, args.semilla)
    print('')
    print('Conteos correctos escritos en: %s' % ruta)
    return 0


if __name__ == '__main__':
    sys.exit(main())
