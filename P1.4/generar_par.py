# -*- coding: utf-8 -*-
r"""
generar_par.py

Genera un par de cadenas de ADN para comparar, con un numero de diferencias
conocido de antemano.

POR QUE HACE FALTA
El enunciado pide comparar dos cadenas y senalar los caracteres distintos.
Para demostrar que el programa acierta hace falta un par cuyas diferencias se
conozcan exactamente: si el motor dice "encontre 4.812 diferencias" hay que
poder contrastarlo contra la verdad.

Las dos cadenas se derivan SIEMPRE de un FASTA real (un recorte del genoma
humano), nunca de secuencia inventada. Es la misma decision que se tomo en el
P1.2 con ensuciar.py: una prueba sobre datos sinteticos no demuestra que el
programa funcione sobre datos de verdad.

COMO SE MUTA
La cadena B es una copia de la A con un numero pedido de posiciones
cambiadas. En cada posicion elegida se sustituye la base por OTRA distinta,
nunca por la misma, de modo que toda mutacion cuenta como diferencia. Las
posiciones se sortean sin repeticion, asi que el numero de diferencias es
exacto y no aproximado.

Junto a los dos archivos se escribe un .esperado.txt con la lista completa de
posiciones, filas y columnas mutadas. Ese archivo es la respuesta correcta
contra la que verificar los tres motores.

Uso:
    python generar_par.py --origen ..\P1.3\datos\recorte_50MB.fna --mb 10
    python generar_par.py --origen X.fna --mb 50 --diferencias 5000
"""

import argparse
import os
import sys

import numpy as np

import comparador


# Bases entre las que se muta. Se excluyen la N y los codigos IUPAC a
# proposito: mutar una N por una A seria una diferencia legitima, pero
# enturbia la lectura del informe, porque una N no es una base concreta sino
# una posicion sin resolver.
BASES = np.frombuffer(b'ACGT', dtype=np.uint8)

# Diferencias por defecto, en partes por millon de la cadena.
TASA_POR_DEFECTO = 100


def escribir_fasta(ruta, secuencia, cabecera, ancho=comparador.ANCHO_LINEA):
    """Escribe una secuencia como FASTA con lineas del ancho estandar."""
    with open(ruta, 'wb') as f:
        f.write(cabecera.encode('ascii', 'replace') + b'\n')
        total = secuencia.size
        bloque = bytearray()
        for i in range(0, total, ancho):
            bloque += secuencia[i:i + ancho].tobytes()
            bloque += b'\n'
            if len(bloque) >= 4 * 1024 * 1024:
                f.write(bloque)
                bloque = bytearray()
        if bloque:
            f.write(bloque)


def generar(origen, salida='datos', mb=10, diferencias=None,
            tasa=TASA_POR_DEFECTO, semilla=1234, progreso=None,
            cache=None):
    """Crea el par de cadenas y el archivo de diferencias esperadas.

    Devuelve un diccionario con las rutas y las posiciones mutadas.
    """
    if not os.path.isdir(salida):
        os.makedirs(salida)

    azar = np.random.default_rng(semilla)

    # Se prepara el origen y se recorta al tamano pedido. El recorte se hace
    # sobre la secuencia ya limpia, no sobre el archivo, para que las dos
    # cadenas midan exactamente lo mismo.
    # La cache del .seq va aparte de la carpeta de salida. Si fueran la
    # misma, generar un recorte de 20 MB a partir del genoma obligaria a
    # extraer otra vez sus 3.3 GB de secuencia dentro de la carpeta de
    # pruebas, en vez de reutilizar el .seq que ya existe en datos\.
    cadena = comparador.preparar(origen, cache=(cache or 'datos'),
                                 progreso=progreso)
    objetivo = int(mb * 1024 * 1024)
    if cadena.largo > objetivo:
        seq_a = np.array(cadena.datos[:objetivo])
    else:
        seq_a = np.array(cadena.datos)
    cadena.cerrar()

    largo = seq_a.size
    if largo == 0:
        raise ValueError('El archivo de origen no contiene secuencia.')

    if diferencias is None:
        diferencias = max(1, int(largo * tasa / 1e6))
    diferencias = min(int(diferencias), largo)

    # Posiciones sin repeticion: asi el numero de diferencias es exacto.
    posiciones = np.sort(azar.choice(largo, size=diferencias, replace=False))

    seq_b = seq_a.copy()
    originales = seq_a[posiciones]

    # Cada posicion recibe una base DISTINTA de la que tenia. Se sortea un
    # desplazamiento de 1 a 3 sobre el alfabeto ACGT, lo que garantiza que
    # nunca cae la misma base. Para posiciones que no eran ACGT (una N, por
    # ejemplo) se asigna una base cualquiera, que tambien sera distinta.
    indice = np.searchsorted(BASES, originales)
    es_base = (indice < BASES.size) & (BASES[np.minimum(indice, 3)] == originales)
    desplazamiento = azar.integers(1, 4, size=diferencias)
    nuevo_indice = np.where(es_base,
                            (indice + desplazamiento) % BASES.size,
                            azar.integers(0, BASES.size, size=diferencias))
    seq_b[posiciones] = BASES[nuevo_indice]

    nombre = 'par_%dMB' % mb
    ruta_a = os.path.join(salida, nombre + '_A.fna')
    ruta_b = os.path.join(salida, nombre + '_B.fna')

    escribir_fasta(ruta_a, seq_a,
                   '>P14_A cadena de referencia, recorte de %s'
                   % os.path.basename(origen))
    escribir_fasta(ruta_b, seq_b,
                   '>P14_B cadena mutada, %d diferencias sobre P14_A'
                   % diferencias)

    ruta_esperado = ruta_b + '.esperado.txt'
    filas, columnas = comparador.filas_columnas(posiciones)
    with open(ruta_esperado, 'w', encoding='utf-8') as f:
        f.write('# Diferencias correctas entre %s y %s\n'
                % (os.path.basename(ruta_a), os.path.basename(ruta_b)))
        f.write('# Generado por generar_par.py con semilla %d\n' % semilla)
        f.write('LARGO\t%d\n' % largo)
        f.write('DIFERENCIAS\t%d\n' % diferencias)
        f.write('#\n')
        f.write('# posicion\tfila\tcolumna\tA\tB\n')
        for p, fi, co in zip(posiciones, filas, columnas):
            f.write('DIF\t%d\t%d\t%d\t%s\t%s\n'
                    % (p, fi, co, chr(int(seq_a[p])), chr(int(seq_b[p]))))

    return {
        'ruta_a': ruta_a,
        'ruta_b': ruta_b,
        'ruta_esperado': ruta_esperado,
        'largo': largo,
        'diferencias': diferencias,
        'posiciones': posiciones,
    }


def leer_esperado(ruta_esperado):
    """Lee un .esperado.txt y devuelve (largo, posiciones esperadas)."""
    largo = 0
    posiciones = []
    with open(ruta_esperado, encoding='utf-8') as f:
        for linea in f:
            if linea.startswith('#'):
                continue
            partes = linea.rstrip('\n').split('\t')
            if partes[0] == 'LARGO':
                largo = int(partes[1])
            elif partes[0] == 'DIF':
                posiciones.append(int(partes[1]))
    return largo, np.array(posiciones, dtype=np.int64)


def main():
    p = argparse.ArgumentParser(
        description='Genera un par de cadenas de ADN con diferencias '
                    'conocidas.')
    p.add_argument('--origen', required=True,
                   help='FASTA real del que se derivan las dos cadenas')
    p.add_argument('--salida', default='datos', help='carpeta de destino')
    p.add_argument('--mb', type=float, default=10,
                   help='tamano de cada cadena en MB de secuencia')
    p.add_argument('--diferencias', type=int, default=None,
                   help='numero exacto de diferencias a inyectar')
    p.add_argument('--tasa', type=float, default=TASA_POR_DEFECTO,
                   help='diferencias por millon, si no se da --diferencias')
    p.add_argument('--semilla', type=int, default=1234)
    args = p.parse_args()

    if not os.path.exists(args.origen):
        print('No existe el archivo de origen: %s' % args.origen)
        return 1

    def progreso(hechos, total):
        sys.stdout.write('\r  leyendo origen... %5.1f %%'
                         % (100.0 * hechos / total if total else 100.0))
        sys.stdout.flush()

    print('Origen : %s' % args.origen)
    datos = generar(args.origen, salida=args.salida, mb=args.mb,
                    diferencias=args.diferencias, tasa=args.tasa,
                    semilla=args.semilla, progreso=progreso)
    print('')
    print('')
    print('Cadena A    : %s' % datos['ruta_a'])
    print('Cadena B    : %s' % datos['ruta_b'])
    print('Largo       : %d bases (%.1f MB de secuencia)'
          % (datos['largo'], datos['largo'] / (1024 * 1024)))
    print('Diferencias : %d (exactas, sin repeticion)' % datos['diferencias'])
    print('Esperado    : %s' % datos['ruta_esperado'])
    print('')
    print('Para probarlo:')
    print('    python motor_cpu.py %s %s' % (datos['ruta_a'], datos['ruta_b']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
