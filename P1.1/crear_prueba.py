# -*- coding: utf-8 -*-
"""
crear_prueba.py

Genera archivos FASTA pequenos para probar el motor y la interfaz sin
tener que esperar a los 3 GB del genoma completo.

Dos modos:

  sintetico   Crea un FASTA artificial con secuencia aleatoria. Reproduce
              todas las trampas del archivo real: varias cabeceras, bloques
              de N, soft-masking en minusculas y una ultima linea corta.
              Ademas escribe un archivo .esperado.txt con el conteo exacto
              de A, C, G y T, que sirve como respuesta correcta para
              validar el motor.

  recortar    Copia los primeros MB de un FASTA real (por ejemplo el
              GRCh38) respetando los limites de linea. Sirve para medir
              rendimiento con datos y proporciones reales.

Ejemplos:
    python crear_prueba.py sintetico --mb 50
    python crear_prueba.py sintetico --mb 2 --salida mini.fna
    python crear_prueba.py recortar --origen GCF_000001405.40_GRCh38.p14_genomic.fna --mb 200
"""

import argparse
import os
import random
import sys

# Ancho de linea del FASTA estandar.
_ANCHO = 80

# Bloque de escritura en disco.
_BLOQUE_ESCRITURA = 4 * 1024 * 1024

# Tabla que convierte bytes aleatorios 0-255 en las cuatro bases. Como 256
# es multiplo de 4, las cuatro bases salen equiprobables.
_TABLA_BASES = bytes.maketrans(
    bytes(range(256)),
    bytes(b'ACGT'[i % 4] for i in range(256)))

# Tabla de conteo: unifica minusculas para contar igual que el motor.
_TABLA_MAYUS = bytes.maketrans(b'acgt', b'ACGT')


def _formato_tamano(n):
    """Devuelve el tamano en una unidad legible."""
    for unidad in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or unidad == 'GB':
            return '%.2f %s' % (n, unidad)
        n /= 1024.0


def _contar(secuencia):
    """Cuenta A, C, G y T en un bloque de secuencia, sin cabeceras."""
    s = secuencia.translate(_TABLA_MAYUS)
    return {'A': s.count(65), 'C': s.count(67), 'G': s.count(71),
            'T': s.count(84)}


def _generar_cuerpo(largo, rnd):
    """Genera 'largo' bases aleatorias con soft-masking y bloques de N."""
    cuerpo = bytearray(os.urandom(largo).translate(_TABLA_BASES))

    # Bloques de N (bases desconocidas). El motor no debe contarlas.
    # Se usa un bloque grande al inicio, como los telomeros del genoma real.
    if largo > 20000:
        cuerpo[0:5000] = b'N' * 5000
        for _ in range(max(1, largo // 500000)):
            ini = rnd.randrange(0, largo - 1000)
            fin = min(largo, ini + rnd.randrange(100, 900))
            cuerpo[ini:fin] = b'N' * (fin - ini)

    # Soft-masking: regiones en minuscula que SI se cuentan.
    for _ in range(max(1, largo // 100000)):
        ini = rnd.randrange(0, max(1, largo - 5000))
        fin = min(largo, ini + rnd.randrange(500, 5000))
        cuerpo[ini:fin] = bytes(cuerpo[ini:fin]).lower()

    return bytes(cuerpo)


def _escribir_envuelto(salida, cuerpo):
    """Escribe la secuencia partida en lineas de 80 caracteres."""
    total = len(cuerpo)
    trozo = bytearray()

    for i in range(0, total, _ANCHO):
        trozo += cuerpo[i:i + _ANCHO]
        trozo += b'\n'
        if len(trozo) >= _BLOQUE_ESCRITURA:
            salida.write(trozo)
            trozo = bytearray()

    if trozo:
        salida.write(trozo)


def modo_sintetico(args):
    """Crea un FASTA artificial y su archivo de conteo esperado."""
    objetivo = int(args.mb * 1024 * 1024)
    rnd = random.Random(args.semilla)

    # Cabeceras con texto que contiene letras a, c, g y t a proposito: si el
    # motor no descarta las cabeceras, el conteo saldra inflado y la prueba
    # lo detecta.
    cabeceras = [
        b'>PRB_000001.1 Cadena de prueba sintetica, contig alfa, ensamblaje CPD',
        b'>PRB_000002.1 Cadena de prueba sintetica, contig gamma tacita GATTACA',
        b'>PRB_000003.1 Cadena de prueba sintetica, scaffold no localizado CTG7',
        b'>PRB_000004.1 Cadena de prueba sintetica, locus alterno cacatua',
        b'>PRB_000005.1 Cadena de prueba sintetica, mitocondria completa',
    ]

    # Las secuencias tienen tamanos desiguales para que el reparto entre
    # procesos no caiga siempre en el mismo sitio.
    pesos = [0.42, 0.27, 0.16, 0.10, 0.05]

    # El salto de linea cada 80 bases anade un byte extra por linea.
    bases_totales = int(objetivo * _ANCHO / (_ANCHO + 1.0))

    esperado = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
    ruta = args.salida or ('prueba_%dMB.fna' % args.mb)

    print('Generando %s (objetivo %s)...' % (ruta, _formato_tamano(objetivo)))

    with open(ruta, 'wb') as salida:
        for cabecera, peso in zip(cabeceras, pesos):
            largo = max(_ANCHO, int(bases_totales * peso))
            cuerpo = _generar_cuerpo(largo, rnd)

            parcial = _contar(cuerpo)
            for base in esperado:
                esperado[base] += parcial[base]

            salida.write(cabecera + b'\n')
            _escribir_envuelto(salida, cuerpo)

    real = os.path.getsize(ruta)
    total_bases = sum(esperado.values())

    ruta_esperado = ruta + '.esperado.txt'
    with open(ruta_esperado, 'w', encoding='utf-8') as f:
        f.write('# Conteo correcto del archivo %s\n' % ruta)
        f.write('# Generado por crear_prueba.py con semilla %d\n' % args.semilla)
        for base in ('A', 'C', 'G', 'T'):
            f.write('%s\t%d\n' % (base, esperado[base]))
        f.write('TOTAL\t%d\n' % total_bases)

    print('Archivo creado : %s (%s)' % (ruta, _formato_tamano(real)))
    print('Conteo esperado: %s' % esperado)
    print('Total de bases : %d' % total_bases)
    print('Guardado en    : %s' % ruta_esperado)
    return ruta


def modo_recortar(args):
    """Copia los primeros MB de un FASTA real respetando las lineas."""
    origen = args.origen
    if not os.path.isfile(origen):
        print('ERROR: no existe el archivo de origen %s' % origen)
        return None

    objetivo = int(args.mb * 1024 * 1024)
    ruta = args.salida or ('recorte_%dMB.fna' % args.mb)

    print('Recortando %s MB de %s...' % (args.mb, origen))

    escritos = 0
    with open(origen, 'rb') as entrada, open(ruta, 'wb') as salida:
        if args.desde > 0:
            entrada.seek(args.desde)
            entrada.readline()  # descarta la linea partida
            # El recorte debe empezar por una cabecera para seguir siendo
            # un FASTA valido.
            cabecera = b'>PRB_RECORTE recorte de %s desde el byte %d\n' % (
                os.path.basename(origen).encode('ascii', 'replace'), args.desde)
            salida.write(cabecera)
            escritos += len(cabecera)

        while escritos < objetivo:
            bloque = entrada.read(min(_BLOQUE_ESCRITURA, objetivo - escritos))
            if not bloque:
                break
            # Se completa la ultima linea para no cortar a la mitad.
            if not bloque.endswith(b'\n'):
                resto = entrada.readline()
                if resto:
                    bloque += resto
                elif not bloque.endswith(b'\n'):
                    bloque += b'\n'
            salida.write(bloque)
            escritos += len(bloque)

    real = os.path.getsize(ruta)
    print('Archivo creado : %s (%s)' % (ruta, _formato_tamano(real)))
    print('Sin conteo esperado: es un recorte de datos reales. Su validacion')
    print('es que todas las configuraciones del benchmark coincidan.')
    return ruta


def main():
    parser = argparse.ArgumentParser(
        description='Genera archivos FASTA de prueba para el proyecto P1.1.')
    sub = parser.add_subparsers(dest='modo', required=True)

    p1 = sub.add_parser('sintetico',
                        help='FASTA artificial con conteo exacto conocido')
    p1.add_argument('--mb', type=float, default=50,
                    help='tamano aproximado en MB (por defecto 50)')
    p1.add_argument('--salida', default=None, help='nombre del archivo')
    p1.add_argument('--semilla', type=int, default=1234,
                    help='semilla para que el archivo sea reproducible')
    p1.set_defaults(func=modo_sintetico)

    p2 = sub.add_parser('recortar',
                        help='recorte de un FASTA real ya existente')
    p2.add_argument('--origen', required=True, help='FASTA de origen')
    p2.add_argument('--mb', type=float, default=100,
                    help='tamano aproximado en MB (por defecto 100)')
    p2.add_argument('--salida', default=None, help='nombre del archivo')
    p2.add_argument('--desde', type=int, default=0,
                    help='byte del origen desde el que empieza el recorte')
    p2.set_defaults(func=modo_recortar)

    args = parser.parse_args()
    ruta = args.func(args)

    if ruta:
        print('')
        print('Para probarlo:')
        print('    python benchmark.py %s' % ruta)
    return 0


if __name__ == '__main__':
    sys.exit(main())
