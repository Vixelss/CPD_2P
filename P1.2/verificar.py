# -*- coding: utf-8 -*-
"""
verificar.py

Prueba de correccion de los dos motores. Antes de medir rendimiento hay que
demostrar que los motores cuentan bien, porque un motor rapido que cuenta
mal no sirve de nada.

Compara sobre un mismo archivo:
    - CPU secuencial (un solo proceso)
    - CPU paralelo (varios procesos)
    - GPU (kernels CUDA)
y, si existe el archivo .esperado.txt correspondiente, contrasta ademas
contra los conteos correctos conocidos.

La comparacion se hace sobre el histograma completo de 256 bins, no solo
sobre A C G T. Es una prueba mucho mas exigente: obliga a que los tres
caminos coincidan hasta en el ultimo caracter raro del archivo.

NOTA SOBRE WINDOWS
Este archivo tiene que ejecutarse como script real, no por stdin. En Windows
multiprocessing usa el metodo 'spawn': cada proceso hijo vuelve a importar
el modulo principal, asi que necesita un archivo en disco y el guard
if __name__ == '__main__'. Sin eso los hijos se quedan colgados.

Uso:
    python verificar.py <archivo.fna> [n_procesos] [lote_mb]
"""

import multiprocessing
import os
import sys

import motor_cpu

# motor_gpu NO se importa aqui a proposito. En Windows cada proceso hijo que
# crea multiprocessing vuelve a importar este modulo, y si motor_gpu
# estuviera entre los imports de arriba los ocho hijos cargarian numba y todo
# el stack de CUDA para no usarlo. Medido en este proyecto: contar_paralelo
# sobre el archivo de 2 MB pasaba de 1.34 s a 5.51 s solo por ese import.
# Eso le cargaria a la CPU un coste que no es suyo y falsearia la comparativa
# contra la GPU. Se importa dentro de main(), que los hijos nunca ejecutan.


def leer_esperado(ruta):
    """Lee un archivo .esperado.txt y devuelve sus conteos, o None.

    Acepta tanto el formato de P1.1 (solo A C G T y TOTAL) como el que
    genera ensuciar.py (con N, BASES, INVALIDOS y el detalle).
    """
    ruta_esperado = ruta + '.esperado.txt'
    if not os.path.exists(ruta_esperado):
        return None

    datos = {'detalle': {}}
    with open(ruta_esperado, encoding='utf-8') as f:
        for linea in f:
            if linea.startswith('#'):
                continue
            partes = linea.rstrip('\n').split('\t')
            if partes[0] == 'INV' and len(partes) == 3:
                datos['detalle'][partes[1]] = int(partes[2])
            elif partes[0] == 'AMB' and len(partes) == 3:
                datos.setdefault('detalle_iupac', {})[partes[1]] = int(partes[2])
            elif len(partes) == 2 and partes[1].strip():
                datos[partes[0]] = int(partes[1])
    return datos


def comparar_con_esperado(resultado, esperado):
    """Devuelve la lista de discrepancias contra los conteos conocidos."""
    fallos = []

    for clave in ('A', 'C', 'G', 'T', 'N'):
        if clave in esperado and resultado[clave] != esperado[clave]:
            fallos.append('%s: obtenido %d, esperado %d'
                          % (clave, resultado[clave], esperado[clave]))

    if 'INVALIDOS' in esperado and resultado['invalidos'] != esperado['INVALIDOS']:
        fallos.append('invalidos: obtenido %d, esperado %d'
                      % (resultado['invalidos'], esperado['INVALIDOS']))

    if 'AMBIGUOS' in esperado and resultado['ambiguos'] != esperado['AMBIGUOS']:
        fallos.append('codigos IUPAC: obtenido %d, esperado %d'
                      % (resultado['ambiguos'], esperado['AMBIGUOS']))

    # El formato de P1.1 trae TOTAL como suma de A+C+G+T.
    if 'TOTAL' in esperado and resultado['bases'] != esperado['TOTAL']:
        fallos.append('total de bases: obtenido %d, esperado %d'
                      % (resultado['bases'], esperado['TOTAL']))

    if esperado.get('detalle') and resultado['detalle'] != esperado['detalle']:
        fallos.append('el detalle de invalidos no coincide')

    return fallos


def mostrar(nombre, resultado, tiempo, tamano):
    """Imprime una linea de resumen por motor."""
    mbs = (tamano / (1024 * 1024) / tiempo) if tiempo > 0 else 0.0
    print('  %-18s %8.3f s  %8.1f MB/s   A=%d C=%d G=%d T=%d N=%d '
          'iupac=%d inval=%d'
          % (nombre, tiempo, mbs, resultado['A'], resultado['C'],
             resultado['G'], resultado['T'], resultado['N'],
             resultado['ambiguos'], resultado['invalidos']))


def main():
    if len(sys.argv) < 2:
        print('Uso: python verificar.py <archivo.fna> [n_procesos] [lote_mb]')
        return 1

    ruta = sys.argv[1]
    if not os.path.exists(ruta):
        print('No existe el archivo: %s' % ruta)
        return 1

    # Import tardio: ver la nota de la cabecera del modulo.
    import motor_gpu

    fisicos, logicos = motor_cpu.detectar_nucleos()
    procesos = int(sys.argv[2]) if len(sys.argv) > 2 else logicos
    lote_mb = int(sys.argv[3]) if len(sys.argv) > 3 else motor_gpu.LOTE_MB

    tamano = os.path.getsize(ruta)
    print('Archivo : %s  (%.1f MB)' % (ruta, tamano / (1024 * 1024)))
    print('CPU     : %d nucleos fisicos / %d logicos' % (fisicos, logicos))

    datos_gpu = motor_gpu.info_gpu()
    hay_gpu = datos_gpu['disponible']
    if hay_gpu:
        print('GPU     : %s (CC %s, %d SMs, %d MB VRAM)'
              % (datos_gpu['nombre'], datos_gpu['compute_capability'],
                 datos_gpu['sms'], datos_gpu['vram_total_mb']))
        jit = motor_gpu.precalentar()
        print('          compilacion JIT: %.3f s (fuera de la medicion)' % jit)
    else:
        print('GPU     : no disponible, se omite la parte GPU')

    try:
        import motor_opencl as _ocl
        for _datos in _ocl.listar_dispositivos():
            print('OpenCL  : %s (%s, %d unidades)'
                  % (_datos['nombre'], _datos['tipo'], _datos['unidades']))
    except Exception:
        print('OpenCL  : no disponible')

    print('')
    print('Resultados:')

    resultados = {}

    res, tiempo = motor_cpu.contar_secuencial(ruta)
    resultados['CPU secuencial'] = res
    mostrar('CPU secuencial', res, tiempo, tamano)
    referencia = res

    res, tiempo = motor_cpu.contar_paralelo(ruta, procesos)
    resultados['CPU paralelo x%d' % procesos] = res
    mostrar('CPU paralelo x%d' % procesos, res, tiempo, tamano)

    if hay_gpu:
        res, tiempo = motor_gpu.contar_gpu(ruta, lote_mb=lote_mb)
        resultados['GPU CUDA'] = res
        mostrar('GPU CUDA %d MB' % lote_mb, res, tiempo, tamano)

    # Motor OpenCL sobre cada tarjeta del equipo. Es la prueba que de verdad
    # cierra el circulo: si la grafica integrada y la discreta devuelven el
    # mismo histograma que la CPU, el algoritmo es correcto con independencia
    # del hardware y de la tecnologia que lo ejecute.
    try:
        import motor_opencl
    except ImportError:
        motor_opencl = None

    if motor_opencl is not None and motor_opencl.opencl_disponible():
        for datos in motor_opencl.listar_dispositivos():
            motor_opencl.precalentar(datos['indice'])
            res, tiempo = motor_opencl.contar_opencl(
                ruta, dispositivo=datos['indice'], lote_mb=lote_mb)
            etiqueta = 'OpenCL %s' % datos['tipo']
            resultados[etiqueta] = res
            mostrar(etiqueta, res, tiempo, tamano)

    print('')
    print('Consistencia entre motores (254 bins, sin contar saltos de linea):')
    todo_bien = True
    patron = motor_cpu.bins_significativos(referencia['histograma'])
    for nombre, res in resultados.items():
        igual = motor_cpu.bins_significativos(res['histograma']) == patron
        print('  %-18s %s' % (nombre, 'identico' if igual else 'DIFIERE'))
        if not igual:
            todo_bien = False
            # Se detalla en que bytes esta la discrepancia, para poder
            # depurar sin tener que volver a lanzar todo el proceso.
            for i in range(motor_cpu.BINS):
                if i in motor_cpu.IGNORADOS:
                    continue
                mio = res['histograma'][i]
                suyo = referencia['histograma'][i]
                if mio != suyo:
                    print('      byte %3d (%-6s): %d frente a %d'
                          % (i, motor_cpu.nombre_byte(i), mio, suyo))

    # El bin del salto de linea se informa aparte porque se espera que
    # difiera entre CPU y GPU sin que eso sea un error. Ver la explicacion
    # en motor_cpu.bins_significativos.
    saltos = {n: r['histograma'][10] for n, r in resultados.items()}
    if len(set(saltos.values())) > 1:
        print('')
        print('  Nota: los saltos de linea difieren entre motores, y es lo')
        print('  esperado. La GPU sobreescribe las cabeceras con saltos de')
        print('  linea en vez de eliminarlas. No afecta a ningun conteo.')
        for nombre, valor in saltos.items():
            print('      %-18s %d' % (nombre, valor))

    esperado = leer_esperado(ruta)
    if esperado:
        print('')
        print('Contraste contra %s.esperado.txt:' % os.path.basename(ruta))
        for nombre, res in resultados.items():
            fallos = comparar_con_esperado(res, esperado)
            if fallos:
                todo_bien = False
                print('  %-18s FALLA' % nombre)
                for fallo in fallos:
                    print('      - %s' % fallo)
            else:
                print('  %-18s correcto' % nombre)
    else:
        print('')
        print('No hay archivo .esperado.txt para este archivo, se omite el')
        print('contraste contra conteos conocidos.')

    if referencia['detalle_iupac']:
        print('')
        print('Codigos IUPAC de ambiguedad (validos, NO son errores):')
        significados = {
            'R': 'A o G (purina)', 'Y': 'C o T (pirimidina)',
            'S': 'G o C (enlace fuerte)', 'W': 'A o T (enlace debil)',
            'K': 'G o T (ceto)', 'M': 'A o C (amino)',
            'B': 'C, G o T (no A)', 'D': 'A, G o T (no C)',
            'H': 'A, C o T (no G)', 'V': 'A, C o G (no T)',
        }
        for caracter, veces in sorted(referencia['detalle_iupac'].items(),
                                      key=lambda x: -x[1]):
            print('    %-3s %8d   %s'
                  % (caracter, veces, significados.get(caracter, '')))
        print('    %-3s %8d   total' % ('', referencia['ambiguos']))

    if referencia['detalle']:
        print('')
        print('Caracteres INVALIDOS encontrados (%d distintos, %d en total):'
              % (len(referencia['detalle']), referencia['invalidos']))
        for caracter, veces in sorted(referencia['detalle'].items(),
                                      key=lambda x: -x[1]):
            print('    %-6s %d' % (repr(caracter), veces))
    else:
        print('')
        print('No se encontro ningun caracter invalido en este archivo.')

    print('')
    print('RESULTADO GLOBAL: %s' % ('TODO CORRECTO' if todo_bien
                                    else 'HAY DISCREPANCIAS'))
    return 0 if todo_bien else 2


if __name__ == '__main__':
    multiprocessing.freeze_support()
    sys.exit(main())
