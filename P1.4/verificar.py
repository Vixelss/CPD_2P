# -*- coding: utf-8 -*-
"""
verificar.py

Prueba de correccion y de robustez del P1.4. Tiene dos partes y las dos son
necesarias:

  PARTE 1 - CORRECCION
  Demuestra que las tres plataformas cuentan lo mismo. Un motor rapido que
  compara mal no sirve de nada. Se contrasta contra un par de cadenas con un
  numero de diferencias conocido de antemano (ver generar_par.py), de modo
  que la prueba no es "los tres coinciden entre si" sino "los tres coinciden
  con la verdad".

  PARTE 2 - ROBUSTEZ, O PRUEBAS A PRUEBA DE USUARIO
  Comprueba que el programa no se rompe ni miente cuando recibe una entrada
  equivocada. Estan cubiertas las categorias habituales de este tipo de
  prueba:

    valores limite        0, 1, el maximo, el maximo mas uno
    tipos invalidos       letras donde va un numero
    valores negativos     -1, -100
    campo vacio           cadena vacia, solo espacios
    desbordamiento        numeros enormes
    archivo inexistente   ruta que no esta
    archivo equivocado    binario, carpeta, archivo vacio
    archivo no FASTA      texto que no es secuencia
    mismo archivo dos veces
    cadenas de distinto largo

La regla de todas ellas es la misma: el programa o corrige el valor y lo
explica, o rechaza la entrada con un mensaje legible. Lo que nunca puede
hacer es caerse con una traza ni devolver un resultado que parezca valido.

NOTA SOBRE WINDOWS
Este archivo tiene que ejecutarse como script real, no por la entrada
estandar. En Windows multiprocessing usa 'spawn': cada proceso hijo vuelve a
importar el modulo principal, asi que necesita un archivo en disco y el guard
if __name__ == '__main__'.

Uso:
    python verificar.py                 (usa el par de datos\\ o lo genera)
    python verificar.py <a.fna> <b.fna>
"""

import multiprocessing
import os
import sys

import numpy as np

import comparador
import motor_cpu
from comparador import ErrorEntrada


CARPETA_DATOS = 'datos'
CARPETA_TEMP = os.path.join(CARPETA_DATOS, '_pruebas')

_aciertos = 0
_fallos = 0


def comprobar(nombre, condicion, detalle=''):
    """Registra el resultado de una comprobacion y lo imprime."""
    global _aciertos, _fallos
    if condicion:
        _aciertos += 1
        print('  [ok]    %s' % nombre)
    else:
        _fallos += 1
        print('  [FALLA] %s   %s' % (nombre, detalle))
    return bool(condicion)


def espera_error(nombre, funcion, *args, **kwargs):
    """Comprueba que una llamada rechaza la entrada con ErrorEntrada.

    Lo que se exige no es solo que falle, sino que falle BIEN: con la
    excepcion propia del programa y con un mensaje con texto dentro. Una
    excepcion cualquiera significaria que el fallo no estaba previsto.
    """
    try:
        funcion(*args, **kwargs)
    except ErrorEntrada as error:
        return comprobar(nombre, bool(str(error).strip()),
                         'el mensaje esta vacio')
    except Exception as error:
        return comprobar(nombre, False,
                         'lanzo %s en vez de ErrorEntrada: %s'
                         % (type(error).__name__, error))
    return comprobar(nombre, False, 'no rechazo la entrada')


# ---------------------------------------------------------------------------
# Datos de prueba
# ---------------------------------------------------------------------------

def preparar_datos():
    """Devuelve (ruta_a, ruta_b, diferencias_esperadas), generando si falta."""
    import generar_par

    ruta_a = os.path.join(CARPETA_DATOS, 'par_10MB_A.fna')
    ruta_b = os.path.join(CARPETA_DATOS, 'par_10MB_B.fna')
    esperado = ruta_b + '.esperado.txt'

    if not (os.path.exists(ruta_a) and os.path.exists(ruta_b)
            and os.path.exists(esperado)):
        origen = None
        for candidato in (os.path.join('..', 'P1.3', 'datos',
                                       'recorte_50MB.fna'),
                          os.path.join('..', 'P1.1', 'prueba_50MB.fna'),
                          os.path.join('..', 'P1.1', 'mini.fna')):
            if os.path.exists(candidato):
                origen = candidato
                break
        if origen is None:
            print('No hay datos de prueba ni archivo de origen para '
                  'generarlos.')
            return None, None, 0
        print('Generando el par de prueba a partir de %s...' % origen)
        generar_par.generar(origen, salida=CARPETA_DATOS, mb=10,
                            diferencias=5000)

    _, posiciones = generar_par.leer_esperado(esperado)
    return ruta_a, ruta_b, posiciones


def archivos_trampa():
    """Crea los archivos con los que se prueban las entradas equivocadas."""
    if not os.path.isdir(CARPETA_TEMP):
        os.makedirs(CARPETA_TEMP)

    rutas = {}

    rutas['vacio'] = os.path.join(CARPETA_TEMP, 'vacio.fna')
    open(rutas['vacio'], 'wb').close()

    rutas['binario'] = os.path.join(CARPETA_TEMP, 'binario.fna')
    with open(rutas['binario'], 'wb') as f:
        f.write(bytes(range(256)) * 20)

    rutas['no_fasta'] = os.path.join(CARPETA_TEMP, 'no_fasta.fna')
    with open(rutas['no_fasta'], 'w', encoding='utf-8') as f:
        f.write('Esto es un documento de texto cualquiera.\n'
                'No tiene ninguna cabecera ni secuencia de ADN dentro.\n')

    rutas['solo_cabecera'] = os.path.join(CARPETA_TEMP, 'solo_cabecera.fna')
    with open(rutas['solo_cabecera'], 'w', encoding='utf-8') as f:
        f.write('>PRB solo cabeceras, sin una sola base\n')
        f.write('>PRB otra cabecera mas\n')

    rutas['corto'] = os.path.join(CARPETA_TEMP, 'corto.fna')
    with open(rutas['corto'], 'w', encoding='utf-8') as f:
        f.write('>PRB cadena corta\nACGTACGTAC\n')

    rutas['corto_distinto'] = os.path.join(CARPETA_TEMP,
                                           'corto_distinto.fna')
    with open(rutas['corto_distinto'], 'w', encoding='utf-8') as f:
        f.write('>PRB cadena corta con dos cambios y mas larga\n')
        f.write('ACGTTCGTAGTTTT\n')

    rutas['carpeta'] = CARPETA_TEMP
    rutas['inexistente'] = os.path.join(CARPETA_TEMP, 'no_existe_jamas.fna')

    return rutas


# ---------------------------------------------------------------------------
# Parte 1: correccion
# ---------------------------------------------------------------------------

def probar_correccion(ruta_a, ruta_b, esperadas):
    print('')
    print('PARTE 1 - CORRECCION')
    print('-' * 62)

    ca = comparador.preparar(ruta_a)
    cb = comparador.preparar(ruta_b)
    total_esperado = int(esperadas.size)
    lista_esperada = list(esperadas)
    print('  Par de prueba: %d bases, %d diferencias conocidas'
          % (ca.largo, total_esperado))

    resultados = {}

    res, _ = motor_cpu.comparar(ca, cb, detalle=total_esperado)
    resultados['CPU x1'] = res

    for n in (2, 4, 8, 12):
        ajustado, _ = motor_cpu.procesos_validos(n, min(ca.largo, cb.largo))
        res, _ = motor_cpu.comparar_paralelo(ca, cb, procesos=n,
                                             detalle=total_esperado)
        resultados['CPU x%d (pedidos %d)' % (ajustado, n)] = res

    try:
        import motor_gpu
        if motor_gpu.gpu_disponible():
            for lote in (16, 64):
                res, _ = motor_gpu.comparar_gpu(ca, cb, lote_mb=lote,
                                                detalle=total_esperado)
                resultados['GPU lote %d MB' % lote] = res
            # Con pocos bloques cada hilo recorre mas posiciones; el
            # resultado tiene que ser el mismo con 1 bloque que con 1024.
            for bloques in (1, 20):
                res, _ = motor_gpu.comparar_gpu(ca, cb, bloques=bloques,
                                                detalle=total_esperado)
                resultados['GPU %d bloques' % bloques] = res
        else:
            print('  GPU: no disponible, se omite')
    except Exception as error:
        print('  GPU: error al probar (%s)' % error)

    try:
        import motor_npu
        if motor_npu.runtime_disponible():
            for capas in (4, 8):
                res, _ = motor_npu.comparar_npu(ca, cb, capas=capas,
                                                detalle=total_esperado)
                resultados['NPU %d capas' % capas] = res
        else:
            print('  NPU: onnxruntime no disponible, se omite')
    except Exception as error:
        print('  NPU: error al probar (%s)' % error)

    print('')
    for nombre, res in resultados.items():
        comprobar('%-24s cuenta %d diferencias'
                  % (nombre, res['diferencias']),
                  res['diferencias'] == total_esperado,
                  'esperadas %d' % total_esperado)

    print('')
    for nombre, res in resultados.items():
        posiciones = [d['posicion'] for d in res['detalle']]
        comprobar('%-24s posiciones exactas' % nombre,
                  posiciones == lista_esperada,
                  'la lista de posiciones no coincide con la verdad')

    ca.cerrar()
    cb.cerrar()
    return resultados


# ---------------------------------------------------------------------------
# Parte 2: robustez
# ---------------------------------------------------------------------------

def probar_control_sustituciones():
    """Prueba de control: sustituciones conocidas sobre un recorte real de B.

    POR QUE ES IMPRESCINDIBLE
    Al emparejar las secuencias, los dos genomas del profesor dan CERO
    diferencias, porque son el mismo ensamblaje publicado por dos bases de
    datos distintas. Ese cero es el resultado correcto, pero por si solo no
    demuestra nada: un comparador averiado que devolviera siempre cero
    pasaria exactamente igual.

    Esta es la unica prueba que demuestra lo contrario: que el comparador SI
    encuentra diferencias cuando las hay, sobre datos del mismo archivo que
    se entrega. Se toma un recorte del archivo B de verdad, se le SUSTITUYEN
    (nunca se insertan ni se borran, para que la longitud no cambie) una
    cantidad exacta de bases, y se exige que las tres plataformas encuentren
    esa cantidad y en esas posiciones.

    Se prueba por los dos caminos, porque son codigo distinto:
      - comparando las cadenas de corrido
      - comparando con EMPAREJAMIENTO POR SECUENCIA activado, que es el modo
        por defecto de la aplicacion
    """
    import generar_par
    import secuencias

    print('')
    print('PARTE 1b - PRUEBA DE CONTROL: SUSTITUCIONES SOBRE B')
    print('-' * 62)

    origen = None
    for candidato in (os.path.join('..', 'P1.1',
                                   'GCF_000001405.40_GRCh38.p14_genomic.fna'),
                      os.path.join('..', 'P1.3', 'datos',
                                   'recorte_50MB.fna')):
        if os.path.exists(candidato):
            origen = candidato
            break
    if origen is None:
        print('  (no hay archivo de origen; se omite)')
        return

    esperadas = 12345
    print('  Origen : %s' % origen)
    datos = generar_par.generar(origen, salida=CARPETA_TEMP, mb=20,
                                diferencias=esperadas, semilla=987)
    print('  Recorte: %d bases con %d sustituciones conocidas'
          % (datos['largo'], datos['diferencias']))
    print('')

    ca = comparador.preparar(datos['ruta_a'], cache=CARPETA_TEMP)
    cb = comparador.preparar(datos['ruta_b'], cache=CARPETA_TEMP)

    comprobar('la sustitucion no cambio la longitud',
              ca.largo == cb.largo,
              '%d frente a %d' % (ca.largo, cb.largo))

    lista = list(datos['posiciones'])

    motores = [('CPU x1', lambda a, b: motor_cpu.comparar(
        a, b, detalle=esperadas))]
    for n in (4, 8):
        motores.append(('CPU x%d' % n,
                        lambda a, b, n=n: motor_cpu.comparar_paralelo(
                            a, b, procesos=n, detalle=esperadas)))
    try:
        import motor_gpu
        if motor_gpu.gpu_disponible():
            motores.append(('GPU', lambda a, b: motor_gpu.comparar_gpu(
                a, b, detalle=esperadas)))
    except Exception:
        pass
    try:
        import motor_npu
        if motor_npu.runtime_disponible():
            motores.append(('NPU', lambda a, b: motor_npu.comparar_npu(
                a, b, capas=8, detalle=esperadas)))
    except Exception:
        pass

    print('  De corrido:')
    for nombre, motor in motores:
        try:
            resultado, _ = motor(ca, cb)
        except Exception as error:
            comprobar('%-7s ejecuta sin error' % nombre, False, str(error))
            continue
        comprobar('%-7s encuentra las %d sustituciones'
                  % (nombre, esperadas),
                  resultado['diferencias'] == esperadas,
                  'encontro %d' % resultado['diferencias'])
        comprobar('%-7s marca el resultado como valido' % nombre,
                  resultado.get('valido', False))
        posiciones = [d['posicion'] for d in resultado['detalle']]
        comprobar('%-7s acierta las posiciones exactas' % nombre,
                  posiciones == lista,
                  'la lista no coincide con la verdad')

    # -- el mismo par, ahora con emparejamiento por secuencia -------------
    print('')
    print('  Con emparejamiento por secuencia:')
    ra = secuencias.indexar(datos['ruta_a'], cache=CARPETA_TEMP, forzar=True)
    rb = secuencias.indexar(datos['ruta_b'], cache=CARPETA_TEMP, forzar=True)
    parejas, solo_a, solo_b = secuencias.emparejar(ra, rb)

    comprobar('empareja la unica secuencia del par',
              len(parejas) == 1 and not solo_a and not solo_b,
              '%d parejas, %d y %d sueltas'
              % (len(parejas), len(solo_a), len(solo_b)))

    for nombre, motor in motores:
        try:
            resumen = secuencias.comparar_parejas(ca, cb, parejas, motor)
        except Exception as error:
            comprobar('%-7s con emparejamiento ejecuta sin error' % nombre,
                      False, str(error))
            continue
        comprobar('%-7s con emparejamiento encuentra las %d'
                  % (nombre, esperadas),
                  resumen['diferencias'] == esperadas,
                  'encontro %d' % resumen['diferencias'])
        comprobar('%-7s con emparejamiento no deja filas invalidas' % nombre,
                  resumen['invalidas'] == 0,
                  '%d invalidas' % resumen['invalidas'])

    ca.cerrar()
    cb.cerrar()


def probar_validacion():
    """Comprueba que un conteo imposible no se puede publicar.

    Es la red que se puso despues del fallo de DirectML: el motor de NPU
    devolvia veinte digitos y el numero salia tan campante en la ventana y en
    la matriz de evidencias.
    """
    print('')
    print('PARTE 1c - VALIDACION DE CONTEOS IMPOSIBLES')
    print('-' * 62)

    class Falsa(object):
        largo = 1000
        datos = np.zeros(1000, dtype=np.uint8)

    a = Falsa()
    b = Falsa()
    vacio = np.empty(0, dtype=np.int64)

    bueno = comparador.resumir(10, vacio, a, b, 1000, 'prueba')
    comprobar('un conteo normal se marca valido', bueno['valido'])

    for total in (1001, 10 ** 20, -5):
        malo = comparador.resumir(total, vacio, a, b, 1000, 'prueba')
        comprobar('un conteo de %d sobre 1000 se marca invalido' % total,
                  not malo['valido'] and bool(malo['motivo_invalido']))
        comprobar('  y no publica una similitud inventada',
                  malo['similitud_pct'] == 0.0)


def probar_emparejamiento():
    """Comprueba el indice de secuencias y el emparejamiento entre archivos."""
    import secuencias

    print('')
    print('PARTE 1d - EMPAREJAMIENTO POR SECUENCIA')
    print('-' * 62)

    ruta = os.path.join(CARPETA_TEMP, 'mini_par.fna')
    with open(ruta, 'w', encoding='utf-8') as f:
        f.write('>CM000001.1 Homo sapiens chromosome 1, GRCh38 reference '
                'primary assembly\n')
        f.write('ACGT' * 20 + '\n')
        f.write('>KI000002.1 Homo sapiens chromosome 1 unlocalized contig\n')
        f.write('TTTT' * 5 + '\n')

    registros = secuencias.indexar(ruta, cache=CARPETA_TEMP, forzar=True)
    comprobar('el indice encuentra los dos registros', len(registros) == 2,
              'encontro %d' % len(registros))
    if len(registros) == 2:
        comprobar('el primer registro empieza en 0 y mide 80',
                  registros[0].inicio == 0 and registros[0].largo == 80,
                  '%d + %d' % (registros[0].inicio, registros[0].largo))
        comprobar('el segundo empieza donde acaba el primero',
                  registros[1].inicio == 80 and registros[1].largo == 20,
                  '%d + %d' % (registros[1].inicio, registros[1].largo))
        comprobar('reconoce el cromosoma principal',
                  registros[0].cromosoma == '1',
                  'dijo %r' % registros[0].cromosoma)
        comprobar('no confunde un contig con un cromosoma',
                  registros[1].cromosoma is None,
                  'dijo %r' % registros[1].cromosoma)

        # Mismos registros en distinto orden: tienen que emparejarse igual.
        # Es exactamente lo que pasa entre GenBank y RefSeq.
        invertidos = [registros[1], registros[0]]
        parejas, solo_a, solo_b = secuencias.emparejar(registros, invertidos)
        comprobar('empareja aunque cambie el orden', len(parejas) == 2,
                  '%d parejas' % len(parejas))
        comprobar('no deja registros sueltos',
                  not solo_a and not solo_b)

        # Un registro sin pareja se informa, no se compara.
        parejas, solo_a, solo_b = secuencias.emparejar(registros,
                                                       registros[:1])
        comprobar('un registro sin pareja se informa aparte',
                  len(parejas) == 1 and len(solo_a) == 1,
                  '%d parejas, %d sueltos' % (len(parejas), len(solo_a)))


def probar_numeros():
    print('')
    print('PARTE 2a - VALORES NUMERICOS INVALIDOS')
    print('-' * 62)

    _, logicos = motor_cpu.detectar_nucleos()
    largo = 100 << 20     # 100 MB de referencia

    # Valores limite y disparatados para el numero de procesos.
    casos = [
        ('cero procesos', 0, 1),
        ('un proceso', 1, 1),
        ('el maximo exacto', logicos, logicos),
        ('uno mas que el maximo', logicos + 1, logicos),
        ('el doble del maximo', logicos * 2, logicos),
        ('negativo', -5, 1),
        ('muy negativo', -999999, 1),
        ('absurdamente grande', 10 ** 9, logicos),
    ]
    for nombre, pedido, esperado in casos:
        obtenido, motivo = motor_cpu.procesos_validos(pedido, largo)
        ok = obtenido == esperado
        # Cuando el programa cambia un valor tiene que decir por que. Un
        # ajuste silencioso es peor que el error: el usuario cree que corrio
        # lo que pidio.
        if ok and obtenido != pedido:
            ok = bool(motivo)
        comprobar('procesos, %-22s %s -> %s'
                  % (nombre, pedido, obtenido), ok,
                  'esperado %s, motivo=%r' % (esperado, motivo))

    # Tipos que no son numeros.
    for basura in ('abc', '', '   ', None, '3.5', '12a', [], {}):
        obtenido, motivo = motor_cpu.procesos_validos(basura, largo)
        comprobar('procesos, basura %-14r -> %s' % (basura, obtenido),
                  1 <= obtenido <= logicos and bool(motivo),
                  'devolvio %s con motivo=%r' % (obtenido, motivo))

    # Cadenas pequenas: no se deben repartir entre muchos procesos.
    obtenido, motivo = motor_cpu.procesos_validos(logicos, 1 << 20)
    comprobar('procesos, cadena de 1 MB con %d pedidos -> %d'
              % (logicos, obtenido),
              obtenido < logicos and bool(motivo),
              'no recorto el numero de procesos')

    print('')
    try:
        import motor_gpu
        for nombre, pedido, esperado in (
                ('cero', 0, motor_gpu.LOTE_MIN_MB),
                ('negativo', -10, motor_gpu.LOTE_MIN_MB),
                ('gigante', 10 ** 6, motor_gpu.LOTE_MAX_MB),
                ('texto', 'abc', motor_gpu.LOTE_MB),
                ('valido', 64, 64)):
            obtenido, motivo = motor_gpu.lote_valido(pedido)
            ok = obtenido == esperado
            if ok and obtenido != pedido:
                ok = bool(motivo)
            comprobar('lote GPU, %-10s %s -> %s' % (nombre, pedido, obtenido),
                      ok, 'esperado %s' % esperado)
        for nombre, pedido, esperado in (
                ('cero', 0, motor_gpu.BLOQUES_MIN),
                ('negativo', -4, motor_gpu.BLOQUES_MIN),
                ('gigante', 10 ** 6, motor_gpu.BLOQUES_MAX),
                ('texto', 'abc', motor_gpu.BLOQUES),
                ('valido', 20, 20)):
            obtenido, motivo = motor_gpu.bloques_validos(pedido)
            ok = obtenido == esperado
            if ok and obtenido != pedido:
                ok = bool(motivo)
            comprobar('bloques GPU, %-8s %s -> %s' % (nombre, pedido,
                                                      obtenido),
                      ok, 'esperado %s' % esperado)
    except ImportError:
        print('  GPU: no se pudo importar, se omite')

    print('')
    try:
        import motor_npu
        for nombre, pedido, esperado in (
                ('cero', 0, motor_npu.CAPAS_MIN),
                ('negativo', -3, motor_npu.CAPAS_MIN),
                ('demasiadas', 500, motor_npu.CAPAS_MAX),
                ('texto', 'ocho', motor_npu.CAPAS),
                ('valido', 8, 8)):
            obtenido, motivo = motor_npu.capas_validas(pedido)
            ok = obtenido == esperado
            if ok and obtenido != pedido:
                ok = bool(motivo)
            comprobar('capas NPU, %-11s %s -> %s'
                      % (nombre, pedido, obtenido), ok,
                      'esperado %s' % esperado)
    except ImportError:
        print('  NPU: no se pudo importar, se omite')


def probar_archivos(rutas):
    print('')
    print('PARTE 2b - ARCHIVOS DE ENTRADA INVALIDOS')
    print('-' * 62)

    espera_error('ruta vacia', comparador.validar_fasta, '')
    espera_error('ruta None', comparador.validar_fasta, None)
    espera_error('archivo inexistente', comparador.validar_fasta,
                 rutas['inexistente'])
    espera_error('una carpeta en vez de un archivo',
                 comparador.validar_fasta, rutas['carpeta'])
    espera_error('archivo de 0 bytes', comparador.validar_fasta,
                 rutas['vacio'])
    espera_error('archivo binario', comparador.validar_fasta,
                 rutas['binario'])
    espera_error('texto que no es FASTA', comparador.validar_fasta,
                 rutas['no_fasta'])

    # Un FASTA con cabeceras pero sin una sola base es valido como archivo,
    # asi que pasa la validacion de formato pero tiene que fallar al
    # preparar, que es cuando se descubre que no hay secuencia.
    espera_error('FASTA con cabeceras pero sin secuencia',
                 comparador.preparar, rutas['solo_cabecera'],
                 CARPETA_TEMP)

    # Un FASTA legitimo tiene que pasar.
    try:
        cadena = comparador.preparar(rutas['corto'], cache=CARPETA_TEMP)
        comprobar('FASTA correcto se acepta', cadena.largo == 10,
                  'largo %d, esperado 10' % cadena.largo)
    except Exception as error:
        comprobar('FASTA correcto se acepta', False, str(error))


def probar_cache_mismo_nombre():
    """Dos FASTA distintos con el mismo nombre no comparten cache.

    Reproduce el caso que dejo vacio el indice de par_200MB_B: el segundo
    archivo se llama igual y tiene la MISMA fecha que el primero (como pasa
    al copiar en Windows), y aun asi tiene que leerse lo suyo.
    """
    print('')
    print('PARTE 2e - CACHE CON DOS ARCHIVOS DEL MISMO NOMBRE')
    print('-' * 62)
    import shutil
    import secuencias
    raiz = os.path.join(CARPETA_TEMP, 'mismo_nombre')
    cache = os.path.join(raiz, 'cache')
    if os.path.isdir(raiz):
        shutil.rmtree(raiz)
    for carpeta, texto in (('uno', 'ACGTACGTAC'), ('dos', 'TTTTGGGGCCCCAA')):
        os.makedirs(os.path.join(raiz, carpeta))
        with open(os.path.join(raiz, carpeta, 'gen.fna'), 'w') as f:
            f.write('>' + carpeta + chr(10) + texto + chr(10))
    uno = os.path.join(raiz, 'uno', 'gen.fna')
    dos = os.path.join(raiz, 'dos', 'gen.fna')
    fecha = os.stat(uno).st_mtime_ns
    os.utime(dos, ns=(fecha, fecha))

    a = comparador.preparar(uno, cache=cache)
    b = comparador.preparar(dos, cache=cache)
    comprobar('el segundo archivo no reutiliza el .seq del primero',
              not b.desde_cache and b.largo == 14,
              'desde_cache=%s largo=%d' % (b.desde_cache, b.largo))
    comprobar('su contenido es el suyo',
              bytes(b.datos[:4]) == b'TTTT', repr(bytes(b.datos[:4])))
    c = comparador.preparar(dos, cache=cache)
    comprobar('el mismo archivo otra vez si usa la cache', c.desde_cache)
    del a, b, c

    ra = secuencias.indexar(uno, cache=cache)
    rb = secuencias.indexar(dos, cache=cache)
    comprobar('el indice del segundo archivo es el suyo',
              [r.largo for r in ra] == [10] and [r.largo for r in rb] == [14],
              '%s / %s' % ([r.largo for r in ra], [r.largo for r in rb]))


def probar_pares(rutas):
    print('')
    print('PARTE 2c - PARES DE CADENAS PROBLEMATICOS')
    print('-' * 62)

    corta = comparador.preparar(rutas['corto'], cache=CARPETA_TEMP)
    larga = comparador.preparar(rutas['corto_distinto'], cache=CARPETA_TEMP)

    # Mismo archivo dos veces: cero diferencias y un aviso.
    avisos = comparador.validar_par(corta, corta)
    comprobar('mismo archivo dos veces avisa', bool(avisos),
              'no genero ningun aviso')
    res, _ = motor_cpu.comparar(corta, corta)
    comprobar('mismo archivo dos veces da 0 diferencias',
              res['diferencias'] == 0,
              'dio %d' % res['diferencias'])
    comprobar('mismo archivo dos veces da 100 % de similitud',
              abs(res['similitud_pct'] - 100.0) < 1e-9,
              'dio %.4f' % res['similitud_pct'])

    # Cadenas de distinto largo: se compara lo comun y se avisa del sobrante.
    avisos = comparador.validar_par(corta, larga)
    comprobar('largos distintos avisan', bool(avisos),
              'no genero ningun aviso')
    res, _ = motor_cpu.comparar(corta, larga)
    comprobar('largos distintos comparan solo lo comun',
              res['comparados'] == min(corta.largo, larga.largo),
              'comparo %d' % res['comparados'])
    comprobar('largos distintos publican el desfase',
              res['desfase'] == corta.largo - larga.largo,
              'desfase %d' % res['desfase'])

    # ACGTACGTAC contra ACGTTCGTAG: difieren en las posiciones 4 y 9.
    posiciones = [d['posicion'] for d in res['detalle']]
    comprobar('diferencias de una cadena corta son exactas',
              posiciones == [4, 9],
              'dio %s, esperado [4, 9]' % posiciones)
    comprobar('fila y columna de la primera diferencia',
              res['detalle'][0]['fila'] == 1
              and res['detalle'][0]['columna'] == 5,
              'dio fila %d columna %d' % (res['detalle'][0]['fila'],
                                          res['detalle'][0]['columna']))

    corta.cerrar()
    larga.cerrar()


def probar_rejilla():
    print('')
    print('PARTE 2d - CONVERSION A FILA Y COLUMNA')
    print('-' * 62)

    casos = [
        (0, 1, 1),
        (1, 1, 2),
        (79, 1, 80),
        (80, 2, 1),
        (159, 2, 80),
        (160, 3, 1),
        (1945577269, 24319716, 70),
    ]
    for posicion, fila, columna in casos:
        obtenida_f, obtenida_c = comparador.posicion_a_fila_columna(posicion)
        comprobar('posicion %-12d -> fila %d, columna %d'
                  % (posicion, fila, columna),
                  (obtenida_f, obtenida_c) == (fila, columna),
                  'dio fila %d columna %d' % (obtenida_f, obtenida_c))

    # La version vectorizada tiene que coincidir con la de uno en uno.
    muestra = np.array([p for p, _, _ in casos], dtype=np.int64)
    filas, columnas = comparador.filas_columnas(muestra)
    coinciden = all(
        (int(f), int(c)) == comparador.posicion_a_fila_columna(int(p))
        for p, f, c in zip(muestra, filas, columnas))
    comprobar('la conversion vectorizada coincide con la individual',
              coinciden)


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------

def main():
    print('=' * 62)
    print('VERIFICACION DEL P1.4')
    print('=' * 62)

    fisicos, logicos = motor_cpu.detectar_nucleos()
    print('CPU : %d nucleos fisicos / %d logicos' % (fisicos, logicos))

    try:
        import motor_gpu
        datos = motor_gpu.info_gpu()
        print('GPU : %s' % (datos.get('nombre') if datos['disponible']
                            else 'no disponible'))
    except Exception:
        print('GPU : no disponible')

    try:
        import motor_npu
        datos = motor_npu.info_npu()
        print('NPU : %s%s' % (datos.get('descripcion', 'no disponible'),
                              '' if datos.get('es_npu')
                              else '  (no es una NPU real)'))
    except Exception:
        print('NPU : no disponible')

    if len(sys.argv) >= 3:
        ruta_a, ruta_b = sys.argv[1], sys.argv[2]
        esperadas = np.empty(0, dtype=np.int64)
        print('')
        print('Par indicado por linea de comandos: no hay verdad conocida,')
        print('solo se comprobara que las plataformas coincidan entre si.')
    else:
        ruta_a, ruta_b, esperadas = preparar_datos()
        if ruta_a is None:
            return 1

    if esperadas.size:
        probar_correccion(ruta_a, ruta_b, esperadas)
    else:
        ca = comparador.preparar(ruta_a)
        cb = comparador.preparar(ruta_b)
        base, _ = motor_cpu.comparar(ca, cb)
        otro, _ = motor_cpu.comparar_paralelo(ca, cb)
        comprobar('CPU secuencial y paralelo coinciden',
                  base['diferencias'] == otro['diferencias'])

    rutas = archivos_trampa()
    probar_control_sustituciones()
    probar_validacion()
    probar_emparejamiento()
    probar_numeros()
    probar_archivos(rutas)
    probar_cache_mismo_nombre()
    probar_pares(rutas)
    probar_rejilla()

    print('')
    print('=' * 62)
    print('RESULTADO: %d comprobaciones correctas, %d fallidas'
          % (_aciertos, _fallos))
    print('=' * 62)
    return 0 if _fallos == 0 else 2


if __name__ == '__main__':
    multiprocessing.freeze_support()
    sys.exit(main())
