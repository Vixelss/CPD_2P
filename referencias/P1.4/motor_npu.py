# -*- coding: utf-8 -*-
r"""
motor_npu.py

Comparacion de dos cadenas de ADN sobre la NPU, la unidad de procesamiento
neuronal que llevan los procesadores modernos. Es la columna "NPU" de la
rubrica del P1.4.

QUE ES UNA NPU Y POR QUE ESTE MOTOR ES DISTINTO A LOS OTROS DOS
Una NPU no se programa como una CPU ni como una GPU. No se le escriben
kernels, no se le lanzan hilos y no ejecuta codigo arbitrario. Es un
acelerador de INFERENCIA: se le entrega un grafo de red neuronal, ya
cuantizado, y un runtime lo reparte entre el hardware disponible. Su
repertorio de operaciones aceleradas es estrecho, basicamente convoluciones y
multiplicaciones de matrices sobre enteros.

Por eso el paralelismo de la NPU no se mide en procesos ni en hilos sino en
CAPAS del grafo, que es justo como lo plantea el enunciado ("NPU n capas").

COMO SE CONVIERTE UNA COMPARACION EN UNA RED NEURONAL
Comparar dos caracteres es preguntar si son distintos. Escrito como grafo:

    1. Cast      : los dos lotes de bytes pasan a numeros en coma flotante.
    2. Sub       : se restan. Si los caracteres son iguales da cero.
    3. Mul       : se eleva al cuadrado, para que el signo no importe y toda
                   diferencia quede estrictamente por encima de cero.
    4. Greater   : se compara contra 0.5. Dos bytes distintos se diferencian
                   al menos en 1, asi que el cuadrado vale al menos 1 y el
                   umbral separa sin ambiguedad.
    5. Cast      : el resultado booleano vuelve a numero. Es la MASCARA: vale
                   1 donde las cadenas difieren y 0 donde coinciden.

    6..n. Conv   : n capas de convolucion de nucleo 2 y paso 2, con pesos
                   fijos [1, 1]. Cada capa suma los elementos de dos en dos y
                   reduce el vector a la mitad. Es una reduccion en arbol.
    n+1. ReduceSum: suma lo que queda y da el numero total de diferencias.

    NonZero      : rama aparte que devuelve las posiciones de la mascara.

Las capas de convolucion son las que hacen que este grafo tenga algo que la
NPU pueda acelerar de verdad: una convolucion es una multiplicacion de
matrices, que es exactamente para lo que esta construida. Un grafo que solo
restara y comparara se le devolveria entero a la CPU sin que nadie avisara,
y se estaria diciendo "esto corre en la NPU" mientras corre en la CPU.

Que la reduccion se haga en arbol y no de un golpe tambien tiene sentido
fisico: una NPU trabaja por capas, y cada capa procesa todos sus elementos en
paralelo. Sumar tres millones de numeros de dos en dos son veintidos capas,
no tres millones de pasos.

SOBRE LA CUANTIZACION
El enunciado del profesor insiste en la cuantizacion INT8 como el rasgo
propio de la NPU. Aqui hay un detalle que conviene entender y que es un buen
argumento para la defensa: la mascara solo contiene ceros y unos, y los pesos
de las convoluciones son unos. Al cuantizar ese grafo a enteros de 8 bits no
se pierde NADA: el resultado es identico bit a bit al de coma flotante,
porque el dominio del dato ya era binario. En una red de vision la
cuantizacion siempre degrada un poco la precision; aqui no, y se puede
demostrar comparando contra la CPU.

COMO SE ELIGE EL HARDWARE
ONNX Runtime no habla directamente con la NPU: lo hace a traves de un
"proveedor de ejecucion", una pieza distinta por fabricante. Este motor los
prueba en orden de preferencia y se queda con el primero disponible:

    QNNExecutionProvider       NPU Hexagon de Qualcomm (Snapdragon X)
    OpenVINOExecutionProvider  NPU de Intel (Core Ultra), con device NPU
    VitisAIExecutionProvider   NPU XDNA de AMD (Ryzen AI)
    DmlExecutionProvider       DirectML: cualquier dispositivo DirectX 12
    CPUExecutionProvider       reserva final, siempre disponible

El resultado dice SIEMPRE cual se uso y si es una NPU de verdad o no. Un
equipo sin NPU no obtiene una columna falsa: obtiene el mismo grafo corriendo
donde se pueda y un aviso explicito de que no habia NPU. Eso es lo unico
honesto que se puede publicar en un informe.

POR QUE EL RUNTIME VIAJA DENTRO DEL PROYECTO
En la carpeta vendor\ va una copia de onnxruntime. El modulo la anade al
camino de busqueda de Python antes de importar nada, de modo que el programa
funciona en un equipo donde nunca se instalo. Es lo que permite copiar la
carpeta a otra maquina y ejecutar sin preparar nada.

RESPUESTAS A LA RUBRICA, COLUMNA "NPU"
    LIBRERIA    onnxruntime (con el proveedor de ejecucion del fabricante)
    INSTRUCCION onnxruntime.InferenceSession(modelo, providers=[...])
                seguido de sesion.run(...)
    Ver evidencias.py, que es donde se recogen formalmente.

API publica:
    npu_disponible()                    -> bool
    info_npu()                          -> dict con el proveedor elegido
    capas_validas(n)                    -> (n_ajustado, motivo)
    comparar_npu(cadena_a, cadena_b)    -> (dict_resultado, tiempo)
"""

import os
import sys
import time

import numpy as np

import comparador
from comparador import ErrorEntrada


# ---------------------------------------------------------------------------
# Carga del runtime, primero desde vendor\
# ---------------------------------------------------------------------------

_AQUI = os.path.dirname(os.path.abspath(__file__))
_VENDOR = os.path.join(_AQUI, 'vendor')

if os.path.isdir(_VENDOR) and _VENDOR not in sys.path:
    # Se antepone a proposito: si la maquina tuviera una version distinta
    # instalada, la del proyecto es la que se probo y la que debe mandar.
    sys.path.insert(0, _VENDOR)

try:
    import onnxruntime as ort
    _HAY_ORT = True
    _ERROR_ORT = None
except Exception as _error:                           # pragma: no cover
    ort = None
    _HAY_ORT = False
    _ERROR_ORT = str(_error)


# Proveedores de ejecucion en orden de preferencia. El segundo elemento dice
# si ese proveedor corresponde a una NPU de verdad, y el tercero es el nombre
# legible que se publica en el informe.
PREFERENCIA = [
    ('QNNExecutionProvider', True, 'NPU Hexagon (Qualcomm Snapdragon)'),
    ('OpenVINOExecutionProvider', True, 'NPU Intel AI Boost (Core Ultra)'),
    ('VitisAIExecutionProvider', True, 'NPU XDNA (AMD Ryzen AI)'),
    ('DmlExecutionProvider', False, 'DirectML (acelerador DirectX 12)'),
    ('CPUExecutionProvider', False, 'CPU (sin acelerador)'),
]

# Tamano de lote por defecto, en numero de posiciones. La NPU trabaja sobre
# tensores en memoria, asi que el lote acota cuanta memoria pide el grafo.
LOTE = 4 << 20              # 4 M posiciones

# Capas de reduccion por defecto. Con 22 capas se reduce un lote de 4 M
# elementos hasta 1, que es el arbol completo.
CAPAS = 8

CAPAS_MIN = 1
CAPAS_MAX = 22

LOTE_MIN = 1 << 16          # 64 K
LOTE_MAX = 32 << 20         # 32 M

# Carpeta donde se guardan los modelos ONNX generados.
CARPETA_MODELOS = os.path.join(_AQUI, 'modelos')


# ---------------------------------------------------------------------------
# Disponibilidad
# ---------------------------------------------------------------------------

def runtime_disponible():
    """Indica si se pudo cargar onnxruntime, de vendor\\ o del sistema."""
    return _HAY_ORT


def proveedores_disponibles():
    """Lista de proveedores de ejecucion que ofrece este equipo."""
    if not _HAY_ORT:
        return []
    try:
        return list(ort.get_available_providers())
    except Exception:
        return []


def elegir_proveedor(forzado=None):
    """Devuelve (nombre, es_npu, descripcion) del mejor proveedor disponible.

    Si se pasa 'forzado' se intenta ese y solo ese; sirve para el benchmark,
    que compara el mismo grafo sobre distintos destinos.
    """
    disponibles = proveedores_disponibles()
    if not disponibles:
        return None, False, 'ninguno'

    if forzado:
        for nombre, es_npu, descripcion in PREFERENCIA:
            if nombre == forzado:
                if nombre in disponibles:
                    return nombre, es_npu, descripcion
                return None, False, 'no disponible'
        if forzado in disponibles:
            return forzado, False, forzado
        return None, False, 'no disponible'

    for nombre, es_npu, descripcion in PREFERENCIA:
        if nombre in disponibles:
            return nombre, es_npu, descripcion

    # Proveedor que no esta en la lista de preferencia: se acepta igual, pero
    # no se afirma que sea una NPU porque no se sabe.
    return disponibles[0], False, disponibles[0]


def npu_disponible():
    """Dice si hay una NPU de verdad accesible en este equipo."""
    _, es_npu, _ = elegir_proveedor()
    return bool(es_npu)


def info_npu():
    """Diccionario con el estado de la ruta NPU, listo para mostrar."""
    datos = {
        'runtime': _HAY_ORT,
        'version_runtime': getattr(ort, '__version__', None) if _HAY_ORT
        else None,
        'desde_vendor': os.path.isdir(_VENDOR),
        'proveedores': proveedores_disponibles(),
    }

    if not _HAY_ORT:
        datos.update({
            'disponible': False,
            'es_npu': False,
            'motivo': ('No se pudo cargar onnxruntime, que es la libreria que '
                       'habla con la NPU.\nDetalle: %s' % _ERROR_ORT),
        })
        return datos

    nombre, es_npu, descripcion = elegir_proveedor()
    datos.update({
        'disponible': nombre is not None,
        'proveedor': nombre,
        'es_npu': es_npu,
        'descripcion': descripcion,
    })

    if not es_npu:
        datos['motivo'] = (
            'Este equipo no expone ninguna NPU a traves de onnxruntime. '
            'El grafo se ejecutara sobre %s, que es el mejor destino '
            'disponible aqui. En un equipo con NPU el programa la elige '
            'por si solo, siempre que este instalado el onnxruntime de su '
            'fabricante (el de vendor\\ solo trae DirectML).' % descripcion)

    return datos


# ---------------------------------------------------------------------------
# Validacion de parametros
# ---------------------------------------------------------------------------

def capas_validas(pedidas):
    """Ajusta el numero de capas de reduccion a un valor con sentido.

    Cada capa divide el vector entre dos, asi que mas de veintidos capas
    sobre un lote de cuatro millones no tendrian nada que reducir, y menos de
    una dejaria el grafo sin ninguna operacion que la NPU pueda acelerar.
    """
    try:
        pedidas = int(pedidas)
    except (TypeError, ValueError):
        return CAPAS, ('"%s" no es un numero de capas valido; se usan %d.'
                       % (pedidas, CAPAS))

    if pedidas < CAPAS_MIN:
        return CAPAS_MIN, ('El grafo necesita al menos %d capa de reduccion; '
                           'se usa %d.' % (CAPAS_MIN, CAPAS_MIN))

    if pedidas > CAPAS_MAX:
        return CAPAS_MAX, ('Con mas de %d capas el vector ya se ha reducido a '
                           'un solo elemento y no queda nada que sumar; se '
                           'usan %d.' % (CAPAS_MAX, CAPAS_MAX))

    return pedidas, None


def lote_valido(pedido):
    """Ajusta el tamano de lote a un valor que la memoria pueda sostener."""
    try:
        pedido = int(pedido)
    except (TypeError, ValueError):
        return LOTE, ('"%s" no es un tamano de lote valido; se usan %d.'
                      % (pedido, LOTE))
    if pedido < LOTE_MIN:
        return LOTE_MIN, ('Un lote de %d posiciones es demasiado pequeno; se '
                          'usa %d.' % (pedido, LOTE_MIN))
    if pedido > LOTE_MAX:
        return LOTE_MAX, ('Un lote de %d posiciones pediria demasiada '
                          'memoria; se usa %d.' % (pedido, LOTE_MAX))
    return pedido, None


# ---------------------------------------------------------------------------
# Construccion del grafo
# ---------------------------------------------------------------------------

def ruta_modelo(capas):
    """Ruta del archivo .onnx correspondiente a un numero de capas."""
    return os.path.join(CARPETA_MODELOS, 'comparador_%dcapas.onnx' % capas)


def construir_modelo(capas, destino=None):
    """Genera el grafo ONNX del comparador con 'capas' capas de reduccion.

    Necesita la libreria onnx, que solo hace falta para CREAR el modelo. Los
    modelos ya construidos viajan con el proyecto, de modo que en la maquina
    donde se ejecuta basta con onnxruntime.
    """
    try:
        import onnx
        from onnx import helper, TensorProto, numpy_helper
    except ImportError:
        raise ErrorEntrada(
            'Para construir el grafo hace falta la libreria onnx.\n'
            'Los modelos ya generados estan en la carpeta modelos\\ y no '
            'requieren nada.')

    if destino is None:
        destino = ruta_modelo(capas)
    if not os.path.isdir(os.path.dirname(destino)):
        os.makedirs(os.path.dirname(destino))

    # Entradas: dos vectores de bytes de longitud variable. La dimension se
    # deja simbolica ('L') para que el mismo modelo sirva para cualquier lote,
    # incluido el ultimo, que casi nunca esta completo.
    ent_a = helper.make_tensor_value_info('A', TensorProto.UINT8, [1, 1, 'L'])
    ent_b = helper.make_tensor_value_info('B', TensorProto.UINT8, [1, 1, 'L'])

    nodos = []
    iniciales = []

    # Capa 1: a numeros con los que se pueda operar.
    nodos.append(helper.make_node('Cast', ['A'], ['fa'],
                                  to=TensorProto.FLOAT, name='capa1_cast_a'))
    nodos.append(helper.make_node('Cast', ['B'], ['fb'],
                                  to=TensorProto.FLOAT, name='capa1_cast_b'))

    # Capa 2: resta. Cero donde las cadenas coinciden.
    nodos.append(helper.make_node('Sub', ['fa', 'fb'], ['resta'],
                                  name='capa2_resta'))

    # Capa 3: cuadrado. Elimina el signo sin usar Abs, que algunos
    # proveedores no aceleran.
    nodos.append(helper.make_node('Mul', ['resta', 'resta'], ['cuadrado'],
                                  name='capa3_cuadrado'))

    # Capa 4: umbral. Dos bytes distintos se diferencian al menos en 1, asi
    # que su cuadrado vale al menos 1 y 0.5 separa sin ambiguedad.
    umbral = numpy_helper.from_array(np.array([0.5], dtype=np.float32),
                                     'umbral')
    iniciales.append(umbral)
    nodos.append(helper.make_node('Greater', ['cuadrado', 'umbral'],
                                  ['distinto'], name='capa4_umbral'))

    # Capa 5: la mascara. Vale 1 donde difieren y 0 donde coinciden.
    nodos.append(helper.make_node('Cast', ['distinto'], ['mascara'],
                                  to=TensorProto.FLOAT, name='capa5_mascara'))

    # Capas de reduccion en arbol. Cada una suma de dos en dos con una
    # convolucion de nucleo 2 y paso 2, con pesos fijos a uno. Es lo que la
    # NPU acelera de verdad: una convolucion es una multiplicacion de
    # matrices.
    pesos = numpy_helper.from_array(
        np.ones((1, 1, 2), dtype=np.float32), 'pesos_suma')
    iniciales.append(pesos)

    entrada = 'mascara'
    for nivel in range(capas):
        salida = 'reduccion_%d' % nivel
        nodos.append(helper.make_node(
            'Conv', [entrada, 'pesos_suma'], [salida],
            kernel_shape=[2], strides=[2], pads=[0, 0],
            name='capa%d_conv' % (6 + nivel)))
        entrada = salida

    # Ultima capa: suma lo que quede del arbol.
    nodos.append(helper.make_node('ReduceSum', [entrada], ['suma'],
                                  keepdims=0, name='capa_final_suma'))
    nodos.append(helper.make_node('Cast', ['suma'], ['cuenta'],
                                  to=TensorProto.INT64, name='capa_final_cast'))

    # Rama aparte: las posiciones donde esta la mascara encendida. NonZero no
    # es una operacion que una NPU acelere, asi que el runtime la coloca donde
    # pueda; es una parte legitima del grafo y el informe lo dice.
    nodos.append(helper.make_node('NonZero', ['distinto'], ['indices'],
                                  name='rama_posiciones'))

    sal_cuenta = helper.make_tensor_value_info('cuenta', TensorProto.INT64,
                                               [1])
    sal_indices = helper.make_tensor_value_info('indices', TensorProto.INT64,
                                                [3, 'D'])

    grafo = helper.make_graph(nodos, 'comparador_adn_%dcapas' % capas,
                              [ent_a, ent_b], [sal_cuenta, sal_indices],
                              initializer=iniciales)
    modelo = helper.make_model(
        grafo, producer_name='P1.4 comparador de ADN',
        opset_imports=[helper.make_opsetid('', 13)])
    modelo.ir_version = 9
    onnx.checker.check_model(modelo)
    onnx.save(modelo, destino)
    return destino


def asegurar_modelo(capas):
    """Devuelve la ruta del modelo, construyendolo si todavia no existe."""
    ruta = ruta_modelo(capas)
    if not os.path.exists(ruta):
        construir_modelo(capas, ruta)
    return ruta


# ---------------------------------------------------------------------------
# Sesion
# ---------------------------------------------------------------------------

_sesiones = {}


def _crear_sesion(capas, proveedor):
    """Crea (o recupera de cache) la sesion de inferencia.

    Cargar un modelo y preparar el proveedor cuesta tiempo, y ese tiempo no es
    trabajo del algoritmo: es el equivalente a la compilacion JIT de los
    kernels CUDA, que en el P1.2 y el P1.3 tambien quedaba fuera del reloj.
    Por eso la sesion se cachea y se crea antes de medir.
    """
    clave = (capas, proveedor)
    if clave in _sesiones:
        return _sesiones[clave]

    ruta = asegurar_modelo(capas)
    opciones = ort.SessionOptions()
    # Silencia los avisos del proveedor sobre nodos que no puede acelerar.
    # No se ocultan: se publican en el resultado, ordenadamente.
    opciones.log_severity_level = 3

    proveedores = [proveedor]
    if proveedor != 'CPUExecutionProvider':
        # Reserva obligatoria: cualquier nodo que el acelerador no soporte
        # tiene que poder ejecutarse en algun sitio.
        proveedores.append('CPUExecutionProvider')

    opciones_proveedor = []
    for nombre in proveedores:
        if nombre == 'OpenVINOExecutionProvider':
            # En Intel hay que pedir la NPU explicitamente: por defecto
            # OpenVINO elige la CPU, y entonces la columna NPU del informe
            # estaria mintiendo.
            opciones_proveedor.append({'device_type': 'NPU'})
        elif nombre == 'QNNExecutionProvider':
            # backend_path apunta al acelerador Hexagon; sin esto QNN usa su
            # backend de CPU.
            opciones_proveedor.append({'backend_path': 'QnnHtp.dll'})
        else:
            opciones_proveedor.append({})

    try:
        sesion = ort.InferenceSession(ruta, sess_options=opciones,
                                      providers=proveedores,
                                      provider_options=opciones_proveedor)
    except Exception:
        # Si el proveedor rechaza sus opciones (driver viejo, backend que no
        # esta), se reintenta sin ellas antes de rendirse. Es preferible
        # ejecutar en el acelerador con la configuracion por defecto que caer
        # a la CPU por un detalle de configuracion.
        sesion = ort.InferenceSession(ruta, sess_options=opciones,
                                      providers=proveedores)

    _sesiones[clave] = sesion
    return sesion


def precalentar(capas=CAPAS, proveedor=None):
    """Crea la sesion y hace una inferencia de prueba, fuera de la medicion.

    Devuelve los segundos que costo. La primera inferencia de una sesion
    siempre es mas lenta: el proveedor reserva memoria, compila su plan y, en
    una NPU, carga el grafo en el acelerador.
    """
    if not _HAY_ORT:
        return 0.0
    capas, _ = capas_validas(capas)
    if proveedor is None:
        proveedor, _, _ = elegir_proveedor()
    if proveedor is None:
        return 0.0

    inicio = time.perf_counter()
    sesion = _crear_sesion(capas, proveedor)

    # Lote minimo que satisface las capas: cada una divide entre dos.
    minimo = 1 << capas
    muestra_a = np.zeros((1, 1, minimo), dtype=np.uint8)
    muestra_b = np.zeros((1, 1, minimo), dtype=np.uint8)
    muestra_b[0, 0, 0] = 1
    sesion.run(None, {'A': muestra_a, 'B': muestra_b})
    return time.perf_counter() - inicio


# ---------------------------------------------------------------------------
# API publica
# ---------------------------------------------------------------------------

def comparar_npu(cadena_a, cadena_b, capas=CAPAS, lote=LOTE, proveedor=None,
                 detalle=None, progreso=None):
    """Compara las dos cadenas ejecutando el grafo sobre la NPU.

    Devuelve (dict_resultado, tiempo_en_segundos). El tiempo no incluye la
    creacion de la sesion ni la primera inferencia: ver 'preparacion_s'.
    """
    if not _HAY_ORT:
        raise ErrorEntrada(
            'No se pudo cargar onnxruntime, que es la libreria que habla con '
            'la NPU.\nDetalle: %s' % _ERROR_ORT)

    largo = min(cadena_a.largo, cadena_b.largo)
    if largo == 0:
        raise ErrorEntrada('No hay nada que comparar: una cadena esta vacia.')

    capas, aviso_capas = capas_validas(capas)
    lote, aviso_lote = lote_valido(lote)

    nombre_proveedor, es_npu, descripcion = elegir_proveedor(proveedor)
    if nombre_proveedor is None:
        raise ErrorEntrada(
            'onnxruntime no ofrece ningun proveedor de ejecucion en este '
            'equipo.')

    # El lote tiene que ser multiplo de 2**capas para que las convoluciones
    # de paso 2 cuadren sin restos. Se redondea hacia abajo.
    alineacion = 1 << capas
    lote = max(alineacion, (lote // alineacion) * alineacion)

    tope = comparador.MAX_DETALLE if detalle is None else max(0, detalle)

    # -- preparacion, deliberadamente fuera del cronometro ----------------
    inicio_preparacion = time.perf_counter()
    sesion = _crear_sesion(capas, nombre_proveedor)
    precalentar(capas, nombre_proveedor)
    usados = list(sesion.get_providers())
    preparacion = time.perf_counter() - inicio_preparacion

    va = cadena_a.datos
    vb = cadena_b.datos
    total = 0
    recogidas = []
    faltan = tope
    hechos = 0

    # -- medicion ---------------------------------------------------------
    arranque = time.perf_counter()
    posicion = 0
    while posicion < largo:
        utiles = min(lote, largo - posicion)
        # El ultimo lote casi nunca es multiplo de la alineacion. En vez de
        # rellenarlo con ceros (que se compararian como iguales y falsearian
        # el conteo), se recorta a la alineacion y el resto se procesa en una
        # pasada final con menos capas.
        cuadra = (utiles // alineacion) * alineacion
        if cuadra == 0:
            break

        bloque_a = np.ascontiguousarray(
            va[posicion:posicion + cuadra]).reshape(1, 1, cuadra)
        bloque_b = np.ascontiguousarray(
            vb[posicion:posicion + cuadra]).reshape(1, 1, cuadra)

        entradas = {'A': bloque_a, 'B': bloque_b}

        # SE PIDE SOLO 'cuenta', NUNCA LAS DOS SALIDAS A LA VEZ.
        #
        # TRAMPA YA PISADA, y es de las caras. El grafo tiene dos salidas: el
        # conteo y la rama NonZero con las posiciones. Cuando un lote no
        # tiene NINGUNA diferencia, NonZero devuelve un tensor de tamano cero,
        # y DirectML corrompe entonces la OTRA salida: el conteo sale con
        # basura de memoria sin inicializar, distinta en cada ejecucion.
        #
        # Medido sobre los dos genomas: hay unos 130 lotes identicos, cada uno
        # aportaba un numero aleatorio de hasta 19 digitos, y el total salia
        # con veinte digitos, mayor que el numero de posiciones comparadas.
        # Se comprobo que el fallo no depende del tipo de la salida ni de
        # keepdims: quitar la rama NonZero lo arregla y volver a ponerla lo
        # reproduce.
        #
        # Al pedir un solo nombre de salida, ONNX Runtime poda el nodo que no
        # hace falta y el conteo es correcto siempre. De paso se ahorra
        # calcular NonZero en los cientos de lotes donde ya no se necesita
        # ninguna posicion mas.
        cuenta = sesion.run(['cuenta'], entradas)[0]
        # ReduceSum con keepdims=0 devuelve un escalar sin dimensiones, no un
        # vector de uno. Se aplana antes de leerlo para que de igual como lo
        # entregue cada proveedor: DirectML y la CPU no siempre coinciden en
        # la forma exacta del tensor de salida.
        aqui = int(np.asarray(cuenta).reshape(-1)[0])
        total += aqui

        # Las posiciones se piden en una segunda inferencia, y solo cuando
        # de verdad hacen falta: mientras queden huecos por llenar y este
        # lote tenga al menos una diferencia. La condicion 'aqui > 0' evita
        # ademas pedir NonZero sobre un lote identico, que es justo el caso
        # que dispara el fallo descrito arriba.
        if faltan and aqui:
            indices = sesion.run(['indices'], entradas)[0]
            if indices.size:
                # NonZero devuelve una fila por dimension; la tercera es la
                # que lleva la posicion dentro del lote.
                locales = indices[2][:faltan]
                recogidas.append(locales.astype(np.int64) + posicion)
                faltan -= locales.size

        posicion += cuadra
        hechos += cuadra
        if progreso is not None:
            progreso(hechos, largo)

    # Cola final: lo que no cuadraba con la alineacion de las capas.
    if posicion < largo:
        resto_a = np.asarray(va[posicion:largo])
        resto_b = np.asarray(vb[posicion:largo])
        distintos = resto_a != resto_b
        aqui = int(np.count_nonzero(distintos))
        total += aqui
        if faltan and aqui:
            locales = np.flatnonzero(distintos)[:faltan] + posicion
            recogidas.append(locales.astype(np.int64))
            faltan -= locales.size
        hechos = largo

    transcurrido = time.perf_counter() - arranque

    if progreso is not None:
        progreso(largo, largo)

    if recogidas:
        posiciones = np.concatenate(recogidas)[:tope]
    else:
        posiciones = np.empty(0, dtype=np.int64)

    resultado = comparador.resumir(total, posiciones, cadena_a, cadena_b,
                                   largo, 'NPU', detalle=detalle)
    resultado['npu'] = {
        'proveedor': nombre_proveedor,
        'proveedores_activos': usados,
        'es_npu': es_npu,
        'descripcion': descripcion,
        'capas': capas,
        'lote': lote,
        'preparacion_s': round(preparacion, 4),
        'modelo': os.path.basename(ruta_modelo(capas)),
        'runtime': getattr(ort, '__version__', None),
        'desde_vendor': os.path.isdir(_VENDOR),
    }
    avisos = [a for a in (aviso_capas, aviso_lote) if a]
    if not es_npu:
        avisos.append(
            'Este equipo no expone una NPU a traves de onnxruntime. El grafo '
            'se ejecuto sobre %s. En un equipo con NPU el programa la elige '
            'por si solo, siempre que este instalado el onnxruntime de su '
            'fabricante (el de vendor\\ solo trae DirectML).' % descripcion)
    if avisos:
        resultado['npu']['avisos'] = avisos
    return resultado, transcurrido


# ---------------------------------------------------------------------------
# Prueba rapida por linea de comandos
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    datos = info_npu()
    print('onnxruntime       : %s' % (datos.get('version_runtime') or 'NO'))
    print('Copia del proyecto: %s' % ('si' if datos['desde_vendor'] else 'no'))
    print('Proveedores       : %s' % ', '.join(datos['proveedores']))
    if datos.get('proveedor'):
        print('Elegido           : %s  (%s)'
              % (datos['proveedor'], datos['descripcion']))
    print('Es una NPU real   : %s' % ('SI' if datos.get('es_npu') else 'NO'))
    if datos.get('motivo'):
        print('')
        print('  %s' % datos['motivo'])

    if len(sys.argv) < 3:
        print('')
        print('Uso: python motor_npu.py <cadena_a.fna> <cadena_b.fna> '
              '[capas]')
        sys.exit(0)

    try:
        ca = comparador.preparar(sys.argv[1])
        cb = comparador.preparar(sys.argv[2])
        for aviso in comparador.validar_par(ca, cb):
            print('AVISO: %s' % aviso)
    except ErrorEntrada as error:
        print('ERROR: %s' % error)
        sys.exit(1)

    n_capas = int(sys.argv[3]) if len(sys.argv) > 3 else CAPAS
    mb = min(ca.largo, cb.largo) / (1024.0 * 1024.0)

    print('')
    print('Preparacion del grafo: %.3f s (fuera de la medicion)'
          % precalentar(n_capas))

    res, t = comparar_npu(ca, cb, capas=n_capas)
    print('')
    print('NPU %d capas : %8.3f s  (%8.1f MB/s)'
          % (res['npu']['capas'], t, mb / t if t else 0))
    print('  proveedor : %s' % res['npu']['proveedor'])
    print('  activos   : %s' % ', '.join(res['npu']['proveedores_activos']))
    for aviso in res['npu'].get('avisos', []):
        print('  AVISO: %s' % aviso)
    print('')
    print(comparador.formatear_diferencias(res))
