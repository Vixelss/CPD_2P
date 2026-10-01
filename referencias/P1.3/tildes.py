# -*- coding: utf-8 -*-
"""
tildes.py

Repone las tildes y las enes al texto del informe.

POR QUE EXISTE ESTE MODULO
Todo el codigo del proyecto se escribe sin tildes a proposito, porque los
comentarios y las cadenas conviven con rutas, consolas y editores que no
siempre tratan bien los acentos. Esa convencion vale para el codigo, pero no
para el documento que se entrega: un informe academico en castellano sin
tildes acumula cientos de faltas de ortografia, y las condiciones de entrega
del proyecto penalizan cada una de ellas.

La solucion es separar las dos cosas. Las cadenas del generador siguen
escribiendose sin tildes, como el resto del codigo, y este modulo las corrige
justo antes de escribirlas en el documento. Asi ni el codigo pierde su
convencion ni el informe sale con faltas.

Se sustituyen palabras completas, respetando la mayuscula inicial, y no se
tocan las que aparecen dentro de nombres de archivo, identificadores o
llamadas de codigo citadas en el texto.
"""

import re


# Palabras que cambian. Van en minuscula; la funcion respeta la mayuscula
# inicial de la aparicion. Se listan en singular y plural por separado porque
# el plural de muchas no lleva tilde: 'funcion' la lleva y 'funciones' no.
PALABRAS = {
    # -- terminaciones en -cion y -sion -------------------------------------
    'aplicacion': 'aplicación', 'calibracion': 'calibración',
    'colaboracion': 'colaboración', 'comparacion': 'comparación',
    'compilacion': 'compilación', 'comprobacion': 'comprobación',
    'computacion': 'computación', 'conclusion': 'conclusión',
    'configuracion': 'configuración', 'continuacion': 'continuación',
    'correccion': 'corrección', 'creacion': 'creación',
    'decision': 'decisión', 'desviacion': 'desviación',
    'deteccion': 'detección', 'dispersion': 'dispersión',
    'documentacion': 'documentación', 'ejecucion': 'ejecución',
    'estimacion': 'estimación', 'exposicion': 'exposición',
    'fraccion': 'fracción', 'funcion': 'función',
    'generacion': 'generación', 'implementacion': 'implementación',
    'importacion': 'importación', 'informacion': 'información',
    'instruccion': 'instrucción', 'instrumentacion': 'instrumentación',
    'interrupcion': 'interrupción', 'inyeccion': 'inyección',
    'iteracion': 'iteración', 'leccion': 'lección',
    'medicion': 'medición', 'ocupacion': 'ocupación',
    'operacion': 'operación', 'prediccion': 'predicción',
    'preparacion': 'preparación', 'progresion': 'progresión',
    'proporcion': 'proporción', 'racion': 'ración',
    'refrigeracion': 'refrigeración', 'representacion': 'representación',
    'seccion': 'sección', 'separacion': 'separación',
    'simulacion': 'simulación', 'situacion': 'situación',
    'solucion': 'solución', 'superposicion': 'superposición',
    'suposicion': 'suposición', 'validacion': 'validación',
    'verificacion': 'verificación', 'version': 'versión',
    'condiciono': 'condicionó', 'distorsionaria': 'distorsionaría',

    # -- esdrujulas y otras con tilde ---------------------------------------
    'analisis': 'análisis', 'atomica': 'atómica', 'atomicas': 'atómicas',
    'biologica': 'biológica', 'biologico': 'biológico',
    'algoritmicamente': 'algorítmicamente',
    'bioinformatica': 'bioinformática', 'informatica': 'informática',
    'caracteristica': 'característica',
    'caracteristicas': 'características',
    'electrica': 'eléctrica', 'electricas': 'eléctricas',
    'estandar': 'estándar', 'fisico': 'físico', 'fisicos': 'físicos',
    'grafica': 'gráfica', 'graficas': 'gráficas',
    'grafico': 'gráfico', 'graficos': 'gráficos',
    'linea': 'línea', 'lineas': 'líneas',
    'logica': 'lógica', 'logico': 'lógico', 'logicos': 'lógicos',
    'maquina': 'máquina', 'maquinas': 'máquinas',
    'maximo': 'máximo', 'maximos': 'máximos',
    'metodo': 'método', 'metodos': 'métodos',
    'metodologica': 'metodológica',
    'minimo': 'mínimo', 'minimos': 'mínimos',
    'multinucleo': 'multinúcleo',
    'nucleo': 'núcleo', 'nucleos': 'núcleos',
    'nucleotido': 'nucleótido', 'nucleotidos': 'nucleótidos',
    'numero': 'número', 'numeros': 'números',
    'parametro': 'parámetro', 'parametros': 'parámetros',
    'practica': 'práctica', 'practicamente': 'prácticamente',
    'practico': 'práctico', 'practicos': 'prácticos',
    'proposito': 'propósito', 'propositos': 'propósitos',
    'rapida': 'rápida', 'rapidas': 'rápidas',
    'rapido': 'rápido', 'rapidos': 'rápidos',
    'simbolo': 'símbolo', 'simbolos': 'símbolos',
    'tecnica': 'técnica', 'tecnicas': 'técnicas',
    'tecnico': 'técnico', 'tecnicos': 'técnicos',
    'teoria': 'teoría', 'ultima': 'última', 'ultimas': 'últimas',
    'ultimo': 'último', 'ultimos': 'últimos',
    'unica': 'única', 'unicamente': 'únicamente',
    'unico': 'único', 'unicos': 'únicos',
    'volumen': 'volumen', 'volumenes': 'volúmenes',

    # -- palabras con ene ---------------------------------------------------
    'acompanan': 'acompañan', 'acompana': 'acompaña',
    'anadir': 'añadir', 'anade': 'añade', 'anadiendo': 'añadiendo',
    'espanol': 'español', 'pequeno': 'pequeño', 'pequenos': 'pequeños',
    'pequena': 'pequeña', 'pequenas': 'pequeñas',
    'tamano': 'tamaño', 'tamanos': 'tamaños',
    'ano': 'año', 'anos': 'años', 'diseno': 'diseño',
    'disenada': 'diseñada', 'disenado': 'diseñado',
    'desempeno': 'desempeño',

    # -- monosilabos y adverbios --------------------------------------------
    'mas': 'más', 'asi': 'así', 'despues': 'después', 'segun': 'según',
    'tambien': 'también', 'ademas': 'además', 'aqui': 'aquí',
    'alli': 'allí', 'traves': 'través', 'dia': 'día', 'dias': 'días',
    'energia': 'energía', 'ingenieria': 'ingeniería',
    'catolica': 'católica',

    # -- segunda tanda, localizada revisando el documento ya generado -------
    'aparecia': 'aparecía', 'astrofisica': 'astrofísica', 'cabria': 'cabría',
    'caracter': 'carácter', 'centigrados': 'centígrados',
    'cientifico': 'científico', 'cientificos': 'científicos',
    'climatica': 'climática', 'codigo': 'código', 'codigos': 'códigos',
    'compartirian': 'compartirían', 'computo': 'cómputo',
    'corresponderia': 'correspondería', 'correspondia': 'correspondía',
    'cuestion': 'cuestión', 'decimas': 'décimas',
    'dejarian': 'dejarían', 'desaparecia': 'desaparecía',
    'devolvia': 'devolvía', 'dieciseis': 'dieciséis',
    'dinamica': 'dinámica', 'dinamico': 'dinámico', 'disenar': 'diseñar',
    'estaria': 'estaría', 'farmacos': 'fármacos', 'geneticas': 'genéticas',
    'genomica': 'genómica', 'habia': 'había',
    'heterogeneas': 'heterogéneas', 'heterogeneo': 'heterogéneo',
    'hibrida': 'híbrida', 'hibrido': 'híbrido', 'hipotesis': 'hipótesis',
    'homogeneas': 'homogéneas', 'identicas': 'idénticas',
    'identicos': 'idénticos', 'informaria': 'informaría',
    'libreria': 'librería', 'limite': 'límite', 'limites': 'límites',
    'liquida': 'líquida', 'mayoria': 'mayoría', 'median': 'medían',
    'modulo': 'módulo', 'modulos': 'módulos', 'molecula': 'molécula',
    'multiple': 'múltiple', 'multiples': 'múltiples', 'ningun': 'ningún',
    'ocultaria': 'ocultaría', 'pestana': 'pestaña', 'pestanas': 'pestañas',
    'periodicamente': 'periódicamente', 'podrian': 'podrían',
    'portatil': 'portátil', 'recaeria': 'recaería',
    'reclamaria': 'reclamaría', 'recien': 'recién', 'rendiria': 'rendiría',
    'resolveria': 'resolvería', 'seria': 'sería',
    'significaria': 'significaría', 'simultanea': 'simultánea',
    'simultaneas': 'simultáneas', 'simultaneo': 'simultáneo',
    'simultaneamente': 'simultáneamente', 'sistematica': 'sistemática',
    'situa': 'sitúa', 'solido': 'sólido', 'terminos': 'términos',
    'trasladandolo': 'trasladándolo', 'vacia': 'vacía', 'vacias': 'vacías',
    'volatil': 'volátil', 'zocalos': 'zócalos', 'area': 'área',
    'optimo': 'óptimo', 'optima': 'óptima', 'todavia': 'todavía',
    'explicito': 'explícito', 'explicita': 'explícita',
    'especifico': 'específico', 'especifica': 'específica',
    'imagenes': 'imágenes', 'margenes': 'márgenes',

    # -- preteritos, que sin tilde se leen como presente --------------------
    'alcanzo': 'alcanzó', 'arrojo': 'arrojó', 'atendio': 'atendió',
    'comprobo': 'comprobó', 'convirtio': 'convirtió', 'cumplio': 'cumplió',
    'detecto': 'detectó', 'distribuyo': 'distribuyó', 'ejecuto': 'ejecutó',
    'entrego': 'entregó', 'estructuro': 'estructuró',
    'implemento': 'implementó', 'llego': 'llegó', 'midio': 'midió',
    'mostro': 'mostró', 'ocupo': 'ocupó', 'permanecia': 'permanecía',
    'realizo': 'realizó', 'repitio': 'repitió', 'resulto': 'resultó',
    'tomo': 'tomó', 'verifico': 'verificó', 'situo': 'situó',
}

# Palabras interrogativas que solo llevan tilde en las preguntas. Se corrigen
# por separado porque en el resto de usos van sin ella.
INTERROGATIVAS = {
    'cuantos': 'cuántos', 'cuantas': 'cuántas',
    'cuanto': 'cuánto', 'cuanta': 'cuánta',
}

# Fragmentos que nunca se tocan: si una palabra aparece dentro de uno de
# estos, se deja tal cual. Son nombres de archivo, identificadores y llamadas
# de codigo que el informe cita literalmente y que deben seguir siendo
# copiables y ejecutables.
PROTEGIDOS = re.compile(
    r'\b(?:[A-Za-z_][A-Za-z0-9_]*\.(?:py|png|csv|json|txt|fna)'
    r'|[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_.]*'
    r'|[A-Za-z_][A-Za-z0-9_]*\([^)]*\)'
    r'|[A-Za-z_][A-Za-z0-9_]*\[[^\]]*\]'
    r'|@[A-Za-z_][A-Za-z0-9_.]*'
    r'|%[a-z]+)')


def _misma_caja(original, nuevo):
    """Devuelve 'nuevo' con la mayuscula inicial que tuviera 'original'."""
    if original[:1].isupper():
        return nuevo[:1].upper() + nuevo[1:]
    return nuevo


def tildar(texto, interrogativo=False):
    """Repone tildes y enes en un texto del informe.

    Con 'interrogativo' en True se acentuan ademas los interrogativos, que es
    lo que corresponde en titulos formulados como pregunta.
    """
    if not texto:
        return texto

    mapa = dict(PALABRAS)
    if interrogativo:
        mapa.update(INTERROGATIVAS)

    # Se parte el texto en tramos protegidos y tramos normales, y solo se
    # tocan los segundos.
    partes = []
    ultimo = 0
    for encontrado in PROTEGIDOS.finditer(texto):
        partes.append((texto[ultimo:encontrado.start()], True))
        partes.append((encontrado.group(0), False))
        ultimo = encontrado.end()
    partes.append((texto[ultimo:], True))

    def sustituir(tramo):
        def cambiar(m):
            palabra = m.group(0)
            nuevo = mapa.get(palabra.lower())
            return _misma_caja(palabra, nuevo) if nuevo else palabra
        return re.sub(r'\b[A-Za-z]+\b', cambiar, tramo)

    return ''.join(sustituir(t) if tocar else t for t, tocar in partes)


def pendientes(texto):
    """Lista las palabras del texto que probablemente aun necesiten tilde.

    Sirve de red de seguridad: despues de generar el documento se pasa por
    aqui para comprobar que no ha quedado ninguna sin corregir.
    """
    sospechosas = re.compile(
        r'\b\w*(?:cion|sion|logic|metod|grafic|numer|maxim|minim|rapid'
        r'|unic|ultim|practic|tecnic|atomic|simbol|parametr|proposit'
        r'|caracterist|analisis|estandar|tamano|nucleo|pequen|linea)\w*\b',
        re.I)
    salvadas = {'combinaciones', 'condiciones', 'conclusiones', 'divisiones',
                'especificaciones', 'instalaciones', 'instrucciones',
                'mediciones', 'operaciones', 'peticiones', 'posiciones',
                'raciones', 'recomendaciones', 'secciones', 'ejecuciones',
                'aplicaciones', 'configuraciones', 'convencionales',
                'computacional', 'computacionalmente', 'comunica',
                'condiciona', 'diccionario', 'dimensionados', 'funciona',
                'funcionales', 'maximizar', 'minimizar', 'numerados',
                'posicionarse', 'proporcional', 'lineal', 'lineales'}
    fuera = []
    for palabra in sospechosas.findall(texto):
        limpia = palabra.lower()
        if limpia in salvadas:
            continue
        if any(c in palabra for c in 'áéíóúñÁÉÍÓÚÑ'):
            continue
        fuera.append(palabra)
    return sorted(set(fuera))
