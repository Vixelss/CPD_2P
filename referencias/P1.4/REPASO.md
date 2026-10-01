# P1.4 — Repaso para defender el proyecto

Esto es para que tú lo leas antes de la defensa. Está escrito para que
entiendas qué hace el programa y por qué está hecho así, no para programar.

---

## 1. En una frase

El programa **compara dos cadenas de ADN y dice en qué posiciones son
distintas**, señalando la fila y la columna de cada diferencia, y lo hace de
tres maneras: con el procesador, con la tarjeta gráfica y con la NPU.

No cuenta bases. Eso era el P1.3. Aquí se comparan dos archivos.

---

## 2. Los cuatro proyectos, para no confundirte

| proyecto | qué hace | plataformas |
|---|---|---|
| **P1.1** | cuenta A, C, G, T | CPU secuencial vs paralela |
| **P1.2** | cuenta + detecta caracteres inválidos | CPU, GPU CUDA, OpenCL |
| **P1.3** | cuenta con las dos **a la vez** | CPU + GPU simultáneas |
| **P1.4** | **compara dos cadenas** | CPU, GPU, NPU |

Son **cuatro programas separados**. Si te pide ver el conteo, abres la
carpeta del P1.3. Si te pide la comparación, esta.

---

## 3. Qué ves en la pantalla

Abres con `python app_p14.py`.

**Arriba** — el hardware detectado, cada pieza de su color: azul el
procesador, verde la tarjeta, ámbar la NPU. En tu portátil la NPU sale en
gris porque no tienes.

**Columna izquierda** — eliges los dos archivos, la plataforma, los
parámetros, y abajo ves el resultado y ocho magnitudes de recursos que se
refrescan cada medio segundo.

**Cinco pestañas a la derecha:**

1. **Diferencias** — la tabla con posición, fila, columna y los dos
   caracteres que difieren
2. **Rendimiento** — cuatro gráficas de velocidad
3. **Núcleos y recursos** — cuatro gráficas de uso, temperatura y memoria.
   **Esta es la captura de "uso de recursos" que pide el profesor.**
4. **Evidencias** — la rúbrica rellenada sola
5. **Registro** — todo lo que hizo el programa, incluidos los ajustes que
   aplicó por su cuenta

---

## 4. Las librerías y quién las hizo

| librería | quién | para qué sirve aquí |
|---|---|---|
| **multiprocessing** | biblioteca estándar de Python (Jesse Noller, 2008) | lanza procesos de verdad; es la que responde a "librería que permite usar CPU" |
| **numpy** | Travis Oliphant, 2006 | la comparación en sí; corre en C, no en Python |
| **numba** (`numba.cuda`) | Anaconda Inc., 2012 | convierte funciones de Python en kernels CUDA; es la que permite usar la GPU |
| **onnxruntime** | Microsoft, 2018 | carga el grafo de red neuronal y lo manda al acelerador; es la que permite usar la NPU |
| **psutil** | Giampaolo Rodolà | núcleos, RAM, memoria de procesos |
| **pynvml** (nvidia-ml-py) | NVIDIA | uso, VRAM y temperatura de la tarjeta |
| **matplotlib** | John Hunter, 2003 | las ocho gráficas |
| **tkinter** | biblioteca estándar | la ventana |

Las tres que importan para la rúbrica son **multiprocessing**, **numba.cuda**
y **onnxruntime**: una por columna.

---

## 5. Cómo funciona cada plataforma

### CPU — trocear y repartir

El archivo se corta en *n* tramos y cada proceso compara el suyo. Lo único
que compara de verdad es una línea:

```python
distintos = va[inicio:fin] != vb[inicio:fin]
```

numpy recorre los dos tramos en C de una pasada.

**El detalle que te pueden preguntar:** los procesos hijos no reciben los
datos, reciben la *ruta* del archivo. Cada uno lo abre con `numpy.memmap`,
que no copia nada — el sistema operativo hace que las mismas páginas de
memoria aparezcan en los doce procesos. Si se les mandaran los datos, serían
79 GB de copias antes de empezar.

### GPU — un hilo por posición

Un kernel CUDA escrito a mano. Cada hilo compara un carácter. Si difieren,
suma 1 a un contador que vive en la **memoria compartida de su bloque**, y
solo al final el hilo 0 vuelca el total del bloque al contador global.

**Por qué no suma directo al contador global:** con mil millones de
diferencias serían mil millones de sumas atómicas sobre la misma dirección
de memoria. Todos los hilos se pondrían en fila y el kernel se serializaría.
Acumular barato y local, y pagar la sincronización cara una sola vez por
bloque.

Resultado medido: 24 registros por hilo, 8 bytes de memoria compartida por
bloque, **ocupación 48 de 48 warps por multiprocesador, o sea el 100 %**.

### NPU — la parte interesante

---

## 6. Cómo resolvimos lo de la NPU

**Este es el punto que más te van a preguntar.** Léelo despacio.

### El problema

Tu portátil **no tiene NPU**. El i5-13420H es de 13ª generación, y las NPU de
Intel empiezan en la serie Core Ultra. Lo comprobamos: no aparece ningún
dispositivo de NPU en el sistema ni existe el contador de rendimiento
correspondiente.

Y hay un problema de fondo más gordo: **una NPU no se programa como una GPU**.
No le escribes kernels, no le lanzas hilos, no ejecuta código cualquiera. Es
un acelerador de *inferencia*: le das un grafo de red neuronal y un runtime
lo reparte. Su repertorio de operaciones aceleradas es estrecho —
básicamente convoluciones y multiplicaciones de matrices sobre enteros.

Por eso el profesor dijo **"NPU n capas"** y no "n hilos": el paralelismo de
una NPU se expresa en capas del grafo.

### La solución: convertir la comparación en una red neuronal

Comparar dos caracteres es preguntar si son distintos. Escrito como grafo:

```
  capa 1   Cast       los bytes pasan a números decimales
  capa 2   Sub        se restan  →  0 si son iguales
  capa 3   Mul        al cuadrado  →  quita el signo
  capa 4   Greater    ¿mayor que 0.5?  →  sí/no
  capa 5   Cast       la MÁSCARA: 1 donde difieren, 0 donde coinciden

  capas 6..n  Conv    n convoluciones que suman de dos en dos
                      y reducen el vector a la mitad cada vez
  final    ReduceSum  el total de diferencias
```

**Por qué el umbral 0.5:** dos bytes distintos se diferencian al menos en 1,
así que su cuadrado vale al menos 1. El 0.5 separa sin ambigüedad.

**Por qué las convoluciones y no una suma directa:** si el grafo solo restara
y comparara, el runtime se lo devolvería entero a la CPU sin avisar, y
estarías diciendo "esto corre en la NPU" mientras corre en la CPU. Una
convolución **es** una multiplicación de matrices, que es exactamente para lo
que está construida una NPU.

**Por qué en árbol:** una NPU trabaja por capas y cada capa procesa todos sus
elementos en paralelo. Sumar tres millones de números de dos en dos son 22
capas, no tres millones de pasos.

### El argumento estrella: la cuantización INT8

El profesor insiste en la cuantización a enteros de 8 bits. Aquí tienes una
respuesta que vale oro:

> La máscara solo contiene ceros y unos, y los pesos de las convoluciones son
> unos. Al cuantizar ese grafo a INT8 **no se pierde absolutamente nada**: el
> resultado es idéntico bit a bit al de coma flotante, porque el dominio del
> dato ya era binario. En una red de visión la cuantización siempre degrada
> algo la precisión; aquí no, y se puede demostrar comparando contra la CPU.

### Cómo el programa encuentra la NPU sola

ONNX Runtime no habla directo con la NPU: lo hace a través de un *proveedor
de ejecución*, distinto por fabricante. El programa los prueba en orden:

| proveedor | hardware | ¿es NPU? |
|---|---|---|
| `QNNExecutionProvider` | Hexagon de Qualcomm (Snapdragon X) | sí |
| `OpenVINOExecutionProvider` | Intel AI Boost (Core Ultra) | sí |
| `VitisAIExecutionProvider` | XDNA de AMD (Ryzen AI) | sí |
| `DmlExecutionProvider` | DirectML, cualquier GPU DirectX 12 | no |
| `CPUExecutionProvider` | el procesador | no |

En tu máquina sale **DirectML**, y el programa lo dice claramente en pantalla
y en el informe: *"este equipo no expone una NPU; el grafo se ejecutó sobre
DirectML"*. No finge.

### Y para que corra en la laptop del profesor sin instalar nada

Dentro del proyecto hay una carpeta `vendor/` con una copia de onnxruntime
(68 MB). El programa la añade al camino de búsqueda de Python **antes** de
importar. Copias la carpeta a un pendrive, la llevas, y funciona.

**Aviso honesto:** la copia que llevamos incluye DirectML, que funciona en
cualquier equipo con DirectX 12, pero **no** incluye los proveedores de Intel
ni de Qualcomm. Si su laptop tiene una NPU Intel y quieres usarla de verdad,
habría que llevar también `onnxruntime-openvino`. Y si su portátil no tiene
Python instalado, nada arranca.

---

## 7. Los resultados y qué significan

### Comparando los dos genomas reales (3.3 GB cada uno)

```
CPU x1    7.329 s     429 MB/s
CPU x4    1.573 s    2000 MB/s   ← el mejor, speedup 4.66x
CPU x8    1.678 s    1875 MB/s
CPU x12   1.793 s    1754 MB/s
```

Las cuatro dan **exactamente 1.072.801.765 diferencias**.

### El hallazgo bonito

Los dos archivos son **GCA** (versión GenBank) y **GCF** (versión RefSeq) del
mismo genoma humano. La primera diferencia aparece en la posición
**1.945.577.269** — es decir, **los dos son idénticos durante los primeros
1.95 GB** y a partir de ahí divergen por completo (similitud final: 67.47 %).

No es un fallo. Las dos versiones ordenan de manera distinta los *scaffolds*
no ubicados, y en cuanto uno cambia de longitud, todo lo que viene después
queda desalineado.

**Eso es una lección de verdad y la puedes usar en la defensa:** la
comparación posición a posición solo vale mientras las dos secuencias están
alineadas. Resolver el desalineamiento requiere *alineamiento de secuencias*
(Needleman-Wunsch, Smith-Waterman), que es un problema de otro orden de
coste.

### Por qué el speedup se planta en 4x

Comparar dos bytes es **una instrucción**. Lo caro no es calcular, es traer
los bytes desde la memoria. Añadir procesos no añade ancho de banda de RAM,
solo más competidores por la que hay.

Es el **mismo análisis que ya defendiste en el P1.3**, pero con la RAM en
lugar del disco. Si te lo preguntan:

> El problema está limitado por el ancho de banda de memoria, no por cómputo.
> Por eso el paralelismo deja de compensar a partir de cuatro procesos, y por
> eso la GPU no le gana a la CPU: los datos cruzan el PCIe para volver casi
> sin tocarse.

---

## 8. Por qué el programa no se puede romper

Te pedí que fuera a prueba de tontos. Hay **121 comprobaciones automáticas**
en dos archivos:

```
python verificar.py          → 70 comprobaciones
python pruebas_interfaz.py   → 51 comprobaciones
```

Las categorías cubiertas:

| categoría | ejemplo |
|---|---|
| valores límite | 0 procesos, 1, 12, 13, el doble |
| tipos inválidos | `abc`, `3.5`, `12a`, `None` |
| negativos | −5, −999999 |
| campo vacío | `''`, `'   '` |
| desbordamiento | 1.000.000.000 |
| archivo inexistente | ruta borrada entre elegir y procesar |
| archivo equivocado | binario, carpeta, 0 bytes, texto que no es FASTA |
| mismo archivo dos veces | da 0 diferencias y avisa |
| cadenas de distinto largo | compara lo común y publica el desfase |
| orden de operaciones | Comparar sin archivos, Exportar sin resultados |
| doble clic | pulsar Comparar dos veces no lanza dos trabajos |
| cierre a mitad | pide confirmación |

**La regla que gobierna todo:** el programa o corrige el valor **y lo explica
en el registro**, o rechaza la entrada con un mensaje escrito para una
persona. Nunca se cae, y **nunca ajusta un parámetro en silencio**.

Ejemplo real: si pides 12 procesos con cadenas de 10 MB, usa 2 y escribe en
el registro *"con 12 procesos cada uno recibiría menos de 4 MB y arrancarlo
costaría más que el trabajo"*.

---

## 9. Detalles que demuestran que lo entiendes

Cosas que puedes soltar si te aprietan:

**Sobre los núcleos.** En la gráfica de uso por núcleo se ven **dos grupos**:
ocho núcleos al 88-99 % y cuatro al 39-53 %. No es un reparto malo — es que
el i5-13420H tiene núcleos de rendimiento y núcleos de eficiencia, y no
rinden igual.

**Sobre la temperatura de la NPU.** No existe sensor propio: la NPU va
integrada en el mismo chip que el procesador y comparte su zona térmica. El
programa reporta la del paquete y lo explica, en vez de dejar la casilla en
blanco.

**Sobre la temperatura por multiprocesador de GPU.** Tampoco existe. La
tarjeta tiene **un único sensor de die**. NVML no expone temperatura por SM
porque ese sensor no está físicamente.

**Sobre la memoria de la NPU.** No tiene memoria dedicada: usa la RAM del
sistema por un bus compartido, **sin transferencia por PCIe**. Es la
diferencia de fondo con la GPU y contesta directo al objetivo 2 del enunciado
("diferenciar los propósitos de CPU vs GPU vs NPU").

**Sobre por qué la preparación se mide aparte.** Arrancar doce procesos en
Windows cuesta medio segundo y compilar el kernel CUDA cuatro. Medimos que
con el arranque dentro del cronómetro, doce procesos salían **ochenta veces
más lentos** que uno. Eso no dice nada del paralelismo y todo del coste de
crear procesos.

---

## 10. Lo que falta, por si te lo preguntan

- Medir GPU y NPU sobre los genomas completos (solo se midió la CPU)
- Las capturas de pantalla para el informe con datos grandes
- El informe en Word
- Explicar cada pantalla por escrito

Los tres últimos son las tres líneas que el profesor dictó en clase:
*"capturar pantalla con el uso de recursos / capturar pantalla con el
programa / explicar cada pantalla"*.

El programa `capturas.py` ya saca las imágenes; solo hay que ejecutarlo con
los archivos grandes y escribir las explicaciones.

---

## 11. Comandos que necesitas saber

```bat
python app_p14.py                    abrir la aplicación
python verificar.py                  probar los motores
python pruebas_interfaz.py           probar la ventana
python capturas.py                   generar las capturas del informe

python generar_par.py --origen ..\P1.3\datos\recorte_50MB.fna --mb 10
                                     crear un par de prueba nuevo
```

Y los archivos grandes para la demo de verdad:

```
..\P1.1\GCA_000001405.29_GRCh38.p14_genomic.fna
..\P1.1\GCF_000001405.40_GRCh38.p14_genomic.fna
```

> **Nota sobre el espacio:** la carpeta `datos/` pesa unos 7.5 GB. Son
> archivos `.seq` con la secuencia ya limpia de los genomas — una caché que
> hace que la comparación arranque en milisegundos en vez de 17 segundos. Se
> pueden borrar cuando quieras y se regeneran solos.
