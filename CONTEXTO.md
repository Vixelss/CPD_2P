# CONTEXTO DEL PROYECTO: Clúster heterogéneo y distribuido para procesamiento de ADN (P2.3)

Este documento es la especificación completa del proyecto. Está escrito para que un asistente de código (Claude Code) lo construya **por etapas**, sin tener que adivinar decisiones. Léelo entero antes de escribir una sola línea de código. Si algo de este documento entra en conflicto con lo que encuentres en el código de referencia, **manda este documento**; si algo no está cubierto aquí, toma la decisión más simple que cumpla la rúbrica, anótala en `ESTADO.md` y sigue.

---

## 0. CÓMO TRABAJAR CON ESTE DOCUMENTO

### 0.1 Reglas de trabajo

1. **Trabaja por etapas, en el orden de la sección 17.** No empieces una etapa sin que la anterior tenga sus pruebas pasando.
2. **Cada etapa termina con un commit propio** (o una rama y un pull request), con un mensaje que diga qué etapa es: `E3: master y workers dinamicos`.
3. **Mantén un archivo `ESTADO.md` en la raíz** y actualízalo al final de cada etapa con: qué se hizo, qué pruebas pasan, qué quedó pendiente, qué decisiones tomaste por tu cuenta y por qué, y qué no pudiste probar en la nube y debe probarse en las laptops reales. Ese archivo es el traspaso hacia otra sesión de Claude Code que correrá en las máquinas reales, así que debe ser suficiente para retomar sin leer el historial.
4. **No pidas confirmación para cada paso.** Las decisiones grandes ya están tomadas aquí. Pregunta (o deja anotado en `ESTADO.md`) solo si algo es imposible tal como está escrito.
5. **Todo debe poder probarse en la nube sin hardware especial.** Ver sección 16: hay un clúster simulado local, un simulador de CUDA y alternativas en CPU para todo lo que dependa de GPU o NPU. Lo que solo se puede probar en hardware real se marca en `ESTADO.md` como "pendiente de prueba en hardware".
6. **Nunca subas datos de ADN al repositorio** (sección 1.3).

### 0.2 Idioma y convenciones

- Identificadores, comentarios, mensajes de la interfaz, logs y documentación **en español**.
- **Comentarios del código sin tildes** (convención heredada del Parcial 1: `# Calcula el histograma del tramo`, no `# Cálculo`). La documentación en `.md` sí lleva tildes.
- Python 3.12 (la versión de Ubuntu 24.04). Debe funcionar también en Python 3.12/3.13 de macOS.
- Anotaciones de tipo en funciones públicas. Docstrings cortos que digan qué hace, qué recibe y qué devuelve.
- Toda dependencia opcional (pynvml, numba-cuda, coremltools, pyopencl, mpi4py) se importa de forma perezosa y protegida: si falta, el programa sigue funcionando sin esa parte y lo informa con un mensaje claro.
- Cero rutas o IPs fijas en el código. Todo sale de `cluster.yaml` o de argumentos de línea de comandos.
- Cero APIs exclusivas de Windows. Todo el sistema corre en **Linux (Ubuntu 24.04)** y el worker de NPU en **macOS (Apple Silicon)**.

---

## 1. PRIMERA TAREA: REORGANIZAR EL REPOSITORIO

### 1.1 Qué hay ahora

El repositorio `CPD_2P` tiene hoy en la raíz cuatro carpetas: `P1.1`, `P1.2`, `P1.3`, `P1.4`, más un `README.md`. **Esas cuatro carpetas son el contexto:** son los cuatro proyectos del Parcial 1, ya terminados y validados, hechos en Windows. No son parte del producto nuevo; son referencia y fuente de código reutilizable.

### 1.2 Qué hacer

1. Crear la carpeta `referencias/` y mover ahí las cuatro carpetas **con `git mv`** para conservar el historial:
   ```bash
   mkdir referencias
   git mv P1.1 P1.2 P1.3 P1.4 referencias/
   ```
2. Cada carpeta P1.x puede traer una copia del PDF del enunciado del profesor (`COMPUTACION PDN - ... PROYECTOS 2026-04-08.pdf`). Dejar **una sola copia** como `referencias/enunciado.pdf` y borrar las demás.
3. Si alguna carpeta trae `vendor/`, `datos/`, `__pycache__/`, archivos `.fna`, `.seq`, `.idx`, `.origen`, `.zip`, o los archivos de datos `ManuSporny-genome_*.txt` y `H1N5_*.csv` del P1.1, **borrarlos del repositorio** (`git rm -r --cached` si estaban versionados). Son datos o binarios de Windows.
4. Crear `.gitignore` en la raíz con, al menos:
   ```
   # Datos de ADN: nunca al repositorio
   datos/
   pdn-datos/
   *.fna
   *.fa
   *.fasta
   *.seq
   *.idx
   *.origen
   *.huellas
   referencias/P1.1/ManuSporny-genome_*.txt
   referencias/P1.1/H1N5_*.csv

   # Binarios de Windows
   vendor/

   # Generados
   __pycache__/
   *.pyc
   .venv/
   pdn-env/
   *.so
   build/
   resultados/*
   !resultados/.gitkeep
   *.zip
   ```
5. Crear `.gitattributes` para normalizar los finales de línea (varios archivos del P1.4 tienen CRLF de Windows):
   ```
   * text=auto eol=lf
   *.pdf binary
   *.docx binary
   *.png binary
   *.onnx binary
   *.mlpackage binary
   ```
6. Commit: `E0: reorganiza el contexto en referencias/`.

### 1.3 Regla permanente sobre datos

Los genomas reales pesan unos 3,1 GB cada uno y GitHub rechaza archivos de más de 100 MB. **Ningún archivo de ADN real se versiona.** Las pruebas en la nube usan datos sintéticos generados por el propio proyecto (sección 16.2). Los datos reales viven fuera del repositorio, en `~/pdn-datos/` de cada laptop.

### 1.4 Cómo leer las referencias

Lee, en este orden: `referencias/P1.4/Contex.txt` (el más completo, 950 líneas, describe los cuatro proyectos), `referencias/P1.3/Contex.txt`, `referencias/P1.2/Contex.txt`, `referencias/P1.1/Contex.txt`, `referencias/P1.4/REPASO.md`, y las páginas del P2.1, P2.2 y P2.3 de `referencias/enunciado.pdf`. Los `Contex.txt` documentan decisiones de diseño y errores ya resueltos; la sección 4 de este documento resume lo que hay que heredar.

---

## 2. CONTEXTO ACADÉMICO

### 2.1 La materia

Computación Paralela, Distribuida y en la Nube (PDN), PUCE, Ing. JW Cóndor. El problema conductor de todo el semestre es procesar cadenas de ADN humano en formato FASTA (genoma de referencia GRCh38, unos 3,1 GB). En el Parcial 1 se resolvió dentro de una sola máquina (CPU, GPU, NPU). En el Parcial 2 se resuelve **repartiendo el trabajo entre varias máquinas**.

### 2.2 Qué piden los tres proyectos del Parcial 2

Este proyecto es el **P2.3, el integrador**, y debe cubrir también lo que piden el P2.1 y el P2.2, porque se presenta como un único sistema.

**P2.1, Clúster de computadoras.** Implementar un clúster con máquinas reales, verificar sistemas operativos distribuidos, para HPC y para alta disponibilidad (HA), revisar comandos de verificación de conexión entre nodos y la estructura de nodo maestro y nodos del clúster. Rúbrica (0, 3, 6, 10 por criterio): diseño gráfico del clúster; explicación de conectividad y direcciones IP; clúster funcionando; funcionamiento de uno, dos y de todo el clúster desde el nodo maestro; recomendaciones (1, 3 o 5). **Decisión del grupo: la parte de máquinas virtuales del P2.1 NO se hace.** Solo el clúster con máquinas reales.

**P2.2, Procesamiento distribuido.** Analizar una cadena de ADN repartiendo la carga entre los nodos. Clúster homogéneo con Linux, Python con Ray o mpi4py, un nodo maestro y tres o cuatro esclavos. Tareas: fragmentación del archivo, conteo de bases A, T, C, G, **búsqueda de patrones genéticos específicos**, reporte y consolidación de resultados. Actividades: dividir, diseñar la lógica distribuida, configurar el clúster, ejecutar y medir, **comparar contra el procesamiento secuencial**. Resultados esperados: reducción del tiempo, **validación de la integridad de los resultados entre nodos**, evaluación de eficiencia, escalabilidad y **tolerancia a fallos**. Recomendaciones del profesor: usar FASTA, aprovechar ciclos ociosos y **simular fallos**, usar speedup y eficiencia. Rúbrica (10 puntos cada uno): diseño de arquitectura del clúster; división y distribución del trabajo; implementación técnica funcional; resultados y validación; documentación y presentación.

**P2.3, Procesamiento distribuido optimizado ("Ecosistema Genómico Heterogéneo y Distribuido").** Un sistema que orqueste distintos tipos de hardware en una red de computadores personales, eligiendo dinámicamente el mejor recurso para cada tarea. Tres capas:
- **Orquestación (Master):** la laptop del líder divide el archivo de ADN, gestiona la red (NFS, SSH) y asigna bloques según la potencia de cada nodo.
- **Cómputo (Workers):** las laptops ofrecen CPU, GPU o NPU.
- **Análisis (Dashboard):** interfaz que muestra en tiempo real qué arquitectura procesa los datos y su eficiencia (tiempo contra energía).

Cronograma del enunciado: semana 1, CPU y red, comparación de ADN con el 100 % de los núcleos de todas las CPU mediante **MPI**, foco en **afinidad de procesos** y latencia de red; semana 2, GPU, el Master detecta qué nodos tienen GPU y les envía los fragmentos más pesados (CUDA u OpenCL), foco en **solapar transferencia PCIe y cálculo**; semana 3, NPU, modelos de clasificación o búsqueda de motivos para detectar "zonas de interés", foco en **cuantización INT8** y eficiencia energética (TOPS por vatio).

Rúbrica del P2.3 (20 puntos cada una, total 100):
| Categoría | Criterio |
|---|---|
| Infraestructura (red) | Estabilidad del clúster bajo carga máxima y configuración correcta del sistema de archivos distribuido (NFS) |
| Optimización CPU | Uso correcto de instrucciones vectoriales (SIMD) y control total de la afinidad de núcleos |
| Eficiencia GPU/NPU | Transferir datos a la VRAM sin bloquear la CPU y uso de modelos cuantizados en la NPU |
| Balanceo de carga | El Master asigna más trabajo al nodo más rápido, evitando que el sistema se frene por el más lento |
| Informe técnico | Qué arquitectura fue más rentable (tiempo, costo, energía), con Ley de Amdahl |

Checklist del enunciado: Ubuntu o Debian, OpenMPI, GCC con soporte AVX, CUDA Toolkit, OpenVINO o TensorRT. (El grupo usa Core ML en la Mac para la NPU; ver sección 10.)

**Examen.** Se entrega un archivo de ADN de 10 GB. Gana el grupo que mejor balancee la carga, logrando que un i3 trabaje en armonía con un i9 con GPU, todos al 100 % de su capacidad. **El grupo no tiene un archivo de 10 GB: se trabajará con los genomas de 3,1 GB.** El sistema debe funcionar con cualquier tamaño porque todo va por bloques.

### 2.3 Decisiones ya tomadas por el grupo

- **El programa debe hacer tres operaciones**: (1) conteo y validación, (2) búsqueda de patrones, (3) comparación de dos cadenas. Además, la operación de "zonas de interés" para la NPU (sección 8.4).
- **No se hacen máquinas virtuales.**
- **Doble arranque con Ubuntu 24.04 LTS** en las cinco laptops Windows. La Mac se queda con macOS.
- **El Master es la laptop de Alex.** El Master de respaldo es la de Daniel.
- **El Wi-Fi está permitido** (el profesor lo autorizó), aunque se preferirá cable.
- **Cada nodo tiene su propia copia local del ADN** como modo por defecto; leer por NFS es un modo alternativo para comparar (sección 5.4).

### 2.4 Condiciones de entrega que afectan al software

- El informe va en PDF por la plataforma EVA. Hay que **contrastar la solución propia con la de una IA**, identificar las variaciones y argumentarlas en la defensa. Crea y mantén `docs/decisiones.md` donde cada decisión de diseño no obvia quede anotada con su motivo y la alternativa descartada. Eso alimenta esa sección del informe.
- **Nunca mostrar valores de ejemplo** en paneles que se capturan para el informe. Si no hay dato, se muestra "sin dato" con el motivo.

---

## 3. EL EQUIPO Y EL HARDWARE

### 3.1 Los seis nodos

| Integrante | Equipo | Hostname | Sistema | CPU | Hilos | RAM | Aceleradores | Rol |
|---|---|---|---|---|---|---|---|---|
| Alex Vivanco | HP Victus 15 | `nodo-vivanco` | Ubuntu 24.04 (doble arranque) | Intel i5-13420H (4 P + 4 E) | 12 | 32 GB | NVIDIA RTX 4050 Laptop 6 GB (CC 8.9, 20 SMs) + Intel UHD integrada | **Master**, servidor NFS, worker CPU y worker GPU principal |
| Daniel Carranza | (laptop) | `nodo-carranza` | Ubuntu 24.04 | AMD Ryzen 7 7730U | 16 | 16 GB | Radeon integrada (sin CUDA) | Worker CPU y **Master de respaldo** |
| David Ocampo | (laptop) | `nodo-ocampo` | Ubuntu 24.04 | AMD Ryzen 7 5700U | 16 | por detectar | Radeon integrada (sin CUDA) | Worker CPU |
| Gabriel Naranjo | (laptop) | `nodo-naranjo` | Ubuntu 24.04 | Intel i7-1165G7 | 8 | por detectar | NVIDIA MX450 2 GB + Intel Iris Xe | Worker CPU y worker GPU secundario |
| Wendy Palomo | (laptop) | `nodo-palomo` | Ubuntu 24.04 | Intel i7-1065G7 | 8 | 12 GB | Intel Iris Plus | Worker CPU. Su tarjeta de red puede ser Fast Ethernet (100 Mbps) |
| David Hidalgo | MacBook Air M5 | `nodo-hidalgo` | macOS (Apple Silicon, ARM64) | Apple M5 | por detectar | 16 GB | Neural Engine de 16 núcleos, GPU Apple | **Worker NPU**. No participa en MPI. Sin ventilador: baja su velocidad bajo carga sostenida |

Todos los datos "por detectar" los debe averiguar el propio programa al arrancar (sección 9.2). **No confíes en esta tabla para la lógica**; úsala solo como referencia y para valores por defecto de `cluster.yaml`.

### 3.2 Software base ya instalado en los nodos Linux

- Usuario `pdn` en todos los nodos Linux (mismo nombre, mismo `/home/pdn`).
- Paquetes de apt: `openssh-server openmpi-bin libopenmpi-dev python3-mpi4py python3-venv python3-pip nfs-common build-essential htop`. En el Master además `nfs-kernel-server`.
- Entorno virtual en `~/pdn-env`, creado con `python3 -m venv --system-site-packages ~/pdn-env` para que vea el `mpi4py` de apt, enlazado al OpenMPI del sistema. **No instalar mpi4py con pip en los nodos.**
- En los nodos con NVIDIA: `nvidia-ml-py` y `numba-cuda[cu12]` dentro del entorno. El driver ya funciona (`nvidia-smi`).
- SSH activo en todos. El Master tiene una llave `~/.ssh/id_ed25519` que se copiará a los workers.
- Firewall `ufw` inactivo.
- Carpeta `/cluster` creada en el Master, propiedad de `pdn`, que se exportará por NFS.
- En la Mac: Homebrew, Python de Homebrew, entorno `~/pdn-env`, SSH activo. El usuario de la Mac **no** se llama `pdn`: es configurable en `cluster.yaml`.

### 3.3 Datos reales disponibles

- `GCF_000001405.40_GRCh38.p14_genomic.fna` (RefSeq), 3,11 GB, 705 registros.
- `GCA_000001405.29_GRCh38.p14_genomic.fna` (GenBank), 3,19 GB, 709 registros.
- Viven en `~/pdn-datos/` de cada laptop. Nunca en el repositorio.

---

## 4. LECCIONES DEL PARCIAL 1 QUE ESTE PROYECTO DEBE RESPETAR

Estas reglas salen de errores reales ya cometidos y resueltos. No se negocian.

1. **Un mismo algoritmo en todas las plataformas.** El conteo se hace con un histograma de 256 casillas (una por valor de byte) en CPU, GPU y cualquier otro motor, y de ahí se derivan todos los conteos. Si una plataforma usa un atajo distinto, los tiempos no son comparables y el balanceo reparte trabajos que no cuestan lo mismo.
2. **Las costuras entre bloques son la fuente número uno de errores.** En el P1.3, la GPU duplicaba una línea cuando un tramo terminaba justo en un salto de línea; fallaba una de cada cinco o seis corridas. Consecuencia para este proyecto: las pruebas de correctitud deben correr **muchas configuraciones distintas de tamaño de bloque y de número de workers** para que las costuras caigan en sitios distintos. Una sola corrida correcta no prueba nada. En este proyecto el riesgo es mayor porque hay costuras entre bloques y entre nodos.
3. **La preparación queda fuera del cronómetro.** Compilar kernels (Numba tarda 0,5 a 3 s), crear pools de procesos, cargar modelos y calibrar se hace antes de arrancar el reloj y se publica aparte como `preparacion_s`. Si el arranque cae dentro de la medición, el que arranca primero se lleva los primeros bloques y el reparto deja de reflejar el rendimiento real.
4. **Reparto por raciones adaptativas, no por porcentajes fijos.** El P1.3 ya resolvió el balanceo entre CPU y GPU con raciones cuyo tamaño ajusta cada trabajador según su velocidad recién medida, apuntando a un tiempo objetivo por ración. Se observó que si todos piden al mismo tiempo en el instante cero, se reparten trozos antes de que nadie demuestre nada; por eso todos empiezan con una ración mínima. Ver `referencias/P1.3/motor_hibrido.py`. Este proyecto lleva esa idea a varias máquinas (sección 7.3).
5. **El problema está limitado por la entrada y salida de datos, no por el cálculo.** En el P1.3, CPU y GPU juntas fueron **más lentas que la GPU sola** (GPU sola 2161 MB/s; híbrido con 12 procesos 718 MB/s), porque los procesos de CPU competían por el disco y le quitaban turno al hilo que alimentaba a la GPU. Consecuencias: (a) cuando un nodo tiene worker de GPU, el worker de CPU de ese mismo nodo debe reservar al menos un núcleo para alimentar a la GPU (configurable); (b) los resultados que muestren que agregar recursos no ayuda **no se esconden**: se miden y se explican.
6. **Medición justa.** La cache de páginas del sistema cambia mucho los tiempos: en el P1.4 la CPU pasaba de 429 MB/s a 2000 MB/s sobre los mismos datos solo por correr en segundo lugar. Toda medición publicada debe tener una opción de calentamiento descartado y repeticiones con mediana.
7. **Recursos no medidos quedan vacíos, nunca en cero.** Si no hay al menos dos muestras de un recurso en un intervalo, el valor es `null` y se muestra "sin dato". Un cero se leería como "no se usó".
8. **Un solo muestreador de `psutil.cpu_percent` por proceso.** Con `interval=None`, el porcentaje se calcula contra la última llamada y esa referencia es global al proceso; dos muestreadores se la corrompen.
9. **Reglas del formato FASTA** (sección 6.1): las cabeceras se ignoran; minúsculas cuentan igual que mayúsculas; N es base desconocida y se cuenta aparte; los códigos IUPAC (R Y S W K M B D H V) son válidos y se cuentan aparte; cualquier otro carácter en una línea de secuencia es inválido.
10. **Hallazgo GenBank contra RefSeq.** Los dos genomas son el mismo ensamblaje con los registros en distinto orden: GenBank intercala los contigs no ubicados de cada cromosoma tras el cromosoma; RefSeq los agrupa aparte. Comparando posición por posición salen 1.072.801.765 diferencias falsas desde la posición 1.945.577.269 (registro 30, un contig de 100.316 bases). Emparejando registro por registro salen **cero diferencias**. La comparación debe emparejar por secuencia por defecto (sección 8.3).
11. **Hallazgo DirectML.** Con ONNX Runtime y DirectML, cuando un grafo tiene dos salidas y una es un tensor vacío, la otra sale corrompida. Lección general: **pedir al runtime solo las salidas que hacen falta** y validar siempre que los conteos sean coherentes (por ejemplo, diferencias nunca mayores que posiciones comparadas).
12. **El modelo de NPU del P1.4 está en float32, no cuantizado.** En este proyecto la cuantización INT8 sí es requisito de la rúbrica (sección 10).
13. **Mayúsculas y minúsculas en la comparación.** En FASTA la minúscula marca región repetitiva (soft-masking), pero es la misma base. En un par de prueba de 200 MB con 20.000 diferencias, 1.693 eran solo de caso. Este proyecto debe reportar esas diferencias **por separado** (sección 8.3).
14. **Código de Windows que no se porta:** la lectura de temperatura con PowerShell de `monitor.py`, las llamadas a `ctypes.windll` de `capturas.py` e `interfaz.py`, y la carpeta `vendor/` de ONNX Runtime. En Linux la temperatura se lee con `psutil.sensors_temperatures()`.

---

## 5. ARQUITECTURA GENERAL

### 5.1 Vista de conjunto

```
                        ┌──────────────────────────────────────┐
                        │   nodo-vivanco (Master)              │
                        │   - Planificador y fila de tareas    │
                        │   - Servidor ZeroMQ (tareas/latidos) │
                        │   - Dashboard web (FastAPI)          │
                        │   - Servidor NFS: /cluster           │
                        │   - Worker CPU + Worker GPU (RTX)    │
                        └───────────────┬──────────────────────┘
                                        │  red local (switch o router)
        ┌──────────────┬────────────────┼────────────────┬───────────────┐
        │              │                │                │               │
  nodo-carranza   nodo-ocampo     nodo-naranjo      nodo-palomo     nodo-hidalgo
  Worker CPU      Worker CPU      Worker CPU        Worker CPU      Worker NPU
  Master respaldo                 Worker GPU (MX450)                (Core ML, macOS)
```

### 5.2 Componentes

- **Master** (`pdn/master/`): mantiene la fila de tareas, asigna bloques, recibe resultados, detecta caídas, consolida, valida, persiste resultados y sirve el dashboard.
- **Master de respaldo** (`pdn/master/respaldo.py`): recibe copias periódicas del estado y asume el rol si el Master principal cae (sección 7.7).
- **Worker** (`pdn/worker/`): un proceso por dispositivo de cómputo en cada laptop (worker CPU, worker GPU, worker NPU). Detecta hardware, se registra, se calibra, pide tareas, las procesa con su motor y devuelve resultados y métricas.
- **Motores** (`pdn/motores/`): el cálculo puro sobre un tramo de bytes: CPU (numpy y núcleo C con AVX2), GPU (Numba CUDA), NPU (Core ML), OpenCL (opcional).
- **Operaciones** (`pdn/operaciones/`): la definición exacta de cada operación y su implementación de referencia en un solo hilo. Todo motor se valida contra esto.
- **Modo MPI** (`pdn/mpi/`): ejecución con `mpirun` y reparto estático, para la semana 1 del enunciado, para la referencia secuencial y para comparar contra el modo dinámico.
- **Monitoreo** (`pdn/monitoreo/`): recursos y energía en cada nodo.
- **Dashboard** (`pdn/dashboard/`): interfaz web.
- **Scripts** (`scripts/`): despliegue y operación del clúster.

### 5.3 Dos modos de ejecución

| | Modo MPI (estático) | Modo dinámico (principal) |
|---|---|---|
| Lanzamiento | `mpirun --hostfile ...` desde el Master | Workers persistentes conectados al Master |
| Reparto | Fijo: partes iguales, o proporcionales a velocidades calibradas | Raciones adaptativas desde una fila central |
| Nodos | Solo los Linux (CPU) | Todos, incluida la Mac |
| Dispositivos | CPU | CPU, GPU, NPU, OpenCL opcional |
| Tolerancia a fallos | No: si cae un proceso, MPI aborta el trabajo | Sí: reasignación de tareas y Master de respaldo |
| Para qué sirve | Semana 1 del enunciado, referencia secuencial, línea base | Balanceo heterogéneo, examen, demostración de fallos |

Comparar ambos modos con los mismos datos es material central del informe.

### 5.4 Ubicación de los datos

- **Modo local (por defecto):** cada nodo tiene su copia en `~/pdn-datos/` y lee de su propio disco. Por la red solo viajan instrucciones y resultados (kilobytes). Es la idea de "llevar el cálculo a los datos" de Big Data, y es indispensable para el Master de respaldo, porque los workers no dependen del disco del Master.
- **Modo NFS:** los workers leen el archivo desde `/cluster/datos/` montado por NFS. Cumple literalmente la descripción del enunciado ("el Master gestiona la red con NFS"), pero la red Gigabit (unos 110 MB/s compartidos) se vuelve cuello de botella.
- El dashboard permite elegir el modo, y el sistema registra cuál se usó en cada corrida. Medir ambos es parte del informe.
- **NFS se usa siempre** para: configuración compartida, resultados consolidados y logs de las corridas (en `/cluster/resultados/`). Si NFS falla, los workers deben seguir funcionando y escribir sus logs localmente.

### 5.5 Comunicación

- **ZeroMQ** (`pyzmq`) para el modo dinámico: socket `ROUTER` en el Master, `DEALER` en cada worker. Mensajes en JSON (sección 7.2). Puerto por defecto 5555 (tareas y latidos) y 5556 (replicación al respaldo), configurables.
- **HTTP y WebSocket** para el dashboard (FastAPI + uvicorn), puerto 8000.
- **SSH** para lanzar y detener procesos remotos desde el Master.
- **MPI** solo en el modo MPI.

---

## 6. DATOS: FORMATO, PREPARACIÓN E ÍNDICES

### 6.1 Reglas del formato FASTA

- Cabecera: línea completa que empieza con `>`. Se ignora en el procesamiento, pero se usa para el índice.
- Líneas de secuencia de 80 caracteres en los archivos reales; el código no debe asumirlo para leer, solo para reportar fila y columna (constante `ANCHO_LINEA = 80`, configurable).
- Finales de línea `\n` o `\r\n`: ambos se eliminan en la preparación.
- Clasificación de cada byte de secuencia:
  - **Bases:** `A C G T a c g t` (minúscula cuenta como su mayúscula).
  - **Desconocida:** `N n`.
  - **IUPAC:** `R Y S W K M B D H V` y sus minúsculas.
  - **Inválido:** cualquier otro byte (dígitos, signos, espacios, letras fuera de la lista, bytes no ASCII).

### 6.2 Preparación: de `.fna` a `.seq`

Todas las operaciones trabajan sobre la **secuencia limpia** (`.seq`): todos los bytes de secuencia del archivo, sin cabeceras ni saltos de línea, pegados uno tras otro. En el `.seq`, la posición de un byte es su desplazamiento, lo que hace trivial repartir por rangos de bytes y elimina la mayoría de los problemas de costuras con cabeceras.

Reutiliza `referencias/P1.4/comparador.py` (`preparar`, `huella_origen`, `cache_al_dia`, `validar_fasta`) y adáptalo:
- La preparación lee el `.fna` por bloques de 8 MB alineados a línea; nunca carga el archivo entero.
- Escribe `nombre.seq` y, en la misma pasada o en otra, `nombre.idx` (sección 6.3).
- Guarda junto a ellos un `.origen` con la huella del archivo original (tamaño, fecha y un hash de los primeros y últimos MB). Si la huella no coincide, se regenera. Esto existe porque en el P1.4 una cache vieja hizo que el índice de un par de prueba quedara vacío sin aviso.
- La preparación es **local a cada nodo** y queda fuera del cronómetro de cualquier corrida.

### 6.3 Índice de registros (`.idx`)

Reutiliza `referencias/P1.4/secuencias.py` (`indexar`, `Registro`, `emparejar`). Formato JSON con, por cada registro: identificador, descripción, cromosoma detectado ("chromosome 12", "mitochondrion"), desplazamiento de inicio en el `.seq` y longitud. Permite traducir una posición global del `.seq` a (registro, posición dentro del registro, fila, columna), con búsqueda binaria sobre los inicios.

**Fila y columna se reportan dentro del registro**, que es lo que corresponde a una línea real del archivo FASTA (línea N después de la cabecera de ese registro). Nunca reportar fila global sobre el `.seq` como si fuera una línea del archivo.

### 6.4 Huellas de integridad (`.huellas`)

Durante la preparación, calcular un CRC32 (`zlib.crc32`) por cada **unidad** de 4 MB del `.seq` (la última puede ser menor) y guardarlo en `nombre.huellas` (binario o JSON). Las tareas siempre cubren unidades completas (sección 7.3), así que el worker puede devolver el CRC de cada unidad que procesó y el Master compararlo con su tabla. Esto detecta copias locales dañadas o distintas entre nodos.

### 6.5 Distribución de datos a los nodos

`scripts/distribuir_datos.sh`: copia con `rsync -P` el `.fna` desde el Master a `~/pdn-datos/` de cada nodo, luego ejecuta la preparación remota por SSH y verifica que el `.huellas` del nodo sea idéntico al del Master. Informa nodo por nodo.

---

## 7. MODO DINÁMICO: MASTER, PROTOCOLO Y PLANIFICADOR

### 7.1 Conceptos

- **Corrida:** una ejecución completa de una operación sobre uno o dos archivos, con una configuración. Tiene un `corrida_id`.
- **Tarea:** un tramo del `.seq` (una o varias unidades contiguas de 4 MB) que se asigna a un worker. Tiene un `tarea_id` único dentro de la corrida.
- **Worker:** un proceso en una laptop que procesa tareas con un dispositivo. Una laptop puede tener varios workers (CPU, GPU, NPU). Identificador: `hostname:dispositivo`, por ejemplo `nodo-vivanco:gpu0`.

### 7.2 Protocolo de mensajes (JSON sobre ZeroMQ)

Todo mensaje lleva `tipo`, `worker_id`, `corrida_id` (cuando aplique) y `t` (marca de tiempo del emisor).

Worker a Master:
- `REGISTRO`: hardware detectado (sección 9.2), dispositivos, versión del programa, archivos locales disponibles con su tamaño y huella global.
- `CALIBRACION`: MB/s medidos por operación en el bloque de calibración.
- `PEDIR`: pide trabajo.
- `RESULTADO`: `tarea_id`, resultado parcial (según operación), CRC por unidad, bytes procesados, tiempos (`t_inicio`, `t_fin`, `t_calculo`, `t_transferencia` si aplica), dispositivo usado.
- `LATIDO`: cada 1 s, con métricas de recursos (sección 11).
- `ERROR`: tarea que falló, con el motivo.

Master a Worker:
- `ACEPTADO`: confirma el registro y entrega la configuración de la corrida.
- `TAREA`: `tarea_id`, operación, archivo(s), `inicio`, `fin`, `solape` (bytes extra para patrones), parámetros (patrones, motor, procesos, núcleos, bloques de GPU, lote).
- `ESPERAR`: no hay trabajo por ahora.
- `FIN`: terminó la corrida.
- `DETENER`: apagarse ordenadamente.
- `SIMULAR_FALLO`: `modo` = `caida` (el proceso termina con `os._exit(1)` sin avisar) o `congelado` (deja de responder y de enviar latidos, pero el proceso sigue vivo).

### 7.3 Planificador: raciones adaptativas

Adapta la lógica de `referencias/P1.3/motor_hibrido.py` a nivel de clúster:
1. El `.seq` se divide en unidades de 4 MB. La fila contiene rangos de unidades pendientes.
2. Cada worker tiene una velocidad estimada (MB/s por operación), inicializada con su calibración y actualizada con una media móvil exponencial de sus resultados.
3. El tamaño de cada tarea es el que el worker procesaría en un **tiempo objetivo** (por defecto 0,5 s; configurable de 0,1 a 5 s), redondeado a unidades completas, con mínimo de una unidad y máximo configurable (por defecto 128 MB, y 512 MB para workers GPU).
4. **Fase final decreciente:** cuando lo pendiente es menor que (suma de velocidades × tiempo objetivo × 2), el tamaño de tarea se limita a `pendiente / (2 × número de workers activos)`, para que todos terminen casi a la vez.
5. **Afinidad de capacidades:** las tareas de "zonas de interés" van preferentemente a workers NPU; si no hay, a GPU; si no, a CPU. Las demás operaciones van a cualquier worker.
6. Estrategias seleccionables en el dashboard, para comparar en el informe: `adaptativa` (por defecto), `fija` (tamaño constante), `proporcional` (reparto inicial único según calibración, sin fila).
7. Opcional (etapa 11): **tareas especulativas** al final, como en MapReduce: si la fila se vació y una tarea lleva más de 2 veces lo esperado, se duplica en el worker libre más rápido y vale el primer resultado.

### 7.4 Registro de tareas y reasignación

El Master mantiene, por cada tarea, su estado: `PENDIENTE`, `ASIGNADA(worker, t_asignacion, t_esperado)`, `HECHA`.
- Un worker se considera **perdido** si pasan 5 s sin latido (configurable). Todas sus tareas `ASIGNADA` vuelven a `PENDIENTE`.
- Una tarea se considera **vencida** si pasa `max(3 × t_esperado, 10 s)` sin resultado; vuelve a `PENDIENTE`.
- **Idempotencia:** si llega un resultado de una tarea que ya está `HECHA` (por ejemplo, de un worker que "revivió"), se descarta y se registra.
- Cada reasignación queda registrada con motivo y se muestra en el dashboard.
- El trabajo a medias de una tarea perdida se pierde completo; por eso las tareas no deben ser enormes.

### 7.5 Validación de integridad

1. **Por unidad:** el CRC32 de cada unidad devuelto por el worker se compara con la tabla `.huellas` del Master. Si no coincide, el resultado de esa tarea se rechaza, la tarea vuelve a la fila, y el nodo queda marcado con "copia de datos sospechosa"; tras dos rechazos, deja de recibir tareas de ese archivo. La verificación de CRC tiene costo (leer los bytes una vez más); debe poder desactivarse en el dashboard y su costo debe medirse.
2. **Global:** al terminar, la suma de bytes procesados debe ser igual al tamaño del `.seq`, y no puede haber huecos ni solapes de unidades.
3. **Contra la referencia secuencial:** si existe un resultado de referencia guardado para ese archivo y esa operación (sección 13.4), el resultado distribuido debe coincidir **exactamente**. El dashboard muestra "coincide" o la lista de discrepancias.

### 7.6 Consolidación y resultados

- Conteo: suma de histogramas (reutiliza `sumar_histogramas` de `referencias/P1.3/motor_cpu.py`).
- Patrones: suma de conteos por patrón y unión de posiciones (con tope configurable, por defecto 1000 por patrón), ordenadas.
- Comparación: suma de diferencias por categoría y unión de posiciones (con tope).
- Zonas: unión de ventanas clasificadas.
- Todo se guarda en `/cluster/resultados/<fecha_hora>_<operacion>/`: `resumen.json`, `tareas.csv` (una fila por tarea: worker, dispositivo, inicio, fin, bytes, MB/s, reasignada sí o no, motivo), `recursos.csv`, `energia.csv`, `config.json` y las gráficas.

### 7.7 Alta disponibilidad: Master de respaldo

- El Master principal envía al respaldo (`nodo-carranza`) cada 1 s una instantánea del estado: configuración de la corrida, registro de tareas, resultados parciales consolidados y lista de workers.
- El respaldo vigila esas instantáneas como latido. Si pasan 3 s sin recibirlas, **se promueve a Master**: abre los puertos, marca como `PENDIENTE` todas las tareas `ASIGNADA` de la última instantánea y continúa la corrida.
- Cada worker conoce la lista ordenada de Masters (`[principal, respaldo]`). Si el Master actual no responde durante 3 s, se conecta al siguiente y se vuelve a registrar.
- El dashboard también lo sirve el Master activo; la URL cambia a la IP del respaldo. El dashboard del respaldo debe mostrar claramente "Master de respaldo activo desde HH:MM:SS".
- Esto funciona porque los datos están en copia local en cada nodo (sección 5.4). En modo NFS, la caída del Master es fatal y el dashboard lo advierte antes de iniciar.
- Limitación a documentar: el modo MPI no tiene alta disponibilidad.
- Esta función cubre el objetivo de "sistema operativo para alta disponibilidad" del P2.1.

---

## 8. OPERACIONES: DEFINICIÓN EXACTA

Cada operación tiene una **implementación de referencia** en `pdn/operaciones/` que procesa un tramo de bytes (`numpy.ndarray` de `uint8`) en un solo hilo, de forma simple y obviamente correcta. Todos los motores se validan contra ella.

### 8.1 Conteo y validación

- Entrada: tramo del `.seq`.
- Cálculo: histograma de 256 casillas (`numpy.bincount(datos, minlength=256)` en la referencia).
- Salida derivada: conteo de A, C, G, T (sumando mayúscula y minúscula), N, IUPAC por letra, total de inválidos y el desglose de inválidos por byte (con su representación legible: `'5'`, `' '`, `0xC3`). Además, las primeras K posiciones de inválidos (con registro, fila y columna) si se pide.
- Valores de aceptación sobre `GCF_000001405.40` completo (medidos en el P1.2, deben coincidir exactamente): A = 923.117.203, C = 642.552.917, G = 645.231.996, T = 925.917.038, N = 161.611.379, IUPAC = 103 (Y = 36, R = 29, W = 15, K = 8, M = 8, S = 5, B = 2), inválidos = 0. Total de bytes del `.seq` = 3.298.430.636.
- Para demostrar la detección de inválidos se usan archivos "ensuciados" con errores en cantidad y tipo conocidos: adapta `referencias/P1.2/ensuciar.py` (genera variantes a partir de recortes del genoma real con un `.esperado.txt`).

### 8.2 Búsqueda de patrones

- Entrada: tramo del `.seq`, lista de 1 a 10 patrones de longitud 2 a 64, formados por A, C, G, T y opcionalmente N como comodín de una base. Opción `complemento_inverso` (por defecto activada): busca también el complemento inverso de cada patrón (A↔T, C↔G, invertido), que es como se buscan motivos en ADN de doble hebra.
- Coincidencia sin distinguir mayúsculas y minúsculas. Se cuentan coincidencias **solapadas** (en `AAAA`, el patrón `AAA` aparece 2 veces).
- **Costuras:** cada tarea recibe `solape = longitud_maxima_de_patron - 1` bytes extra del tramo siguiente (si existe) y cuenta solo las coincidencias que **empiezan** en `[inicio, fin)`. Así ninguna coincidencia se pierde ni se cuenta dos veces.
- **Registros:** una coincidencia que cruza el límite entre dos registros del `.seq` (fin de un cromosoma e inicio del siguiente) **no cuenta**, porque no existe en el ADN real. El worker recibe la lista de límites de registros que caen dentro de su tramo.
- Salida: conteo por patrón y por hebra, conteo por registro (cromosoma), y las primeras K posiciones (registro, posición, fila, columna).
- Patrones sugeridos para la demostración: `TATAAA` (caja TATA), `GAATTC` (sitio de corte de la enzima EcoRI), `GGATCC` (BamHI), `CCGG`, y uno largo como `GATTACA`.
- Implementación de referencia: para cada patrón de longitud L, comparación vectorizada con numpy de L desplazamientos (`coincide &= (datos_mayus[i:i+n] == p[i])`), donde `datos_mayus = datos & 0xDF` convierte minúsculas en mayúsculas para letras.

### 8.3 Comparación de dos cadenas

- Entrada: dos archivos preparados (A y B) con sus índices.
- **Modo emparejado (por defecto):** reutiliza `emparejar` de `referencias/P1.4/secuencias.py`: primero por cromosoma ("chromosome N" en la descripción), luego por longitud única entre lo que queda. Lo que no tiene pareja se lista aparte y no se compara. Las tareas son tramos **dentro de una pareja** (las unidades de 4 MB se definen por pareja, sobre el `.seq` de A, con el desplazamiento correspondiente en B). Muchas parejas son contigs pequeños: agrupar varias parejas pequeñas en una misma tarea para no pagar el costo fijo por cada una (el P1.4 midió que comparar pareja por pareja hizo subir el tiempo de la GPU de 3,6 a 6,3 s).
- **Modo posicional:** compara posición contra posición en todo el `.seq`, sobre la longitud menor. Existe para mostrar el contraste del informe.
- Categorías de diferencia, reportadas por separado:
  - `solo_caso`: misma base con distinto caso (`a` contra `A`). Biológicamente no es diferencia.
  - `con_n`: una de las dos posiciones es N.
  - `reales`: el resto.
  - `total` = suma de las tres.
- Salida: conteos por categoría, posiciones comparadas, tabla por pareja (con diferencias por pareja), primeras K diferencias con (registro, posición, fila, columna, byte en A, byte en B), lista de registros sin pareja.
- Validación obligatoria: ninguna categoría puede superar las posiciones comparadas; si pasa, el resultado se marca inválido y no se publica (regla heredada del bug de DirectML).
- Valores de aceptación sobre GCA contra GCF: emparejado, 701 parejas (25 por cromosoma y 676 por longitud), 12 registros sin pareja, 3.298.425.684 posiciones comparadas y **0 diferencias**; posicional, 3.298.430.636 comparadas y 1.072.801.765 diferencias, la primera en la posición 1.945.577.269.
- Para probar que la comparación sí detecta diferencias, adapta `referencias/P1.4/generar_par.py`: genera un par a partir de un recorte del genoma con un número conocido de sustituciones (sin inserciones ni borrados) y su `.esperado.txt` con el desglose por categoría.

### 8.4 Zonas de interés (operación de la NPU)

- Objetivo (semana 3 del enunciado): clasificar ventanas del ADN para detectar regiones de interés biológico. Se usa la definición clásica de **isla CpG** de Gardiner-Garden y Frommer: ventana con contenido de G+C mayor a 0,5 y cociente observado/esperado de dinucleótidos CG mayor a 0,6, donde `obs/esp = n_CG × L / (n_C × n_G)`.
- Entrada: tramo del `.seq`, tamaño de ventana `W` (por defecto 200 bases) y paso `S` (por defecto 200, ventanas sin solape). Las ventanas que contienen alguna N o algún inválido se marcan como "no evaluable". Las ventanas no cruzan límites de registro.
- Costuras: la tarea recibe `W - 1` bytes de solape y solo evalúa ventanas que empiezan en `[inicio, fin)`.
- **Implementación de referencia (regla, en CPU):** cuenta C, G y CG por ventana con numpy y aplica la regla.
- **Implementación como modelo (para la NPU), en `pdn/motores/npu/`:**
  1. Entrada del modelo: lote de ventanas codificadas en one-hot de 4 canales (A, C, G, T; mayúscula y minúscula igual; N y otros en cero), forma `[lote, 4, W]`.
  2. Capas de extracción con pesos fijos: una convolución 1D que suma los canales C y G (contenido GC), y una convolución 1D de tamaño de núcleo 2 que detecta el dinucleótido CG; reducción por suma a lo largo de la ventana. Esto produce `n_C`, `n_G`, `n_CG` por ventana.
  3. Capa de clasificación: una pequeña red densa (2 capas) **entrenada** para reproducir la etiqueta de la regla a partir de las características, con datos etiquetados por la implementación de referencia sobre ventanas reales y sintéticas. Así es un "modelo de clasificación" de verdad, como pide el enunciado.
  4. Salida: probabilidad de "zona de interés" por ventana; umbral 0,5.
  5. Se construye en PyTorch o directamente en numpy, se exporta a Core ML con `coremltools`, y se **cuantiza a INT8** (pesos y, si la versión lo permite, activaciones) con las utilidades de optimización de `coremltools`. Se guarda en `pdn/motores/npu/modelos/` (un archivo `.mlpackage` pequeño puede versionarse; si pesa más de 5 MB, se genera con un script).
  6. Métricas de la cuantización: concordancia entre el modelo INT8, el modelo float32 y la regla, sobre un conjunto de prueba. Se espera concordancia muy alta porque las características de entrada son conteos exactos; cualquier desacuerdo se reporta, no se oculta.
- Salida de la operación: número de ventanas evaluadas, no evaluables y positivas; densidad de zonas por cromosoma; las primeras K zonas (registro, inicio, fin, GC, obs/esp, probabilidad).
- Esta operación también corre en CPU y GPU (implementación de la regla), para comparar plataformas y para cuando la Mac no esté.

---

## 9. WORKERS

### 9.1 Proceso de un worker

Comando: `python -m pdn.worker --master IP_PRINCIPAL --respaldo IP_RESPALDO --dispositivo cpu|gpu|npu|opencl [opciones]`.

1. Detectar hardware (9.2).
2. Verificar archivos locales en `~/pdn-datos/` (o la ruta configurada) y sus huellas.
3. Preparar el motor **antes de cualquier cronómetro**: crear el pool de procesos y fijar afinidades (CPU), compilar kernels y reservar buffers (GPU), cargar el modelo (NPU).
4. Registrarse (`REGISTRO`).
5. Calibrar: procesar un bloque de 64 MB del archivo de la corrida (o sintético si no hay corrida) y enviar `CALIBRACION`.
6. Bucle: `PEDIR`, procesar `TAREA`, enviar `RESULTADO`. Un hilo aparte envía `LATIDO` cada 1 s.
7. Ante `FIN` queda en espera; ante `DETENER` libera recursos y termina.
8. Importar todos los módulos al arrancar (no importaciones tardías durante la corrida), para que una caída de NFS no rompa un worker en marcha.

### 9.2 Detección de hardware (`pdn/worker/hardware.py`)

- CPU: modelo (`/proc/cpuinfo` o `lscpu` en Linux, `sysctl -n machdep.cpu.brand_string` en macOS), núcleos físicos y lógicos (`psutil`), frecuencia, flags relevantes (`avx2`, `avx512f`, `neon`).
- **Núcleos de rendimiento y de eficiencia en Intel híbridos** (el i5-13420H): en Linux, `/sys/devices/cpu_core/cpus` lista los núcleos P y `/sys/devices/cpu_atom/cpus` los E. Si esas rutas no existen, el procesador no es híbrido. Identificar también los hermanos de hyperthreading (`/sys/devices/system/cpu/cpuN/topology/thread_siblings_list`).
- RAM total y disponible.
- GPU CUDA: `numba.cuda.is_available()`, nombre, VRAM, compute capability, número de SMs (vía `pynvml` y `numba.cuda.get_current_device()`).
- GPU integrada por OpenCL: `pyopencl`, si está instalado (opcional).
- NPU: en macOS con Apple Silicon (`platform.machine() == 'arm64'`), Neural Engine disponible si `coremltools` importa correctamente. En Linux, ningún nodo del grupo tiene NPU; detectar igual los proveedores de ONNX Runtime si estuviera instalado, por generalidad.
- Interfaz de red activa, IP y velocidad del enlace (`/sys/class/net/<if>/speed`), para detectar la tarjeta de 100 Mbps.
- Sistema operativo y versiones de Python, numpy, OpenMPI y del propio programa. El Master advierte si las versiones de OpenMPI o Python difieren entre nodos Linux.

### 9.3 Varios workers en una misma laptop

- En `nodo-vivanco` corren `cpu` y `gpu`; en `nodo-naranjo`, `cpu` y `gpu`; en `nodo-hidalgo`, `npu` y opcionalmente `cpu`.
- Cuando un nodo tiene worker GPU, su worker CPU usa por defecto **un núcleo menos** que el total, y ese núcleo queda para el hilo que alimenta a la GPU (lección 5 de la sección 4). Configurable.

---

## 10. MOTORES

### 10.1 Motor CPU (`pdn/motores/cpu.py`)

- Pool persistente de `N` procesos (`multiprocessing`, contexto `fork` en Linux y `spawn` en macOS), creado una vez al arrancar el worker.
- Cada proceso abre el `.seq` con `numpy.memmap` (sin copiar datos entre procesos) y procesa una porción del tramo de la tarea.
- **Afinidad de núcleos:** el inicializador de cada proceso llama a `os.sched_setaffinity(0, {núcleo})`. La lista de núcleos se elige en el dashboard: "todos", "solo rendimiento", "solo eficiencia", "sin hermanos de hyperthreading", o una lista manual. Se registra en el resultado qué núcleo usó cada proceso. En macOS no existe esta llamada: se informa "afinidad no disponible en macOS" y se continúa.
- El número de procesos y la lista de núcleos se validan contra el hardware detectado (no se pueden pedir 100 procesos en una CPU de 8 hilos), igual que hacía el P1.4 en su interfaz.
- Dos implementaciones seleccionables del cálculo, para poder comparar:
  - `numpy`: vectorizada con numpy (SIMD implícito, el que use numpy por dentro).
  - `simd`: núcleo en C con **intrínsecas AVX2 explícitas** (10.2).
- Reutiliza la lógica de `referencias/P1.3/motor_cpu.py` (`contar_rango`) y `referencias/P1.4/motor_cpu.py` (`comparar_rango`).

### 10.2 Núcleo SIMD en C (`pdn/motores/simd/`)

Este es el entregable de "instrucciones vectoriales (SIMD)" de la rúbrica. Debe ser explícito y explicable en la defensa.

- Archivo `simd_adn.c` con funciones exportadas en C puro, llamadas desde Python con `ctypes`:
  - `uint64_t contar_simbolos_avx2(const uint8_t* d, size_t n, uint64_t* salida)`: cuenta A, C, G, T, N (ambos casos) e IUPAC con `_mm256_cmpeq_epi8`, `_mm256_movemask_epi8` y `__builtin_popcount` sobre bloques de 32 bytes, más un recorrido escalar del resto.
  - `void histograma_escalar(const uint8_t* d, size_t n, uint64_t* hist)`: histograma de 256 casillas con 4 tablas parciales para evitar dependencias de memoria (el histograma completo no es vectorizable de forma directa; documentarlo).
  - `uint64_t comparar_avx2(const uint8_t* a, const uint8_t* b, size_t n, uint64_t* categorias)`: cuenta diferencias, separando las de solo caso (comparando también `a & 0xDF` contra `b & 0xDF`) y las que involucran N.
  - `uint64_t buscar_patron_avx2(const uint8_t* d, size_t n, const uint8_t* p, size_t m)`: busca un patrón comparando el primer byte en paralelo de 32 en 32 y verificando candidatos.
  - Versión escalar equivalente de cada función, para medir la ganancia de AVX2.
- `scripts/compilar_simd.sh` compila dos bibliotecas: `libsimd_avx2.so` con `gcc -O3 -mavx2 -mpopcnt -shared -fPIC` y `libsimd_escalar.so` con `gcc -O3 -fno-tree-vectorize -shared -fPIC` (para la comparación justa, sin que el compilador vectorice solo).
- En tiempo de ejecución, verificar el flag `avx2` antes de cargar la versión AVX2. Todos los procesadores Linux del grupo lo tienen. En la Mac (ARM) no existe AVX2: usar la implementación numpy y documentarlo (opcionalmente, una versión NEON en una etapa extra).
- Pruebas: cada función C debe dar exactamente lo mismo que la referencia de `pdn/operaciones/` sobre datos aleatorios con todos los casos (minúsculas, N, IUPAC, inválidos, longitudes que no son múltiplo de 32).
- `benchmark` específico: escalar contra AVX2 contra numpy, en MB/s, para el informe.

### 10.3 Motor GPU CUDA (`pdn/motores/gpu_cuda.py`)

- Numba CUDA (`numba-cuda[cu12]`), kernels escritos a mano con `@cuda.jit`.
- Reutiliza `ContextoGPU` de `referencias/P1.3/motor_gpu.py` (reserva la tarjeta una vez y atiende muchos tramos) y el kernel de comparación de `referencias/P1.4/motor_gpu.py`. Kernels necesarios: histograma (con acumulación en memoria compartida del bloque y sumas atómicas), comparación con categorías, búsqueda de patrones (cada hilo prueba posiciones con grid-stride) y conteo por ventanas para zonas.
- Configurable: hilos por bloque (por defecto 256), bloques (1 a 1024, por defecto 1024), tamaño de lote (por defecto 64 MB; en la MX450 de 2 GB, máximo 256 MB, y el motor debe limitarlo según la VRAM libre como hacía `lote_valido` del P1.4).
- **Solapamiento de transferencia y cálculo** (requisito de la rúbrica): memoria pinned en el host, dos juegos de buffers y dos streams; mientras un lote se procesa, el siguiente se copia. Medir y reportar por tarea `t_transferencia` y `t_kernel`, y el porcentaje de solapamiento.
- `precalentar()` compila todos los kernels antes de cualquier medición.
- `cuda.synchronize()` antes de parar el cronómetro.
- Las posiciones capturadas por la GPU llegan en desorden: ordenarlas antes de devolverlas.
- Censo por SM (opcional, heredado del P1.3, con el registro PTX `%smid`): cuántos bytes procesó cada multiprocesador.
- **Pruebas en la nube sin GPU:** usar el simulador de CUDA de Numba (`NUMBA_ENABLE_CUDASIM=1`) con datos pequeños para validar la lógica de los kernels. Marcar en `ESTADO.md` que el rendimiento real y el solapamiento de streams se prueban en `nodo-vivanco`.

### 10.4 Motor NPU (`pdn/motores/npu/`)

- Solo en macOS con Apple Silicon (`nodo-hidalgo`).
- Ejecuta el modelo cuantizado de la sección 8.4 con `coremltools` (`MLModel(..., compute_units=ComputeUnit.CPU_AND_NE)` para pedir el Neural Engine).
- Alternativa equivalente: el modelo en ONNX ejecutado con ONNX Runtime y su proveedor `CoreMLExecutionProvider`. Elegir una de las dos como principal y documentar por qué en `docs/decisiones.md`. Si se usa ONNX Runtime, respetar la lección del DirectML: pedir solo las salidas necesarias.
- Core ML decide internamente si una capa corre en el Neural Engine, en la GPU o en la CPU. Hay que **verificarlo**: medir la potencia del Neural Engine con `sudo powermetrics` durante la ejecución (sección 11.2) y comparar contra la ejecución con `compute_units=CPU_ONLY`. Si no se puede confirmar, reportarlo tal cual.
- Métricas: inferencias por segundo, ventanas por segundo, energía por ventana, y una estimación de operaciones por segundo y por vatio a partir del número de operaciones del modelo.
- Compatibilidad: si `coremltools` no soporta la versión de Python de Homebrew, documentar en `scripts/setup_mac.sh` la instalación de `python@3.12` y crear el entorno con esa versión.

### 10.5 Motor OpenCL (opcional, etapa 11)

- Reutiliza `referencias/P1.2/motor_opencl.py` para las gráficas integradas: Intel (paquete `intel-opencl-icd`) y AMD (controlador Mesa Rusticl, que se habilita con la variable `RUSTICL_ENABLE=radeonsi`). En el P1.2, la gráfica integrada Intel superó ampliamente a la CPU con Python, así que puede aportar.
- Solo se implementa si las etapas obligatorias están terminadas.

---

## 11. MONITOREO DE RECURSOS Y ENERGÍA

### 11.1 Recursos (`pdn/monitoreo/recursos.py`)

Muestreo cada 1 s en cada worker, enviado en el `LATIDO`, y guardado en `recursos.csv`:
- Uso de cada núcleo lógico (`psutil.cpu_percent(percpu=True)`, un solo muestreador por proceso).
- Frecuencia actual de CPU.
- Temperaturas: `psutil.sensors_temperatures()` (en Intel suele ser `coretemp`, en AMD `k10temp`, como alternativa `acpitz`). En macOS, si no hay lectura sin permisos, "sin dato".
- RAM usada del sistema y memoria del proceso worker.
- GPU (NVML con `pynvml`): uso, memoria usada, temperatura y potencia en mW.
- Bytes enviados y recibidos por la red.
- Reutiliza la estructura de `Monitor` de `referencias/P1.4/monitor.py` (`marcar`, `resumen(desde)`), sin la parte de PowerShell.

### 11.2 Energía (`pdn/monitoreo/energia.py`)

- **CPU en Linux:** contadores RAPL en `/sys/class/powercap/intel-rapl:*/energy_uj` (existen en Intel y, en kernels recientes, también en AMD). Son contadores acumulados en microjoules que se reinician al desbordar (`max_energy_range_uj`): manejar el desborde. Desde hace años esos archivos solo los lee root; `scripts/setup_nodo.sh` debe instalar un servicio o regla que les dé permiso de lectura al arrancar, y el programa debe informar "sin permiso" si no puede leerlos.
- **GPU NVIDIA:** potencia instantánea por NVML, integrada en el tiempo.
- **Mac:** `sudo powermetrics` con los muestreadores de CPU, GPU y Neural Engine, intervalo de 1 s, salida en formato plist para parsear. Requiere sudo sin contraseña solo para ese comando: `scripts/setup_mac.sh` debe explicar cómo agregar la regla en sudoers y verificarla. Si no hay permiso, "sin dato".
- Por cada tarea y cada corrida: energía en joules por dispositivo, MB por joule, y para la NPU, operaciones por vatio.
- Lo no medido queda en `null`, nunca en cero.

---

## 12. DASHBOARD WEB

### 12.1 Tecnología

- FastAPI + uvicorn en el Master activo, puerto 8000. Datos en vivo por WebSocket cada 1 s.
- Interfaz en HTML, CSS y JavaScript sin framework, servida como archivos estáticos desde `pdn/dashboard/static/`.
- Gráficas con Chart.js **copiado dentro del repositorio** (no por CDN): la red del laboratorio puede no tener internet.
- Debe verse bien en una pantalla de 1366×768 y en el proyector.

### 12.2 Pantallas

1. **Topología:** una tarjeta por nodo con hostname, IP, rol (Master, respaldo, worker), estado (conectado, perdido, sospechoso), dispositivos detectados, velocidad del enlace de red, edad del último latido, versión de Python y OpenMPI. Un diagrama de la red con el Master al centro. Esta pantalla es la evidencia del "diseño gráfico del clúster" y de "conectividad y direcciones IP" del P2.1.
2. **Configuración de la corrida:**
   - Operación: conteo y validación, patrones (con los patrones y la opción de complemento inverso), comparación (con el modo emparejado o posicional), zonas de interés (con W y S).
   - Archivo o par de archivos (de los disponibles en todos los nodos seleccionados).
   - Modo: dinámico o MPI.
   - Origen de los datos: copia local o NFS.
   - Nodos participantes (casillas).
   - Por cada nodo: dispositivos a usar (solo los que tiene), número de procesos de CPU, **selección de núcleos** (todos, rendimiento, eficiencia, sin hermanos, manual), implementación de CPU (numpy o SIMD), bloques y lote de GPU.
   - Estrategia de reparto (adaptativa, fija, proporcional), tiempo objetivo por tarea, tamaño máximo de tarea.
   - Verificación de integridad por CRC (activada por defecto), medición justa (calentamiento más mediana de N).
   - Todos los campos numéricos validan su rango mientras se escribe, con el rango calculado a partir del hardware real.
3. **Ejecución en vivo:** progreso global, MB/s del clúster, MB/s por worker, línea de tiempo tipo Gantt de las tareas por worker (cada tarea un rectángulo, color por dispositivo, las reasignadas marcadas), registro de reasignaciones y caídas. Botones: Iniciar, Detener, **Simular fallo** (elegir worker y modo caída o congelado), **Simular caída del Master** (solo si hay respaldo).
4. **Recursos:** por nodo, uso de cada núcleo (barras, distinguiendo núcleos P y E), temperaturas, RAM, GPU, red, potencia y energía acumulada.
5. **Resultados:** resumen de la operación, validación de integridad ("coincide con la referencia" o discrepancias), tabla por cromosoma o por pareja, primeras diferencias o coincidencias con fila y columna, tiempo total, tiempo de preparación aparte, speedup y eficiencia contra la referencia secuencial, energía por arquitectura, MB por joule. Botón Exportar (CSV, JSON y PNG de las gráficas).
6. **Escalabilidad:** lanza series automáticas y grafica tiempo, speedup y eficiencia (sección 13.3), con la estimación de la fracción secuencial por Ley de Amdahl ajustada a los datos medidos.
7. **Evidencias:** la matriz de la rúbrica del P2.3 (sección 14) rellenada con datos reales de la última corrida.
8. **Registro:** log textual de todo el sistema.

### 12.3 Línea de comandos equivalente

Todo lo que hace el dashboard debe poder hacerse por línea de comandos (`python -m pdn.cli ...`), porque las pruebas automáticas y la nube no tienen navegador. Ejemplo: `python -m pdn.cli correr --operacion conteo --archivo GCF.seq --modo dinamico --nodos todos`.

---

## 13. MODO MPI, REFERENCIA SECUENCIAL Y ESCALABILIDAD

### 13.1 Programa MPI (`pdn/mpi/mpi_correr.py`)

- Se lanza con `mpirun` desde el Master, con el código en la misma ruta en todos los nodos (`/home/pdn/CPD_2P`) y el intérprete `~/pdn-env/bin/python`.
- Rank 0 coordina: calcula el reparto, lo envía con `scatter` o cada rank lo calcula de forma determinista.
- Reparto: `iguales` o `proporcional` (a velocidades calibradas guardadas por el modo dinámico).
- Cada rank procesa su tramo de su copia local (o por NFS si se elige) con el motor de CPU (una sola hebra por rank: el paralelismo lo da MPI).
- `MPI.COMM_WORLD.Barrier()` antes de arrancar el reloj; `MPI.Wtime()` para medir.
- Consolidación con `Reduce` (suma de histogramas como arreglos numpy `int64`, `op=MPI.SUM`) y `gather` para posiciones.
- Afinidad: lanzar con `--map-by core --bind-to core --report-bindings` y guardar la salida de `--report-bindings` como evidencia.
- Interfaz de red: pasar `--mca btl_tcp_if_include <subred o interfaz>` para que OpenMPI use la red correcta cuando una laptop tiene cable y Wi-Fi a la vez.
- Mismas operaciones que el modo dinámico (conteo, patrones, comparación posicional, zonas con la regla).
- La Mac no participa.

### 13.2 Hostfile

`scripts/generar_hostfile.py` genera `hostfile` a partir de `cluster.yaml` (`nodo-carranza slots=16`, etcétera), con la opción de limitar slots por nodo.

### 13.3 Series de escalabilidad (`pdn/mpi/escalabilidad.py` y desde el dashboard)

1. **Referencia secuencial:** un proceso, un núcleo, un nodo, archivo completo. Su tiempo es T₁.
2. **Escalabilidad por núcleos en un nodo:** 1, 2, 4, 8, todos.
3. **Escalabilidad por nodos con un núcleo cada uno:** 1, 2, 3, 4, 5 nodos (aísla el efecto de repartir entre máquinas).
4. **Escalabilidad completa:** 1 a 5 nodos con todos sus núcleos.
5. **Modo dinámico:** solo CPU, CPU más GPU, CPU más GPU más NPU.
6. Para cada punto: tiempo, speedup (T₁ / T), eficiencia (speedup / núcleos usados), energía. Exportar CSV y gráficas.
7. Ajustar la Ley de Amdahl a los puntos medidos y reportar la fracción secuencial estimada y el speedup máximo teórico.

### 13.4 Resultados de referencia

Guardar en `/cluster/referencias_resultados/<huella_archivo>_<operacion>_<parametros>.json` el resultado de la corrida secuencial. Toda corrida posterior se valida contra él (sección 7.5).

---

## 14. EVIDENCIAS PARA LA RÚBRICA

`pdn/evidencias.py` arma, a partir de los datos reales de las corridas, una tabla que la pestaña Evidencias del dashboard muestra y que se exporta a `evidencias.md`. Inspirado en `referencias/P1.4/evidencias.py`.

| Criterio | Evidencia que el sistema debe producir |
|---|---|
| Infraestructura: estabilidad bajo carga | Corrida con todos los nodos al máximo durante al menos 5 minutos sin caídas, con su gráfica de uso y latidos |
| Infraestructura: NFS | Estado del montaje de `/cluster` en cada nodo, y comparación de una corrida en modo NFS contra modo local |
| CPU: SIMD | Benchmark escalar contra AVX2 contra numpy, en MB/s |
| CPU: afinidad | Salida de `--report-bindings`, núcleo asignado a cada proceso, y comparación con y sin afinidad, y solo núcleos P contra todos |
| GPU: transferencia sin bloquear | Tiempos de transferencia, de kernel y porcentaje de solapamiento con dos streams contra uno |
| NPU: modelo cuantizado | Tamaño y precisión del modelo float32 contra INT8, concordancia con la regla, energía e inferencias por segundo |
| Balanceo | Línea de tiempo por worker, bytes por worker contra su velocidad, tiempo ocioso de cada worker al final, y comparación adaptativa contra fija |
| Tolerancia a fallos | Registro de una caída simulada de worker y de una caída del Master con la recuperación y el resultado final correcto |
| Integridad | Validación por CRC y coincidencia exacta con la referencia secuencial |
| Informe: rentabilidad y Amdahl | Tabla de tiempo, energía y MB por joule por arquitectura, y ajuste de Amdahl |

---

## 15. ESTRUCTURA DEL REPOSITORIO

```
CPD_2P/
├── README.md                  # Qué es el proyecto y cómo correrlo, en corto
├── CONTEXTO.md                # Este documento
├── ESTADO.md                  # Avance por etapa (lo mantiene Claude Code)
├── MANUAL.md                  # Operación del clúster paso a paso (etapa 10)
├── cluster.yaml               # Nodos, IPs, roles, usuarios, rutas, puertos
├── requirements/
│   ├── base.txt               # numpy, psutil, pyzmq, fastapi, uvicorn[standard], pyyaml, matplotlib, pytest
│   ├── gpu.txt                # numba-cuda[cu12], nvidia-ml-py
│   ├── mac.txt                # coremltools, y torch u onnx si se usan para construir el modelo
│   └── opencl.txt             # pyopencl
├── pdn/
│   ├── __init__.py
│   ├── cli.py
│   ├── comun/                 # config.py, protocolo.py, registro_log.py, formato.py (reglas FASTA), unidades.py, huellas.py
│   ├── preparacion/           # fasta_a_seq.py, indice.py, emparejar.py
│   ├── operaciones/           # conteo.py, patrones.py, comparacion.py, zonas.py (referencias de un hilo)
│   ├── motores/
│   │   ├── cpu.py
│   │   ├── simd/              # simd_adn.c, envoltorio.py
│   │   ├── gpu_cuda.py
│   │   ├── npu/               # construir_modelo.py, cuantizar.py, motor_coreml.py, modelos/
│   │   └── opencl.py          # opcional
│   ├── master/                # servidor.py, planificador.py, estado.py, respaldo.py, consolidador.py, validacion.py, resultados.py
│   ├── worker/                # __main__.py, worker.py, hardware.py, calibracion.py
│   ├── monitoreo/             # recursos.py, energia.py
│   ├── mpi/                   # mpi_correr.py, escalabilidad.py
│   ├── dashboard/             # app.py (FastAPI), static/ (html, css, js, chart.js)
│   ├── sim/                   # clúster simulado local
│   └── evidencias.py
├── herramientas/              # generar_sintetico.py, ensuciar.py, generar_par.py, recortar.py
├── scripts/                   # ver sección 15.1
├── tests/
├── docs/                      # decisiones.md, arquitectura.md, diagramas
├── resultados/                # local, ignorado por Git
└── referencias/               # Parcial 1 y enunciado (solo lectura)
```

### 15.1 Scripts (`scripts/`)

Todos con `set -euo pipefail`, mensajes claros en español, y que lean `cluster.yaml` (con un pequeño ayudante en Python para extraer valores).

| Script | Dónde se corre | Qué hace |
|---|---|---|
| `setup_nodo.sh` | Cada nodo Linux | Verifica paquetes de apt, crea o actualiza `~/pdn-env`, instala `requirements/base.txt` (y `gpu.txt` si hay NVIDIA), compila el núcleo SIMD, da permiso de lectura a RAPL, verifica SSH y firewall, imprime un resumen |
| `setup_master.sh` | Master | Lo de `setup_nodo.sh` más: configura `/etc/exports` para `/cluster` con la subred indicada, `exportfs -ra`, arranca `nfs-kernel-server` |
| `setup_mac.sh` | Mac | Verifica Homebrew y Python, crea el entorno, instala `requirements/mac.txt`, explica la regla de sudoers para `powermetrics` |
| `copiar_llaves.sh` | Master | `ssh-copy-id` a cada nodo y `ssh-keyscan` para registrar las huellas en `known_hosts` (sin esto, MPI se queda esperando la pregunta de confirmación) |
| `montar_nfs.sh` | Cada worker | Monta `/cluster` del Master (Linux: `mount -t nfs`; Mac: `mount -t nfs -o resvport`) |
| `desplegar_codigo.sh` | Master | Copia el código a `/home/pdn/CPD_2P` de cada nodo con `rsync` (la misma ruta en todos, requisito de MPI) |
| `distribuir_datos.sh` | Master | Sección 6.5 |
| `estado_cluster.sh` | Master | Para cada nodo: ping, SSH sin contraseña, hostname, versiones de Python y OpenMPI, montaje NFS, datos presentes, GPU. Tabla de verde o rojo. Evidencia del P2.1 |
| `lanzar_cluster.sh` | Master | Por SSH: arranca los workers de cada nodo con `nohup` dentro de `~/pdn-env`, arranca el Master de respaldo en `nodo-carranza`, y arranca el Master y el dashboard |
| `detener_cluster.sh` | Master | Detiene todo ordenadamente |
| `compilar_simd.sh` | Cada nodo Linux | Sección 10.2 |
| `generar_hostfile.py` | Master | Sección 13.2 |

### 15.2 `cluster.yaml` de ejemplo

```yaml
red:
  subred: 192.168.1.0/24          # se ajusta el dia de la reunion
  puerto_tareas: 5555
  puerto_respaldo: 5556
  puerto_dashboard: 8000
rutas:
  codigo: /home/pdn/CPD_2P
  datos: /home/pdn/pdn-datos
  nfs: /cluster
  entorno: /home/pdn/pdn-env
nodos:
  - hostname: nodo-vivanco
    ip: 192.168.1.10
    usuario: pdn
    roles: [master, worker]
    dispositivos: [cpu, gpu]
  - hostname: nodo-carranza
    ip: 192.168.1.11
    usuario: pdn
    roles: [respaldo, worker]
    dispositivos: [cpu]
  - hostname: nodo-ocampo
    ip: 192.168.1.12
    usuario: pdn
    roles: [worker]
    dispositivos: [cpu]
  - hostname: nodo-naranjo
    ip: 192.168.1.13
    usuario: pdn
    roles: [worker]
    dispositivos: [cpu, gpu]
  - hostname: nodo-palomo
    ip: 192.168.1.14
    usuario: pdn
    roles: [worker]
    dispositivos: [cpu]
  - hostname: nodo-hidalgo
    ip: 192.168.1.15
    usuario: CAMBIAR             # el usuario de la Mac no es pdn
    sistema: macos
    roles: [worker]
    dispositivos: [npu]
    rutas:
      codigo: /Users/CAMBIAR/CPD_2P
      datos: /Users/CAMBIAR/pdn-datos
      entorno: /Users/CAMBIAR/pdn-env
```

Los dispositivos de `cluster.yaml` son la intención; lo que manda es lo que cada worker detecta al registrarse.

---

## 16. PRUEBAS

### 16.1 Principios

- `pytest`. Todas las pruebas que no necesitan hardware especial deben correr en la nube y pasar antes de cerrar cada etapa.
- Las que necesitan GPU real, NPU, varios nodos o MPI multinodo se marcan con marcadores de pytest (`@pytest.mark.gpu`, `npu`, `multinodo`) y se saltan automáticamente si no hay hardware.

### 16.2 Datos sintéticos (`herramientas/generar_sintetico.py`)

Genera archivos FASTA pequeños (de 1 a 200 MB) con **respuesta conocida**, con semilla reproducible:
- Varios registros de longitudes variadas, incluyendo registros diminutos (menos de 100 bases) y uno que no sea múltiplo de 80.
- Mezcla de mayúsculas y minúsculas, tramos de N, códigos IUPAC sueltos e inválidos en cantidad conocida.
- Una variante con finales de línea `\r\n`.
- Patrones insertados en posiciones conocidas, **incluyendo posiciones que caen justo en los bordes de las unidades de 4 MB** y patrones que cruzan límites de registro (que no deben contarse).
- Un archivo `.esperado.json` con todos los conteos, posiciones y zonas esperadas.
- Pares para comparación con sustituciones conocidas, diferencias de solo caso y de N, y un par con los registros en distinto orden (como GenBank contra RefSeq).

### 16.3 Pruebas obligatorias

1. **Preparación:** `.seq`, `.idx` y `.huellas` correctos; regeneración cuando cambia el origen.
2. **Operaciones de referencia** contra los `.esperado.json`.
3. **Equivalencia de motores:** CPU numpy, CPU SIMD, GPU (simulador) y NPU (si hay) dan exactamente lo mismo que la referencia.
4. **Costuras:** para cada operación, repetir con al menos 10 combinaciones distintas de tamaño de unidad (cambiando la constante en la prueba), tamaño de tarea y número de workers, y exigir siempre el mismo resultado.
5. **Clúster simulado (`pdn/sim/`):** levantar el Master y de 3 a 6 workers como procesos locales en `127.0.0.1`, con velocidades artificialmente distintas (un retardo configurable por worker, para simular un i3 y un i9), y verificar: resultado correcto, que el worker rápido procesa más bytes, y que los tiempos ociosos al final son pequeños.
6. **Tolerancia a fallos:** matar un worker (`SIMULAR_FALLO` en modo caída) y congelar otro a mitad de la corrida; el resultado final debe ser idéntico y las tareas deben aparecer reasignadas.
7. **Integridad:** darle a un worker una copia del `.seq` con un byte cambiado; el Master debe rechazar sus resultados, marcar el nodo como sospechoso y terminar con el resultado correcto.
8. **Alta disponibilidad:** matar el proceso del Master principal a mitad de la corrida en el clúster simulado; el respaldo debe promoverse, los workers reconectarse y el resultado final ser idéntico.
9. **MPI local:** si `mpirun` está disponible en la nube, `mpirun -np 4 python -m pdn.mpi.mpi_correr ...` sobre datos sintéticos debe dar el resultado de referencia. Si no está, marcarlo pendiente en `ESTADO.md`.
10. **Dashboard:** pruebas de la API (FastAPI `TestClient`) para iniciar una corrida simulada, leer el estado y exportar.
11. **Validaciones de rango:** pedir más procesos que hilos, más lote que VRAM, núcleos inexistentes, patrones con letras inválidas: errores claros, nunca caídas.

### 16.4 Pruebas en hardware real (las hará otra sesión en las laptops)

Documentar en `ESTADO.md` y en `MANUAL.md` la lista exacta: CUDA real y solapamiento en `nodo-vivanco`; MX450 en `nodo-naranjo`; Core ML y Neural Engine en `nodo-hidalgo`; RAPL; MPI multinodo; NFS; los valores de aceptación de las secciones 8.1 y 8.3 sobre los genomas reales.

---

## 17. ETAPAS DE DESARROLLO

Cada etapa: implementar, probar, actualizar `ESTADO.md`, commit. **Prioridad por el plazo:** las etapas E0 a E7 son indispensables para la presentación; E8 y E9 son muy importantes para la rúbrica; E10 es indispensable para operar el clúster; E11 es opcional.

### E0. Reorganización y esqueleto
Sección 1 completa. Crear la estructura de carpetas de la sección 15, `requirements/`, `cluster.yaml` de ejemplo, `README.md` mínimo, `ESTADO.md` y `docs/decisiones.md`. Configurar pytest.
**Terminado cuando:** el repositorio tiene la estructura, `pytest` corre (aunque sin pruebas todavía) y nada de datos está versionado.

### E1. Datos y operaciones de referencia
`pdn/comun/formato.py`, `pdn/preparacion/` (portado del P1.4), `pdn/operaciones/` (las cuatro operaciones en un hilo), `herramientas/generar_sintetico.py`, `herramientas/ensuciar.py` y `herramientas/generar_par.py` (portados). Pruebas 16.3.1, 16.3.2 y la parte de costuras que aplique a la referencia.
**Terminado cuando:** las cuatro operaciones dan exactamente los `.esperado.json` sobre todos los sintéticos.

### E2. Motor CPU y núcleo SIMD
`pdn/motores/cpu.py` con pool, afinidad y detección de núcleos P y E; `pdn/motores/simd/` y `scripts/compilar_simd.sh`; benchmark local escalar contra AVX2 contra numpy. Pruebas de equivalencia y de costuras.
**Terminado cuando:** el motor CPU da los resultados de referencia con 1 y con N procesos, con y sin SIMD, y el benchmark produce un CSV.

### E3. Master, workers y clúster simulado
`pdn/comun/protocolo.py`, `pdn/master/` (sin respaldo todavía), `pdn/worker/` (solo dispositivo CPU), `pdn/sim/`, `pdn/cli.py`. Planificador adaptativo con fase final, registro de tareas, reasignación, integridad por CRC, consolidación y persistencia de resultados.
**Terminado cuando:** pasan las pruebas 16.3.4 a 16.3.7 en el clúster simulado.

### E4. Modo MPI, referencia secuencial y escalabilidad
`pdn/mpi/`, `scripts/generar_hostfile.py`, resultados de referencia, series de escalabilidad con ajuste de Amdahl.
**Terminado cuando:** pasa la prueba 16.3.9 (o queda documentada como pendiente) y las series generan CSV y gráficas sobre el clúster simulado o con varios procesos locales.

### E5. Motor GPU CUDA
`pdn/motores/gpu_cuda.py` portado del P1.3 y P1.4 con los kernels nuevos, streams dobles y medición de solapamiento; integración como dispositivo `gpu` en el worker; reserva de núcleo para alimentar la GPU.
**Terminado cuando:** las pruebas de equivalencia pasan con el simulador de CUDA.

### E6. Monitoreo y energía
`pdn/monitoreo/`, integración en los latidos y en `recursos.csv` y `energia.csv`.
**Terminado cuando:** el clúster simulado produce series de recursos coherentes, y lo no disponible en la nube aparece como `null` con su motivo.

### E7. Dashboard
`pdn/dashboard/` con las ocho pantallas de la sección 12.2, conectado al Master y al clúster simulado. Chart.js local.
**Terminado cuando:** se puede configurar, lanzar, observar y exportar una corrida del clúster simulado desde el navegador, y simular un fallo desde un botón.

### E8. Alta disponibilidad
`pdn/master/respaldo.py`, lista de Masters en los workers, promoción y reconexión.
**Terminado cuando:** pasa la prueba 16.3.8.

### E9. NPU en la Mac
`pdn/motores/npu/`: construcción del modelo, entrenamiento de la parte densa, exportación a Core ML, cuantización INT8, métricas de concordancia; motor y dispositivo `npu` en el worker; `scripts/setup_mac.sh`; medición con `powermetrics`.
**Terminado cuando:** el modelo se construye y se valida contra la regla en cualquier sistema (la construcción no requiere Mac si se usa la conversión de `coremltools`; si la requiere, dejar todo listo y documentado para ejecutarlo en `nodo-hidalgo`), y el worker NPU se integra en el clúster simulado usando la alternativa en CPU cuando no hay Neural Engine.

### E10. Despliegue y manual
Todos los scripts de la sección 15.1, `MANUAL.md` con el procedimiento completo del día de la presentación (red, IPs, `cluster.yaml`, llaves, NFS, código, datos, verificación, lanzamiento, corridas de evidencia, apagado) y una lista de verificación final.
**Terminado cuando:** los scripts pasan `shellcheck` y el manual permite a alguien que no escribió el código levantar el clúster.

### E11. Opcionales
Motor OpenCL para gráficas integradas, tareas especulativas, versión NEON del núcleo SIMD para la Mac, censo por SM.

---

## 18. LO QUE NO SE DEBE HACER

- Subir datos de ADN, binarios de Windows o entornos virtuales al repositorio.
- Hacer máquinas virtuales (decisión del grupo).
- Inventar métricas, rellenar con ceros o con valores de ejemplo lo que no se midió.
- Ocultar resultados incómodos (por ejemplo, que agregar nodos o dispositivos no acelere porque el cuello es el disco o la red). Se miden y se explican.
- Medir con la preparación dentro del cronómetro.
- Depender de internet en tiempo de ejecución (CDNs, descargas).
- Depender de NFS para que un worker siga vivo.
- Usar `sudo` en tiempo de ejecución, salvo lo documentado para `powermetrics` en la Mac.
- Reescribir desde cero lógica que ya está probada en `referencias/` sin una razón anotada en `docs/decisiones.md`.
- Afirmar en la interfaz o en la documentación que algo corre en la NPU, que está cuantizado o que usa SIMD si no está verificado.

---

## 19. PREGUNTAS ABIERTAS PARA EL PROFESOR

Anotarlas en `ESTADO.md` para que el grupo las haga:
1. ¿El tiempo de distribuir el archivo a los nodos cuenta en el examen?
2. ¿Acepta Core ML con el Neural Engine de Apple como implementación de NPU, en lugar de OpenVINO o TensorRT?
3. Con el archivo de 10 GB del examen, ¿qué operación se pide exactamente?

---

## 20. DEFINICIÓN DE TERMINADO DEL PROYECTO

El proyecto está listo para presentar cuando:
1. Las tres operaciones (más zonas de interés) dan los valores de aceptación de las secciones 8.1 y 8.3 sobre los genomas reales, en modo dinámico y en modo MPI.
2. El clúster de seis nodos corre una corrida completa con todos los dispositivos, y el dashboard muestra en vivo el reparto, los recursos y la energía.
3. La caída simulada de un worker y la del Master se recuperan con resultado correcto.
4. Existen las evidencias de la sección 14, exportadas.
5. `MANUAL.md` permite repetir todo el día de la presentación.
6. `docs/decisiones.md` explica cada decisión no obvia, para la sección de contraste con la IA del informe.
