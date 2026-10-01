# Manual de operación del clúster PDN (P2.3)

Procedimiento completo para el día de la presentación, pensado para que lo siga alguien que no escribió el código. Los comandos se ejecutan como el usuario `pdn` desde `/home/pdn/CPD_2P`, salvo que se indique otra cosa.

Contenido:

1. [Antes del día](#1-antes-del-día)
2. [Red y direcciones IP](#2-red-y-direcciones-ip)
3. [`cluster.yaml`](#3-clusteryaml)
4. [Llaves SSH](#4-llaves-ssh)
5. [Instalación de cada nodo](#5-instalación-de-cada-nodo)
6. [NFS](#6-nfs)
7. [Código](#7-código)
8. [Datos](#8-datos)
9. [Verificación](#9-verificación)
10. [Lanzamiento](#10-lanzamiento)
11. [Corridas de evidencia](#11-corridas-de-evidencia)
12. [Apagado](#12-apagado)
13. [Lista de verificación final](#13-lista-de-verificación-final)
14. [Problemas frecuentes](#14-problemas-frecuentes)

---

## 1. Antes del día

En **cada laptop Linux** (Ubuntu 24.04 en doble arranque):

```bash
sudo adduser pdn                      # mismo usuario en todos los nodos Linux
sudo usermod -aG sudo pdn
sudo apt update
sudo apt install -y openssh-server openmpi-bin libopenmpi-dev python3-mpi4py python3-venv \
    python3-pip python3-yaml nfs-common build-essential htop rsync git
sudo systemctl enable --now ssh
sudo ufw disable                      # o abra 5555, 5556, 8000 y los puertos de MPI
sudo hostnamectl set-hostname nodo-APELLIDO   # nodo-vivanco, nodo-carranza, ...
```

En el **Master** (`nodo-vivanco`), además:

```bash
sudo apt install -y nfs-kernel-server
sudo mkdir -p /cluster && sudo chown pdn:pdn /cluster
```

En los nodos con **NVIDIA** (`nodo-vivanco`, `nodo-naranjo`), el driver debe funcionar: `nvidia-smi` tiene que listar la tarjeta.

En la **Mac** (`nodo-hidalgo`): instalar [Homebrew](https://brew.sh) y activar *Ajustes → General → Compartir → Inicio de sesión remoto*.

Los genomas (`GCF_000001405.40_GRCh38.p14_genomic.fna` y `GCA_000001405.29_GRCh38.p14_genomic.fna`) se copian a `~/pdn-datos/` del Master. **Nunca** al repositorio.

## 2. Red y direcciones IP

1. Conectar todas las laptops al mismo switch o router (cable si es posible; el Wi‑Fi está permitido).
2. Fijar una IP para cada nodo: con reserva DHCP en el router (lo más simple) o con IP estática en *Configuración → Red*. Ejemplo con la subred `192.168.1.0/24`:

   | Nodo | IP |
   |---|---|
   | nodo-vivanco (Master) | 192.168.1.10 |
   | nodo-carranza (respaldo) | 192.168.1.11 |
   | nodo-ocampo | 192.168.1.12 |
   | nodo-naranjo | 192.168.1.13 |
   | nodo-palomo | 192.168.1.14 |
   | nodo-hidalgo (Mac) | 192.168.1.15 |

3. Verificar desde el Master: `ping -c 2 192.168.1.11` para cada nodo.
4. Si una laptop tiene cable y Wi‑Fi a la vez, el modo MPI necesita la subred correcta: se toma de `red.subred` en `cluster.yaml` y se pasa a OpenMPI con `--mca btl_tcp_if_include`.
5. La tarjeta de `nodo-palomo` puede ser de 100 Mb/s: la pantalla Topología lo marca como "lento".

## 3. `cluster.yaml`

Editar `cluster.yaml` en el Master con las IPs reales, el usuario de la Mac y sus rutas:

```yaml
red:
  subred: 192.168.1.0/24
nodos:
  - hostname: nodo-hidalgo
    ip: 192.168.1.15
    usuario: davidhidalgo            # el usuario real de la Mac
    sistema: macos
    rutas:
      codigo: /Users/davidhidalgo/CPD_2P
      datos: /Users/davidhidalgo/pdn-datos
      entorno: /Users/davidhidalgo/pdn-env
```

Los `dispositivos` de cada nodo son la intención: lo que manda es lo que detecta cada worker al registrarse. `slots` es el número de hilos para MPI. Para comprobar que el archivo se lee bien: `python3 scripts/cfg.py nodos`.

## 4. Llaves SSH

En el Master:

```bash
bash scripts/copiar_llaves.sh
```

Crea `~/.ssh/id_ed25519` si no existe, registra las huellas de cada nodo en `known_hosts` (sin esto `mpirun` se queda esperando la pregunta de confirmación) y copia la llave (pide la contraseña de cada nodo una vez). Al volver a correrlo, cada nodo debe decir "la llave ya estaba copiada".

## 5. Instalación de cada nodo

Primero hay que tener el código en cada nodo (la primera vez, con `git clone` o con `scripts/desplegar_codigo.sh` desde el Master, ver la sección 7).

- **Master:** `bash scripts/setup_master.sh` (hace todo lo de `setup_nodo.sh` y además configura el servidor NFS).
- **Cada nodo Linux:** `bash scripts/setup_nodo.sh`. Revisa los paquetes de apt, crea `~/pdn-env` con `--system-site-packages` (para ver el `mpi4py` de apt; **no** instalar `mpi4py` con pip), instala `requirements/base.txt` (y `requirements/gpu.txt` si hay NVIDIA), compila el núcleo SIMD, da permiso de lectura a RAPL con el servicio `pdn-rapl` y muestra el hardware detectado.
- **Mac:** `bash scripts/setup_mac.sh`. Instala Python 3.12 de Homebrew, crea el entorno con `coremltools` y explica la regla de sudoers para `powermetrics`:

  ```bash
  sudo visudo -f /etc/sudoers.d/pdn-powermetrics
  # agregar:  USUARIO ALL=(root) NOPASSWD: /usr/bin/powermetrics
  ```

  Vuelva a correr el script hasta que diga "sudo -n powermetrics funciona".

## 6. NFS

- En el Master, `setup_master.sh` ya exporta `/cluster` a la subred.
- En **cada worker**: `bash scripts/montar_nfs.sh --persistente` (en la Mac se monta en `~/cluster`, porque la raíz de macOS es de solo lectura).
- `/cluster` guarda los resultados (`/cluster/resultados/`), las referencias (`/cluster/referencias_resultados/`) y, para el modo NFS, los datos (`/cluster/datos/`).
- Si NFS falla, los workers siguen funcionando con su copia local y escriben sus logs en `~/pdn-logs/`.

## 7. Código

El código debe estar en **la misma ruta** en todos los nodos Linux (requisito de MPI): `/home/pdn/CPD_2P`.

```bash
bash scripts/desplegar_codigo.sh            # todos los nodos
bash scripts/desplegar_codigo.sh nodo-ocampo  # solo uno
```

Copia con `rsync` (sin `.git`, entornos ni resultados) y compila el núcleo SIMD en cada nodo Linux. Hay que repetirlo cada vez que cambie el código.

## 8. Datos

```bash
bash scripts/distribuir_datos.sh ~/pdn-datos/GCF_000001405.40_GRCh38.p14_genomic.fna
bash scripts/distribuir_datos.sh ~/pdn-datos/GCA_000001405.29_GRCh38.p14_genomic.fna
```

Para cada nodo: copia el `.fna` con `rsync -P` a `~/pdn-datos/`, lo prepara allí (`.seq`, `.idx`, `.huellas`, `.origen`, fuera de cualquier cronómetro) y compara la huella global con la del Master. El resumen final debe decir `OK` en todos los nodos. Copiar 3 GB por Gigabit tarda unos 30 s por nodo; preparar, alrededor de un minuto.

Para el **modo NFS** (comparación del informe), copiar además los archivos preparados a `/cluster/datos/`:

```bash
cp ~/pdn-datos/GCF_000001405.40_GRCh38.p14_genomic.{seq,idx,huellas,origen} /cluster/datos/
```

## 9. Verificación

```bash
bash scripts/estado_cluster.sh
```

Tabla por nodo: ping, SSH sin contraseña, hostname, Python, OpenMPI, NFS montado, archivos `.seq` y GPU. Todo debe salir en verde (en la Mac, OpenMPI dice `n/a`). Esta salida es evidencia del P2.1 (conectividad y direcciones IP).

Pruebas rápidas en cada nodo (opcional pero recomendado):

```bash
~/pdn-env/bin/python -m pytest -q -x tests/test_operaciones.py tests/test_motor_cpu.py
PDN_GPU_REAL=1 ~/pdn-env/bin/python -m pytest -q tests/test_motor_gpu.py   # solo en nodos con NVIDIA
```

Valores de aceptación sobre el genoma real (sección 8.1 de `CONTEXTO.md`), en el Master:

```bash
python -m pdn.cli referencia --operacion conteo --archivo GCF_000001405.40_GRCh38.p14_genomic
# Debe dar A=923117203 C=642552917 G=645231996 T=925917038 N=161611379 IUPAC=103 invalidos=0
```

## 10. Lanzamiento

En el Master:

```bash
bash scripts/lanzar_cluster.sh
```

Arranca por SSH un worker por cada dispositivo de cada nodo (`cpu`, `gpu`, `npu`, según `cluster.yaml`), el Master de respaldo en `nodo-carranza` y el Master con el dashboard en este nodo. Los logs quedan en `~/pdn-logs/` de cada nodo.

Abrir el dashboard: **http://192.168.1.10:8000** (o la IP del Master). En la pestaña Topología deben aparecer todos los workers en verde. Si el Master cae y el respaldo toma el control, el dashboard pasa a **http://192.168.1.11:8000**.

Para lanzar a mano (por ejemplo, solo un worker):

```bash
~/pdn-env/bin/python -m pdn.worker --master 192.168.1.10 --respaldo 192.168.1.11 --dispositivo cpu
~/pdn-env/bin/python -m pdn.cli master --dashboard --replicar-a 192.168.1.11      # Master
~/pdn-env/bin/python -m pdn.cli respaldo --dashboard                            # en nodo-carranza
```

## 11. Corridas de evidencia

Cada corrida se puede lanzar desde el dashboard (pestaña Configuración) o desde la línea de comandos contra la API del Master:

```bash
python -m pdn.cli correr --url http://192.168.1.10:8000 --operacion conteo \
    --archivo GCF_000001405.40_GRCh38.p14_genomic
```

Los resultados quedan en `/cluster/resultados/<fecha>_<operación>_<id>/` (`resumen.json`, `resultado.json`, `tareas.csv`, `recursos.csv`, `energia.csv`, `config.json`). La pestaña Evidencias arma la matriz de la rúbrica con esas carpetas y la exporta a `evidencias.md`.

Orden sugerido (A = `GCF_000001405.40_GRCh38.p14_genomic`, B = `GCA_000001405.29_GRCh38.p14_genomic`):

| # | Evidencia | Cómo |
|---|---|---|
| 1 | Referencia secuencial (T₁) de cada operación | `python -m pdn.cli referencia --operacion conteo --archivo A --repeticiones 3 --calentamiento` (repetir con `patrones`, `zonas`, y `comparacion --archivo-b B`) |
| 2 | Las tres operaciones más zonas en modo dinámico | Dashboard: conteo, patrones (TATAAA, GAATTC, GGATCC, CCGG, GATTACA), comparación emparejada A contra B (0 diferencias, 701 parejas) y posicional (1.072.801.765), zonas |
| 3 | SIMD | En cada nodo Linux: `python -m herramientas.benchmark_simd --seq ~/pdn-datos/A.seq --mb 512` y copiar `resultados/benchmark_simd.csv` a `/cluster/resultados/` |
| 4 | Afinidad | Modo MPI con `--report-bindings` (`python -m pdn.cli correr --modo mpi --np 60 --hostfile hostfile --operacion conteo --archivo A`, con el hostfile de `python scripts/generar_hostfile.py`); en el dashboard, comparar en `nodo-vivanco` "todos" contra "solo rendimiento" |
| 5 | GPU | En `nodo-vivanco`: `python -m herramientas.benchmark_gpu --seq ~/pdn-datos/A.seq --mb 1024` (1 contra 2 streams); copiar el CSV a `/cluster/resultados/` |
| 6 | NPU | En `nodo-hidalgo`: `python -m pdn.motores.npu.construir_modelo --seq ~/pdn-datos/A.seq --verificar-coreml` y `python -m pdn.motores.npu.verificar_ane`; copiar `metricas.json` y `resultados/verificacion_ane.json` |
| 7 | Balanceo | Dashboard: la misma corrida con estrategia adaptativa y con fija; comparar tiempos y el ocioso final en Resultados |
| 8 | Escalabilidad y Amdahl | `python scripts/generar_hostfile.py` y luego `python -m pdn.mpi.escalabilidad --archivo A --repeticiones 3 --calentamiento --url-master http://192.168.1.10:8000` |
| 9 | Tolerancia a fallos | Durante una corrida larga: Ejecución → elegir un worker → "caída", y otro → "congelado". El resultado final debe ser válido y la línea de tiempo muestra las tareas reasignadas |
| 10 | Alta disponibilidad | Durante una corrida: "Simular caída del Master". Abrir http://192.168.1.11:8000: el banner dice "Master de respaldo activo desde HH:MM:SS" y la corrida termina válida |
| 11 | NFS contra local | La misma corrida con "Origen de los datos" en NFS y en copia local |
| 12 | Estabilidad bajo carga | Una corrida de 5 minutos o más con todos los nodos (por ejemplo, patrones con 10 patrones sobre A, o varias repeticiones seguidas) |
| 13 | Energía | Pestaña Recursos durante las corridas; `energia.csv` y "Energía por arquitectura" en Resultados |

Medición justa: en Configuración marcar "calentamiento descartado" y poner 3 repeticiones; el dashboard informa la mediana. La preparación siempre se informa aparte.

## 12. Apagado

```bash
bash scripts/detener_cluster.sh
```

Detiene workers, respaldo y Master en todos los nodos (`SIGTERM`, cierre ordenado). Los resultados ya están en `/cluster/resultados/`. Para desmontar NFS en un worker: `sudo umount /cluster`.

## 13. Lista de verificación final

- [ ] Todas las laptops en la misma red, con la IP de `cluster.yaml`.
- [ ] `bash scripts/estado_cluster.sh` todo en verde.
- [ ] `bash scripts/distribuir_datos.sh` con `OK` en todos los nodos, para A y B.
- [ ] Referencias secuenciales guardadas (sección 11, paso 1).
- [ ] `bash scripts/lanzar_cluster.sh` y todos los workers en verde en Topología.
- [ ] El respaldo conectado (Topología → Resumen → Respaldo: conectado).
- [ ] El conteo de A coincide con los valores de aceptación y la referencia ("coincide").
- [ ] Evidencias de la sección 11 generadas; pestaña Evidencias exportada a `evidencias.md`.
- [ ] Capturas para el informe tomadas con datos reales (nunca con valores de ejemplo).
- [ ] `bash scripts/detener_cluster.sh` al terminar.

## 14. Problemas frecuentes

| Síntoma | Causa probable | Solución |
|---|---|---|
| `mpirun` se queda colgado | Huella SSH sin aceptar | `bash scripts/copiar_llaves.sh` |
| `mpirun` falla con "unable to reach" | Usa la interfaz equivocada (Wi‑Fi y cable) | Revisar `red.subred` en `cluster.yaml` |
| Un worker sale de la corrida con "copia de datos distinta" | Su `.seq` no es igual al del Master | `bash scripts/distribuir_datos.sh` otra vez |
| Un worker queda "sospechoso" | Su copia tiene bytes dañados (CRC distinto) | Igual que arriba; revisar el disco |
| Energía "sin dato (sin permiso)" | RAPL solo lo lee root | `bash scripts/setup_nodo.sh` (instala `pdn-rapl.service`) |
| Energía de la Mac "sin dato" | Falta la regla de sudoers | `bash scripts/setup_mac.sh` |
| GPU "no cabe en la VRAM" | Lote demasiado grande (MX450: 2 GB) | Bajar "Lote GPU" en Configuración |
| La NPU dice "alternativa en CPU" | Core ML no disponible | Revisar `coremltools` con `scripts/setup_mac.sh` |
| `mpi4py` no importa en el entorno | Se instaló con pip o el entorno no ve el de apt | Recrear `~/pdn-env` con `--system-site-packages` y `sudo apt install python3-mpi4py` |
| El dashboard no carga | Firewall | `sudo ufw disable` |
