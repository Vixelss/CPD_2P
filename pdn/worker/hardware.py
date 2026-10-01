"""Deteccion de hardware de un nodo (CONTEXTO.md, seccion 9.2).

Todo lo que no se puede detectar queda en None con su motivo, nunca en cero.
Las dependencias opcionales (numba, pynvml, coremltools, pyopencl) se
importan de forma protegida.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import socket
import subprocess
import sys

import psutil

from pdn import __version__ as VERSION_PDN


def _leer(ruta: str) -> str | None:
    try:
        with open(ruta, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def expandir_lista_cpus(texto: str | None) -> list[int]:
    """Convierte '0-3,8,10-11' en [0, 1, 2, 3, 8, 10, 11]."""
    if not texto:
        return []
    salida: list[int] = []
    for parte in texto.split(","):
        parte = parte.strip()
        if not parte:
            continue
        if "-" in parte:
            a, b = parte.split("-", 1)
            salida.extend(range(int(a), int(b) + 1))
        else:
            salida.append(int(parte))
    return sorted(set(salida))


def modelo_cpu() -> str:
    """Nombre comercial del procesador."""
    if sys.platform == "darwin":
        try:
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True, timeout=5).stdout.strip() or platform.processor()
        except (OSError, subprocess.SubprocessError):
            return platform.processor()
    info = _leer("/proc/cpuinfo") or ""
    m = re.search(r"model name\s*:\s*(.+)", info)
    return m.group(1).strip() if m else platform.processor() or "desconocido"


def flags_cpu() -> list[str]:
    """Flags relevantes: avx2, avx512f, neon."""
    if platform.machine().lower() in ("arm64", "aarch64"):
        return ["neon"]
    info = _leer("/proc/cpuinfo") or ""
    m = re.search(r"flags\s*:\s*(.+)", info)
    presentes = set(m.group(1).split()) if m else set()
    return [f for f in ("avx2", "avx512f", "popcnt") if f in presentes]


def topologia_cpu() -> dict:
    """Nucleos logicos, fisicos, P y E (Intel hibrido) y hermanos de hyperthreading."""
    logicos = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") \
        else list(range(psutil.cpu_count(logical=True) or 1))
    rendimiento = expandir_lista_cpus(_leer("/sys/devices/cpu_core/cpus"))
    eficiencia = expandir_lista_cpus(_leer("/sys/devices/cpu_atom/cpus"))
    hermanos: dict[int, list[int]] = {}
    for c in logicos:
        lista = expandir_lista_cpus(_leer("/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list" % c))
        if lista:
            hermanos[c] = lista
    # Un nucleo por grupo de hermanos (el de menor numero)
    sin_ht = sorted({min(v) for v in hermanos.values()}) if hermanos else list(logicos)
    return {
        "logicos": logicos,
        "fisicos": psutil.cpu_count(logical=False),
        "rendimiento": rendimiento,
        "eficiencia": eficiencia,
        "hibrido": bool(rendimiento and eficiencia),
        "hermanos": {str(k): v for k, v in hermanos.items()},
        "sin_hermanos": [c for c in sin_ht if c in logicos],
    }


def elegir_nucleos(seleccion: str | list[int] | None, topo: dict) -> list[int]:
    """Resuelve una seleccion de nucleos contra la topologia detectada.

    seleccion: 'todos', 'rendimiento', 'eficiencia', 'sin_hermanos', una
    lista manual o un texto '0-3,6'. Lanza ValueError con un mensaje claro si
    la seleccion no es valida en este hardware.
    """
    logicos = topo["logicos"]
    if seleccion in (None, "", "todos"):
        return list(logicos)
    if seleccion == "rendimiento":
        if not topo["rendimiento"]:
            raise ValueError("Este procesador no es hibrido: no hay nucleos de rendimiento separados")
        return [c for c in topo["rendimiento"] if c in logicos]
    if seleccion == "eficiencia":
        if not topo["eficiencia"]:
            raise ValueError("Este procesador no es hibrido: no hay nucleos de eficiencia")
        return [c for c in topo["eficiencia"] if c in logicos]
    if seleccion in ("sin_hermanos", "sin_ht"):
        return list(topo["sin_hermanos"])
    lista = expandir_lista_cpus(seleccion) if isinstance(seleccion, str) else sorted(set(seleccion))
    inexistentes = [c for c in lista if c not in logicos]
    if inexistentes:
        raise ValueError("Nucleos inexistentes en este nodo: %s (disponibles: %s)"
                         % (inexistentes, logicos))
    if not lista:
        raise ValueError("La lista de nucleos esta vacia")
    return lista


def detectar_gpu_cuda() -> dict:
    """GPU NVIDIA via numba y pynvml (si estan instalados)."""
    info: dict = {"disponible": False, "motivo": None}
    if os.environ.get("NUMBA_ENABLE_CUDASIM") == "1":
        info.update({"disponible": True, "simulador": True, "nombre": "Simulador CUDA de Numba",
                     "vram_mb": None, "cc": None, "sms": None})
        return info
    try:
        from numba import cuda  # noqa: PLC0415
    except Exception as e:  # numba no instalado o roto
        info["motivo"] = "numba no disponible: %s" % e
        return info
    try:
        if not cuda.is_available():
            info["motivo"] = "No hay GPU CUDA o falta el driver"
            return info
        dev = cuda.get_current_device()
        cc = dev.compute_capability
        info.update({"disponible": True, "simulador": False,
                     "nombre": dev.name.decode() if isinstance(dev.name, bytes) else str(dev.name),
                     "cc": "%d.%d" % cc, "sms": int(dev.MULTIPROCESSOR_COUNT)})
        libre, total = cuda.current_context().get_memory_info()
        info["vram_mb"] = int(total // (1024 * 1024))
        info["vram_libre_mb"] = int(libre // (1024 * 1024))
    except Exception as e:
        info["disponible"] = False
        info["motivo"] = "Error al consultar la GPU: %s" % e
    return info


def detectar_npu() -> dict:
    """Neural Engine de Apple (Core ML) u otros proveedores de ONNX Runtime."""
    info: dict = {"disponible": False, "motivo": None}
    if sys.platform == "darwin" and platform.machine() == "arm64":
        try:
            import coremltools  # noqa: F401, PLC0415
            info.update({"disponible": True, "tipo": "Apple Neural Engine (Core ML)",
                         "coremltools": coremltools.__version__})
        except Exception as e:
            info["motivo"] = "coremltools no disponible: %s" % e
        return info
    try:
        import onnxruntime  # noqa: PLC0415
        info["proveedores_onnx"] = onnxruntime.get_available_providers()
    except Exception:
        pass
    info["motivo"] = "No es una Mac con Apple Silicon"
    return info


def detectar_opencl() -> dict:
    """Dispositivos OpenCL (opcional)."""
    try:
        import pyopencl as cl  # noqa: PLC0415
        dispositivos = [d.name for p in cl.get_platforms() for d in p.get_devices()]
        return {"disponible": bool(dispositivos), "dispositivos": dispositivos}
    except Exception as e:
        return {"disponible": False, "motivo": "pyopencl no disponible: %s" % e}


def detectar_red(ip_destino: str | None = None) -> dict:
    """Interfaz activa, IP y velocidad del enlace en Mb/s."""
    ip = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((ip_destino or "192.0.2.1", 9))  # no envia nada
        ip = s.getsockname()[0]
        s.close()
    except OSError:
        pass
    interfaz = None
    for nombre, direcciones in psutil.net_if_addrs().items():
        if any(getattr(d, "address", None) == ip for d in direcciones):
            interfaz = nombre
    velocidad = None
    if interfaz:
        valor = _leer("/sys/class/net/%s/speed" % interfaz)
        try:
            velocidad = int(valor) if valor and int(valor) > 0 else None
        except ValueError:
            velocidad = None
        if velocidad is None:
            stats = psutil.net_if_stats().get(interfaz)
            velocidad = stats.speed if stats and stats.speed > 0 else None
    return {"interfaz": interfaz, "ip": ip, "velocidad_mbps": velocidad,
            "enlace_lento": velocidad is not None and velocidad <= 100}


def version_openmpi() -> str | None:
    """Version de OpenMPI (mpirun --version) o None."""
    if not shutil.which("mpirun"):
        return None
    try:
        salida = subprocess.run(["mpirun", "--version"], capture_output=True, text=True,
                                timeout=10).stdout
        m = re.search(r"(\d+\.\d+\.\d+)", salida)
        return m.group(1) if m else salida.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def detectar(completo: bool = True, ip_master: str | None = None) -> dict:
    """Detecta el hardware del nodo. Con completo=False omite GPU/NPU/OpenCL."""
    import numpy  # noqa: PLC0415

    mem = psutil.virtual_memory()
    info = {
        "hostname": socket.gethostname(),
        "sistema": platform.system(),
        "version_sistema": platform.release(),
        "arquitectura": platform.machine(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pdn": VERSION_PDN,
        "cpu": {"modelo": modelo_cpu(), "flags": flags_cpu(), **topologia_cpu()},
        "ram_mb": int(mem.total // (1024 * 1024)),
        "ram_libre_mb": int(mem.available // (1024 * 1024)),
        "red": detectar_red(ip_master),
    }
    try:
        frec = psutil.cpu_freq()
        info["cpu"]["frecuencia_mhz"] = round(frec.max or frec.current) if frec else None
    except Exception:
        info["cpu"]["frecuencia_mhz"] = None
    if completo:
        info["gpu"] = detectar_gpu_cuda()
        info["npu"] = detectar_npu()
        info["opencl"] = detectar_opencl()
        info["openmpi"] = version_openmpi()
    return info


if __name__ == "__main__":
    import json

    print(json.dumps(detectar(), indent=2, ensure_ascii=False))
