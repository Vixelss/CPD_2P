"""Motor GPU CUDA con Numba (CONTEXTO.md, seccion 10.3).

Portado de referencias/P1.3/motor_gpu.py (ContextoGPU: la tarjeta se reserva
una vez y atiende muchos tramos; histograma con acumulacion en memoria
compartida) y referencias/P1.4/motor_gpu.py (comparacion por categorias).

- Conteo: tubo con memoria pinned, dos juegos de buffers y dos streams;
  mientras un lote se procesa, el siguiente se copia. Por tarea se miden
  t_transferencia, t_kernel y el porcentaje de solapamiento con eventos CUDA.
- Patrones, comparacion y zonas: la operacion de referencia reparte el
  trabajo y llama al nucleo GPU (kernels de busqueda, de categorias y de
  conteo por ventanas). Asi las costuras y los limites de registro son
  exactamente los mismos que en la CPU.
- precalentar() compila todos los kernels antes de cualquier medicion y
  cuda.synchronize() se llama antes de parar el cronometro.
- Sin GPU real, NUMBA_ENABLE_CUDASIM=1 activa el simulador de Numba para
  validar la logica de los kernels con datos pequenos.
"""

from __future__ import annotations

import os
import time

import numpy as np

from pdn.operaciones import operacion
from pdn.preparacion.fasta_a_seq import abrir_seq

MB = 1024 * 1024
_KERNELS: dict | None = None
cuda = None  # se asigna en kernels(); los kernels deben verlo como global (simulador)


class GPUNoDisponible(RuntimeError):
    """No hay GPU CUDA ni simulador; el mensaje dice por que."""


def simulador() -> bool:
    return os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"


def _cuda():
    try:
        from numba import cuda  # noqa: PLC0415
    except Exception as e:
        raise GPUNoDisponible("numba no esta instalado (requirements/gpu.txt): %s" % e) from e
    if not cuda.is_available():
        raise GPUNoDisponible("No hay GPU CUDA disponible (o falta el driver de NVIDIA)")
    return cuda


def kernels() -> dict:
    """Compila (una vez por proceso) y devuelve los kernels."""
    global _KERNELS, cuda
    if _KERNELS is not None:
        return _KERNELS
    cuda = _cuda()

    @cuda.jit
    def k_histograma(datos, n, hist):
        # Histograma de 256 casillas: acumula en memoria compartida del bloque
        # y vuelca una sola vez al global (menos contencion atomica)
        local = cuda.shared.array(256, dtype=np.uint32)
        t = cuda.threadIdx.x
        while t < 256:
            local[t] = 0
            t += cuda.blockDim.x
        cuda.syncthreads()
        i = cuda.grid(1)
        paso = cuda.gridsize(1)
        while i < n:
            cuda.atomic.add(local, datos[i], 1)
            i += paso
        cuda.syncthreads()
        t = cuda.threadIdx.x
        while t < 256:
            v = local[t]
            if v != 0:
                cuda.atomic.add(hist, t, np.uint64(v))
            t += cuda.blockDim.x

    @cuda.jit
    def k_comparar(a, b, n, cats):
        # Diferencias por categoria: 0 solo caso, 1 con N, 2 reales
        local = cuda.shared.array(3, dtype=np.uint32)
        if cuda.threadIdx.x < 3:
            local[cuda.threadIdx.x] = 0
        cuda.syncthreads()
        s = np.uint32(0)
        c = np.uint32(0)
        r = np.uint32(0)
        i = cuda.grid(1)
        paso = cuda.gridsize(1)
        while i < n:
            x = a[i]
            y = b[i]
            if x != y:
                ux = x & 0xDF
                uy = y & 0xDF
                if ux == uy:
                    s += 1
                elif ux == 78 or uy == 78:
                    c += 1
                else:
                    r += 1
            i += paso
        if s:
            cuda.atomic.add(local, 0, s)
        if c:
            cuda.atomic.add(local, 1, c)
        if r:
            cuda.atomic.add(local, 2, r)
        cuda.syncthreads()
        if cuda.threadIdx.x < 3:
            v = local[cuda.threadIdx.x]
            if v != 0:
                cuda.atomic.add(cats, cuda.threadIdx.x, np.uint64(v))

    @cuda.jit
    def k_buscar(d, n_inicios, patron, m, pos, contador):
        # Cada hilo prueba inicios con grid-stride; las coincidencias reservan
        # una ranura con una suma atomica (llegan en desorden)
        i = cuda.grid(1)
        paso = cuda.gridsize(1)
        while i < n_inicios:
            ok = True
            k = 0
            while k < m and ok:
                u = d[i + k] & 0xDF
                p = patron[k]
                if p == 78:
                    ok = u == 65 or u == 67 or u == 71 or u == 84
                else:
                    ok = u == p
                k += 1
            if ok:
                ranura = cuda.atomic.add(contador, 0, 1)
                if ranura < pos.size:
                    pos[ranura] = i
            i += paso

    @cuda.jit
    def k_ventanas(d, inicios, w, malo, nc, ng, ncg):
        # Un hilo por ventana: cuenta no-ACGT, C, G y dinucleotidos CG
        j = cuda.grid(1)
        paso = cuda.gridsize(1)
        while j < inicios.size:
            ini = inicios[j]
            cm = 0
            cc = 0
            cgg = 0
            ccg = 0
            previo_c = False
            for k in range(w):
                x = d[ini + k]
                u = x & 0xDF
                es_base = (u == 65 or u == 67 or u == 71 or u == 84) and (x == u or x == u + 32)
                if not es_base:
                    cm += 1
                if u == 67:
                    cc += 1
                if u == 71:
                    cgg += 1
                    if previo_c:
                        ccg += 1
                previo_c = u == 67
            malo[j] = cm
            nc[j] = cc
            ng[j] = cgg
            ncg[j] = ccg
            j += paso

    _KERNELS = {"histograma": k_histograma, "comparar": k_comparar, "buscar": k_buscar,
                "ventanas": k_ventanas}
    return _KERNELS


class NucleoGPU:
    """Nucleo de calculo en la GPU (mismo contrato que NucleoNumpy)."""

    nombre = "gpu"

    def __init__(self, motor: "MotorGPU") -> None:
        self.motor = motor
        self.cuda = motor.cuda
        self.k = kernels()

    def _malla(self, n: int) -> int:
        return max(1, min(self.motor.bloques, -(-n // self.motor.hilos_bloque)))

    def histograma(self, datos: np.ndarray) -> np.ndarray:
        d = self.cuda.to_device(np.ascontiguousarray(datos, dtype=np.uint8))
        h = self.cuda.to_device(np.zeros(256, dtype=np.uint64))
        if datos.shape[0]:
            self.k["histograma"][self._malla(datos.shape[0]), self.motor.hilos_bloque](d, datos.shape[0], h)
        self.cuda.synchronize()
        return h.copy_to_host().astype(np.int64)

    def contar_categorias(self, a: np.ndarray, b: np.ndarray) -> tuple[int, int, int]:
        n = a.shape[0]
        cats = self.cuda.to_device(np.zeros(3, dtype=np.uint64))
        if n:
            da = self.cuda.to_device(np.ascontiguousarray(a, dtype=np.uint8))
            db = self.cuda.to_device(np.ascontiguousarray(b, dtype=np.uint8))
            self.k["comparar"][self._malla(n), self.motor.hilos_bloque](da, db, n, cats)
        self.cuda.synchronize()
        c = cats.copy_to_host()
        return int(c[0]), int(c[1]), int(c[2])

    def coincidencias(self, mayus: np.ndarray, patron: str) -> np.ndarray:
        m = len(patron)
        n = mayus.shape[0] - m + 1
        if n <= 0:
            return np.zeros(0, dtype=bool)
        d = self.cuda.to_device(np.ascontiguousarray(mayus, dtype=np.uint8))
        p = self.cuda.to_device(np.frombuffer(patron.upper().encode(), dtype=np.uint8).copy())
        capacidad = max(1024, n // 4)
        pos = self.cuda.device_array(capacidad, dtype=np.int64)
        cont = self.cuda.to_device(np.zeros(1, dtype=np.int64))
        self.k["buscar"][self._malla(n), self.motor.hilos_bloque](d, n, p, m, pos, cont)
        self.cuda.synchronize()
        hallados = int(cont.copy_to_host()[0])
        if hallados > capacidad:
            # Desborde del buffer de posiciones: este tramo se resuelve en la CPU
            from pdn.operaciones.nucleo import NUMPY  # noqa: PLC0415
            self.motor.desbordes += 1
            return NUMPY.coincidencias(mayus, patron)
        mascara = np.zeros(n, dtype=bool)
        mascara[np.sort(pos.copy_to_host()[:hallados])] = True
        return mascara

    def contar_ventanas(self, raw: np.ndarray, loc: np.ndarray, w: int) -> tuple[np.ndarray, ...]:
        k = loc.shape[0]
        if k == 0:
            vacio = np.zeros(0, dtype=np.int64)
            return vacio, vacio, vacio, vacio
        d = self.cuda.to_device(np.ascontiguousarray(raw, dtype=np.uint8))
        ini = self.cuda.to_device(np.ascontiguousarray(loc, dtype=np.int64))
        salidas = [self.cuda.device_array(k, dtype=np.int64) for _ in range(4)]
        self.k["ventanas"][self._malla(k), self.motor.hilos_bloque](d, ini, w, *salidas)
        self.cuda.synchronize()
        return tuple(s.copy_to_host() for s in salidas)


class MotorGPU:
    """Motor del dispositivo 'gpu' (una tarjeta CUDA por worker)."""

    dispositivo = "gpu"
    OPCIONES = ("hilos_bloque", "bloques", "lote_mb", "streams", "vram_libre_mb", "dispositivo_cuda")
    MAX_LOTE_POR_VRAM = 0.35  # cada lote (x streams) no puede pasar de esta fraccion de la VRAM libre

    def __init__(self, hilos_bloque: int = 256, bloques: int = 1024, lote_mb: float = 64,
                 streams: int = 2, vram_libre_mb: float | None = None, dispositivo_cuda: int = 0) -> None:
        if hilos_bloque not in (32, 64, 128, 256, 512, 1024):
            raise ValueError("hilos_bloque debe ser una potencia de 2 entre 32 y 1024")
        if not 1 <= int(bloques) <= 1024:
            raise ValueError("bloques debe estar entre 1 y 1024")
        if not 0 < float(lote_mb) <= 1024:
            raise ValueError("lote_mb debe estar entre 1 y 1024")
        if int(streams) not in (1, 2):
            raise ValueError("streams debe ser 1 o 2")
        self.hilos_bloque = int(hilos_bloque)
        self.bloques = int(bloques)
        self.lote = max(4096, int(float(lote_mb) * MB))
        self.n_streams = int(streams)
        self.vram_libre_mb = vram_libre_mb
        self.dispositivo_cuda = dispositivo_cuda
        self.cuda = None
        self.nucleo: NucleoGPU | None = None
        self.preparacion_s: float | None = None
        self.info: dict = {}
        self.desbordes = 0
        self._mapas: dict[str, np.ndarray] = {}

    @property
    def nombre(self) -> str:
        return "gpu-cuda" + ("-simulador" if simulador() else "")

    def lote_valido(self, vram_libre_mb: float | None) -> None:
        """Lanza ValueError si los buffers no caben en la VRAM libre."""
        if vram_libre_mb is None or vram_libre_mb == float("inf"):
            return
        pedido = self.lote * self.n_streams / MB
        maximo = vram_libre_mb * self.MAX_LOTE_POR_VRAM
        if pedido > maximo:
            raise ValueError("El lote de %.0f MB x %d streams no cabe en la VRAM libre (%.0f MB): "
                             "use como maximo %.0f MB por lote"
                             % (self.lote / MB, self.n_streams, vram_libre_mb, maximo / self.n_streams))

    def preparar(self) -> float:
        """Reserva la tarjeta, valida el lote contra la VRAM y compila los kernels."""
        t0 = time.perf_counter()
        self.cuda = _cuda()
        if not simulador():
            self.cuda.select_device(self.dispositivo_cuda)
        vram = self.vram_libre_mb
        try:
            libre, total = self.cuda.current_context().get_memory_info()
            if vram is None and libre != float("inf"):
                vram = libre / MB
            self.info["vram_total_mb"] = None if total == float("inf") else round(total / MB)
        except Exception:
            self.info["vram_total_mb"] = None
        self.info["vram_libre_mb"] = None if vram in (None, float("inf")) else round(vram)
        self.lote_valido(vram)
        self.nucleo = NucleoGPU(self)
        self.h_buf = [self.cuda.pinned_array(self.lote, dtype=np.uint8) for _ in range(self.n_streams)]
        self.d_buf = [self.cuda.device_array(self.lote, dtype=np.uint8) for _ in range(self.n_streams)]
        self.streams = [self.cuda.stream() for _ in range(self.n_streams)]
        self.precalentar()
        self.preparacion_s = time.perf_counter() - t0
        return self.preparacion_s

    def precalentar(self) -> None:
        """Compila todos los kernels sobre datos minusculos (fuera del cronometro)."""
        muestra = np.frombuffer(b"ACGTNacgtnCGCGRYX!" * 4, dtype=np.uint8).copy()
        self.nucleo.histograma(muestra)
        self.nucleo.contar_categorias(muestra, muestra[::-1].copy())
        self.nucleo.coincidencias(muestra & 0xDF, "CGN")
        self.nucleo.contar_ventanas(muestra, np.array([0, 4], dtype=np.int64), 10)
        self.cuda.synchronize()

    def _abrir(self, ruta: str) -> np.ndarray:
        if ruta not in self._mapas:
            self._mapas[ruta] = abrir_seq(ruta)
        return self._mapas[ruta]

    # -- conteo con solapamiento -------------------------------------------
    def _conteo(self, seq: np.ndarray, inicio: int, fin: int, params: dict) -> tuple[dict, dict]:
        cuda = self.cuda
        k = kernels()["histograma"]
        d_hist = cuda.to_device(np.zeros(256, dtype=np.uint64))
        eventos = []
        t_copia_host = 0.0
        turno = 0
        lotes = 0
        inicio_ev = cuda.event()
        inicio_ev.record(self.streams[0])
        for a in range(inicio, fin, self.lote):
            n = min(self.lote, fin - a)
            s = self.streams[turno]
            s.synchronize()  # el buffer de este turno quedo libre
            t1 = time.perf_counter()
            self.h_buf[turno][:n] = seq[a:a + n]
            t_copia_host += time.perf_counter() - t1
            e0, e1, e2 = cuda.event(), cuda.event(), cuda.event()
            e0.record(s)
            self.d_buf[turno][:n].copy_to_device(self.h_buf[turno][:n], stream=s)
            e1.record(s)
            malla = max(1, min(self.bloques, -(-n // self.hilos_bloque)))
            k[malla, self.hilos_bloque, s](self.d_buf[turno], n, d_hist)
            e2.record(s)
            eventos.append((e0, e1, e2))
            lotes += 1
            turno = (turno + 1) % self.n_streams
        for s in self.streams:
            s.synchronize()
        cuda.synchronize()  # antes de parar el cronometro
        fin_ev = cuda.event()
        fin_ev.record(self.streams[0])
        fin_ev.synchronize()
        hist = d_hist.copy_to_host().astype(np.int64)
        parcial = {"hist": hist.tolist(), "invalidos_pos": []}
        from pdn.comun.formato import CLASE_INVALIDO, TABLA_CLASES  # noqa: PLC0415
        if params.get("k_invalidos") and int(hist[TABLA_CLASES == CLASE_INVALIDO].sum()):
            # Posiciones de invalidos (raras): se buscan en la CPU solo si hay
            from pdn.operaciones.conteo import posiciones_invalidos  # noqa: PLC0415
            for a in range(inicio, fin, 4 * MB):
                parcial["invalidos_pos"] += posiciones_invalidos(np.asarray(seq[a:min(a + 4 * MB, fin)]), a,
                                                                 params["k_invalidos"] - len(parcial["invalidos_pos"]))
                if len(parcial["invalidos_pos"]) >= params["k_invalidos"]:
                    break
        return parcial, self._tiempos(eventos, inicio_ev, fin_ev, lotes, t_copia_host)

    def _tiempos(self, eventos, inicio_ev, fin_ev, lotes: int, t_copia_host: float) -> dict:
        """t_transferencia, t_kernel y porcentaje de solapamiento (None si no se puede medir)."""
        info = {"lotes": lotes, "streams": self.n_streams, "t_copia_host_s": round(t_copia_host, 6),
                "t_transferencia_s": None, "t_kernel_s": None, "t_gpu_s": None, "solapamiento_pct": None}
        if simulador():
            info["motivo_sin_tiempos"] = "simulador de CUDA: los tiempos de eventos no son reales"
            return info
        try:
            transf = sum(e0.elapsed_time(e1) for e0, e1, _ in eventos) / 1000
            kern = sum(e1.elapsed_time(e2) for _, e1, e2 in eventos) / 1000
            total = inicio_ev.elapsed_time(fin_ev) / 1000
        except Exception:  # el simulador no mide tiempos de eventos
            return info
        info.update({"t_transferencia_s": round(transf, 6), "t_kernel_s": round(kern, 6),
                     "t_gpu_s": round(total, 6)})
        menor = min(transf, kern)
        if menor > 0:
            info["solapamiento_pct"] = round(max(0.0, min(1.0, (transf + kern - total) / menor)) * 100, 1)
        return info

    # -- interfaz de motor ---------------------------------------------------
    def procesar(self, nombre: str, params: dict, ruta_a: str, ruta_b: str | None,
                 carga: dict) -> tuple[dict, dict]:
        """Procesa una tarea en la GPU. Devuelve (parcial, info)."""
        if self.nucleo is None:
            raise RuntimeError("El motor GPU no esta preparado: llame a preparar() antes")
        t0 = time.perf_counter()
        seq_a = self._abrir(ruta_a)
        if nombre == "conteo":
            parcial, info = self._conteo(seq_a, carga["inicio"], carga["fin"], params)
        else:
            op = operacion(nombre)
            if nombre == "comparacion":
                parcial = op.procesar(seq_a, self._abrir(ruta_b), carga["segmentos"], params, self.nucleo)
            else:
                parcial = op.procesar(seq_a, carga["inicio"], carga["fin"], params, carga["limites"],
                                      self.nucleo)
            self.cuda.synchronize()
            info = {}
        info.update({"t_calculo": time.perf_counter() - t0, "impl": self.nombre,
                     "desbordes_cpu": self.desbordes})
        return parcial, info

    def cerrar(self) -> None:
        self._mapas.clear()
        self.nucleo = None

    def describir(self) -> dict:
        return {"dispositivo": "gpu", "impl": self.nombre, "hilos_bloque": self.hilos_bloque,
                "bloques": self.bloques, "lote_mb": self.lote / MB, "streams": self.n_streams,
                "preparacion_s": self.preparacion_s, "simulador": simulador(), **self.info}
