"""Estado del Master: workers, tareas y corridas (CONTEXTO.md, secciones 7.1 y 7.4)."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

PENDIENTE = "PENDIENTE"
ASIGNADA = "ASIGNADA"
HECHA = "HECHA"
DESCARTADA = "DESCARTADA"  # devuelta a la fila; su rango vive en otra tarea

# Estados de corrida
PREPARANDO = "PREPARANDO"
EJECUTANDO = "EJECUTANDO"
TERMINADA = "TERMINADA"
CANCELADA = "CANCELADA"
FALLIDA = "FALLIDA"


@dataclass
class WorkerInfo:
    """Lo que el Master sabe de un worker."""

    wid: str
    hostname: str
    dispositivo: str
    identidad: bytes = b""
    identidad_latido: bytes = b""
    hardware: dict = field(default_factory=dict)
    archivos: dict = field(default_factory=dict)
    version: str = ""
    t_registro: float = field(default_factory=time.time)
    t_ultimo_latido: float = field(default_factory=time.time)
    t_ultimo_mensaje: float = field(default_factory=time.time)
    estado: str = "conectado"  # conectado | perdido | sospechoso
    metricas: dict = field(default_factory=dict)
    calibracion_sintetica: dict = field(default_factory=dict)
    corrida_lista: str | None = None
    corrida_preparando: str | None = None
    calibracion_corrida: float | None = None
    preparacion_s: float | None = None
    motor: dict = field(default_factory=dict)
    rechazos: int = 0
    errores: int = 0
    excluido: str | None = None  # motivo de exclusion en la corrida actual
    bytes_corrida: int = 0
    tareas_corrida: int = 0
    reconexiones: int = 0

    def publico(self, ahora: float) -> dict:
        """Vista serializable para el dashboard."""
        d = {k: v for k, v in asdict(self).items() if k not in ("identidad", "identidad_latido")}
        d["edad_latido_s"] = round(ahora - self.t_ultimo_latido, 2)
        d["ip"] = (self.hardware.get("red") or {}).get("ip")
        return d


@dataclass
class Tarea:
    """Un tramo del espacio asignado a un worker."""

    tid: int
    u_ini: int
    u_fin: int
    inicio: int
    fin: int
    estado: str = PENDIENTE
    worker: str | None = None
    dispositivo: str | None = None
    t_asignacion: float | None = None
    t_esperado: float | None = None
    t_resultado: float | None = None
    t_calculo: float | None = None
    t_crc: float | None = None
    mb_s: float | None = None
    origen: int | None = None  # tarea de la que proviene el rango (si fue reasignada)
    motivo: str | None = None
    info: dict = field(default_factory=dict)

    @property
    def nbytes(self) -> int:
        return self.fin - self.inicio

    def fila(self, t0: float | None) -> dict:
        """Fila para tareas.csv y la linea de tiempo."""
        rel = (lambda t: round(t - t0, 4) if (t is not None and t0) else None)
        return {"tarea_id": self.tid, "worker": self.worker, "dispositivo": self.dispositivo,
                "estado": self.estado, "u_ini": self.u_ini, "u_fin": self.u_fin,
                "inicio": self.inicio, "fin": self.fin, "bytes": self.nbytes,
                "t_asignacion": rel(self.t_asignacion), "t_resultado": rel(self.t_resultado),
                "t_calculo": None if self.t_calculo is None else round(self.t_calculo, 5),
                "t_crc": None if self.t_crc is None else round(self.t_crc, 5),
                "mb_s": None if self.mb_s is None else round(self.mb_s, 2),
                "reasignada": self.origen is not None, "origen": self.origen,
                "motivo": self.motivo}


@dataclass
class Corrida:
    """Una ejecucion completa de una operacion."""

    corrida_id: str
    config: dict
    trabajo: Any
    planificador: Any
    huellas_a: dict | None = None
    huellas_b: dict | None = None
    estado: str = PREPARANDO
    t_creacion: float = field(default_factory=time.time)
    t_inicio: float | None = None
    t_fin: float | None = None
    participantes: list[str] = field(default_factory=list)
    excluidos: dict = field(default_factory=dict)
    tareas: dict[int, Tarea] = field(default_factory=dict)
    sig_tid: int = 0
    parcial: dict | None = None
    reasignaciones: list[dict] = field(default_factory=list)
    descartados: list[dict] = field(default_factory=list)
    resultado: dict | None = None
    validacion: dict | None = None
    carpeta: str | None = None
    preparacion_master_s: float = 0.0
    error: str | None = None

    def nueva_tarea(self, u_ini: int, u_fin: int, origen: int | None = None) -> Tarea:
        tam = self.trabajo.tam_unidad
        t = Tarea(self.sig_tid, u_ini, u_fin, u_ini * tam, min(u_fin * tam, self.trabajo.largo),
                  origen=origen)
        self.tareas[t.tid] = t
        self.sig_tid += 1
        return t

    def bytes_hechos(self) -> int:
        return sum(t.nbytes for t in self.tareas.values() if t.estado == HECHA)

    def en_vuelo(self) -> list[Tarea]:
        return [t for t in self.tareas.values() if t.estado == ASIGNADA]
