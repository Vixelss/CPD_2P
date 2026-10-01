"""Planificador del modo dinamico: raciones adaptativas (CONTEXTO.md, seccion 7.3).

Adapta referencias/P1.3/motor_hibrido.py al nivel del cluster:
- El espacio se divide en unidades; la fila guarda rangos de unidades pendientes.
- Cada worker tiene una velocidad (bytes/s) inicializada con su calibracion y
  actualizada con una media movil exponencial de sus resultados.
- El tamano de cada tarea es lo que el worker procesaria en el tiempo
  objetivo, en unidades completas, entre 1 y un maximo por dispositivo.
- Todos empiezan con una racion minima (1 unidad): asi nadie se lleva trozos
  grandes antes de haber demostrado nada (leccion 4 del P1).
- Fase final decreciente: cuando lo pendiente es menor que
  (suma de velocidades x objetivo x 2), las tareas se limitan a
  pendiente / (2 x workers activos), para que todos terminen casi a la vez.
- Estrategias: adaptativa (por defecto), fija, proporcional.
"""

from __future__ import annotations

import bisect

ESTRATEGIAS = ("adaptativa", "fija", "proporcional")
# Preferencia de dispositivos para la operacion de zonas (la de la NPU)
PREFERENCIA_ZONAS = {"npu": 0, "gpu": 1, "opencl": 2, "cpu": 3}


class Planificador:
    """Decide que rango de unidades recibe cada worker."""

    def __init__(self, n_unidades: int, tam_unidad: int, largo: int, estrategia: str = "adaptativa",
                 tiempo_objetivo: float = 0.5, max_bytes: int = 128 << 20,
                 max_bytes_gpu: int = 512 << 20, tam_fijo: int = 16 << 20, alfa: float = 0.3,
                 operacion: str = "") -> None:
        if estrategia not in ESTRATEGIAS:
            raise ValueError("Estrategia desconocida: %r (validas: %s)" % (estrategia, ", ".join(ESTRATEGIAS)))
        if not 0.1 <= tiempo_objetivo <= 5:
            raise ValueError("El tiempo objetivo debe estar entre 0,1 y 5 s")
        self.n_unidades = n_unidades
        self.tam_unidad = tam_unidad
        self.largo = largo
        self.estrategia = estrategia
        self.tiempo_objetivo = tiempo_objetivo
        self.max_unidades = max(1, max_bytes // tam_unidad)
        self.max_unidades_gpu = max(1, max_bytes_gpu // tam_unidad)
        self.fijo_unidades = max(1, tam_fijo // tam_unidad)
        self.alfa = alfa
        self.operacion = operacion
        self.pendientes: list[list[int]] = [[0, n_unidades]] if n_unidades else []
        self.velocidad: dict[str, float] = {}
        self.dispositivo: dict[str, str] = {}
        self.con_resultado: set[str] = set()
        self.reservas: dict[str, list[int]] = {}

    # -- velocidades -------------------------------------------------------
    def registrar_worker(self, wid: str, dispositivo: str, velocidad: float | None) -> None:
        """Da de alta un worker con su velocidad calibrada (bytes/s)."""
        self.dispositivo[wid] = dispositivo
        if velocidad and velocidad > 0:
            self.velocidad[wid] = float(velocidad)

    def actualizar_velocidad(self, wid: str, nbytes: int, segundos: float) -> None:
        """Media movil exponencial de la velocidad medida."""
        if segundos <= 0 or nbytes <= 0:
            return
        medida = nbytes / segundos
        previa = self.velocidad.get(wid)
        self.velocidad[wid] = medida if previa is None else self.alfa * medida + (1 - self.alfa) * previa
        self.con_resultado.add(wid)

    # -- fila --------------------------------------------------------------
    def bytes_de(self, u_ini: int, u_fin: int) -> int:
        return min(u_fin * self.tam_unidad, self.largo) - u_ini * self.tam_unidad

    def pendiente_bytes(self) -> int:
        return sum(self.bytes_de(a, b) for a, b in self.pendientes) + \
            sum(self.bytes_de(a, b) for a, b in self.reservas.values())

    def pendiente_unidades(self) -> int:
        return sum(b - a for a, b in self.pendientes) + sum(b - a for a, b in self.reservas.values())

    def devolver(self, u_ini: int, u_fin: int) -> None:
        """Devuelve un rango a la fila (fusiona con los vecinos)."""
        if u_fin <= u_ini:
            return
        i = bisect.bisect_left(self.pendientes, [u_ini, u_fin])
        self.pendientes.insert(i, [u_ini, u_fin])
        fusion: list[list[int]] = []
        for a, b in self.pendientes:
            if fusion and a <= fusion[-1][1]:
                fusion[-1][1] = max(fusion[-1][1], b)
            else:
                fusion.append([a, b])
        self.pendientes = fusion

    def _tomar(self, unidades: int) -> tuple[int, int]:
        a, b = self.pendientes[0]
        fin = min(b, a + unidades)
        if fin == b:
            self.pendientes.pop(0)
        else:
            self.pendientes[0][0] = fin
        return a, fin

    # -- decision ----------------------------------------------------------
    def iniciar_proporcional(self, workers: list[str]) -> None:
        """Reparto inicial unico segun calibracion: un bloque contiguo por worker."""
        if not workers:
            return
        vel = [max(self.velocidad.get(w, 0.0), 1e-9) for w in workers]
        total = sum(vel)
        a = 0
        acumulado = 0.0
        for w, v in zip(workers, vel):
            acumulado += v
            b = round(self.n_unidades * acumulado / total) if w != workers[-1] else self.n_unidades
            if b > a:
                self.reservas[w] = [a, b]
            a = b
        self.pendientes = []

    def _tam_adaptativo(self, wid: str, activos: list[str]) -> int:
        if wid not in self.con_resultado or wid not in self.velocidad:
            return 1  # racion minima hasta demostrar velocidad
        maximo = self.max_unidades_gpu if self.dispositivo.get(wid) == "gpu" else self.max_unidades
        unidades = int(self.velocidad[wid] * self.tiempo_objetivo) // self.tam_unidad
        unidades = max(1, min(maximo, unidades))
        # Fase final decreciente
        suma = sum(self.velocidad.get(w, 0.0) for w in activos)
        pendiente = self.pendiente_bytes()
        if pendiente < suma * self.tiempo_objetivo * 2 and activos:
            limite = pendiente // (2 * len(activos))
            unidades = min(unidades, max(1, limite // self.tam_unidad))
        return unidades

    def _cede_por_preferencia(self, wid: str, activos: list[str]) -> bool:
        # En zonas, los dispositivos no preferidos ceden el final a los preferidos
        if self.operacion != "zonas":
            return False
        mio = PREFERENCIA_ZONAS.get(self.dispositivo.get(wid, "cpu"), 9)
        mejores = [w for w in activos if PREFERENCIA_ZONAS.get(self.dispositivo.get(w, "cpu"), 9) < mio]
        if not mejores:
            return False
        capacidad = sum(self.velocidad.get(w, 0.0) for w in mejores) * self.tiempo_objetivo * 4
        return self.pendiente_bytes() <= capacidad

    def pedir(self, wid: str, activos: list[str]) -> tuple[int, int] | None:
        """Rango [u_ini, u_fin) para el worker, o None si no hay nada para el ahora."""
        if wid in self.reservas:
            return tuple(self.reservas.pop(wid))  # type: ignore[return-value]
        if not self.pendientes:
            return None
        if self._cede_por_preferencia(wid, activos):
            return None
        if self.estrategia == "fija":
            return self._tomar(self.fijo_unidades)
        if self.estrategia == "proporcional":
            # Solo quedan bloques devueltos por fallos: se entregan completos
            a, b = self.pendientes[0]
            return self._tomar(b - a)
        return self._tomar(self._tam_adaptativo(wid, activos))

    def liberar_reserva(self, wid: str) -> None:
        """Si un worker con reserva se pierde, su bloque vuelve a la fila."""
        if wid in self.reservas:
            a, b = self.reservas.pop(wid)
            self.devolver(a, b)

    def terminado(self) -> bool:
        return not self.pendientes and not self.reservas
