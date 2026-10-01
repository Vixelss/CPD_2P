"""Reparto estatico del modo MPI (CONTEXTO.md, seccion 13.1).

'iguales': cada rank recibe el mismo numero de unidades (+-1).
'proporcional': cada rank recibe unidades en proporcion a la velocidad
calibrada de su nodo (guardada por el modo dinamico).
Siempre en rangos contiguos de unidades completas, sin huecos ni solapes.
"""

from __future__ import annotations


def repartir(n_unidades: int, pesos: list[float]) -> list[tuple[int, int]]:
    """Rangos contiguos [u_ini, u_fin) proporcionales a los pesos."""
    n = len(pesos)
    if n == 0:
        raise ValueError("hace falta al menos un rank")
    pesos = [max(float(p), 0.0) for p in pesos]
    total = sum(pesos)
    if total <= 0:
        pesos = [1.0] * n
        total = float(n)
    cortes = [0]
    acumulado = 0.0
    for p in pesos[:-1]:
        acumulado += p
        cortes.append(min(n_unidades, round(n_unidades * acumulado / total)))
    cortes.append(n_unidades)
    for i in range(1, len(cortes)):
        cortes[i] = max(cortes[i], cortes[i - 1])
    return [(cortes[i], cortes[i + 1]) for i in range(n)]


def pesos_para(hosts: list[str], estrategia: str, velocidades: dict | None = None) -> list[float]:
    """Peso de cada rank segun su host y la estrategia."""
    if estrategia == "iguales":
        return [1.0] * len(hosts)
    if estrategia != "proporcional":
        raise ValueError("Reparto MPI desconocido: %r (iguales o proporcional)" % estrategia)
    if not velocidades:
        raise ValueError("El reparto proporcional necesita velocidades calibradas (corra antes el modo dinamico)")
    faltan = sorted({h for h in hosts if h not in velocidades})
    if faltan:
        raise ValueError("No hay velocidad calibrada para: %s" % ", ".join(faltan))
    return [float(velocidades[h]) for h in hosts]
