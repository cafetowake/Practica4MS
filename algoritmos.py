"""
Universidad del Valle de Guatemala
Facultad de Ingenieria
Ciencia de la Computacion y Tecnologias de la Informacion
CC2017 - Modelacion y Simulacion
Seccion 30
Semestre 2, 2026

Actividad - Practica 4 Ejercicio aplicado, Logistica del Valle
Modulo: algoritmos - busqueda de rutas de tiempo minimo

Integrantes:
Nombre: Paula De Leon
Carnet: 23202

Fecha: 22/10/2026

Ejecucion:
Este modulo no se ejecuta solo, lo importan simulacion.py, experimentos.py y
la aplicacion. Se prueba con python -m pytest -q test_nucleo.py

Descripcion:
Implementa Dijkstra, que es el algoritmo elegido para la aplicacion, A* con
una heuristica propia basada en el numero minimo de tramos, Bellman-Ford como
contraste y el metodo de Yen para obtener las k rutas simples mas cortas, de
donde salen las rutas alternativas. Todas las funciones de busqueda devuelven
el tiempo, la ruta y una medida del esfuerzo realizado, de modo que se pueden
comparar entre si.
"""

from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Callable, Collection

from grafo import Grafo, tiempo_ruta

Heuristica = Callable[[str], float]
Resultado = tuple[float, list[str] | None, int]


def _reconstruir(prev: dict[str, str], origen: str, destino: str) -> list[str] | None:
    """Reconstruye la ruta siguiendo los predecesores desde el destino."""
    if destino != origen and destino not in prev:
        return None
    ruta = [destino]
    while ruta[-1] != origen:
        ruta.append(prev[ruta[-1]])
    return ruta[::-1]


def dijkstra(grafo: Grafo, origen: str, destino: str,
             nodos_vetados: Collection[str] = (),
             aristas_vetadas: Collection[tuple[str, str]] = ()) -> Resultado:
    """Camino de tiempo mínimo con cola de prioridad.

    Los parámetros de vetos los usa el método de Yen para excluir nodos y
    arcos. Devuelve el tiempo, la ruta y la cantidad de nodos expandidos. Si el
    destino no es alcanzable devuelve infinito y ruta None.
    """
    dist: dict[str, float] = {origen: 0.0}
    prev: dict[str, str] = {}
    cola: list[tuple[float, str]] = [(0.0, origen)]
    hechos: set[str] = set()
    while cola:
        d, u = heapq.heappop(cola)
        if u in hechos:
            continue
        hechos.add(u)
        if u == destino:
            return d, _reconstruir(prev, origen, destino), len(hechos)
        for v, w in grafo.ady[u].items():
            if v in nodos_vetados or (u, v) in aristas_vetadas or v in hechos:
                continue
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(cola, (nd, v))
    return float("inf"), None, len(hechos)


def saltos_minimos_hacia(grafo: Grafo, destino: str) -> dict[str, int]:
    """Número mínimo de tramos de cada nodo al destino.

    Se hace una búsqueda en anchura sobre el grafo con los arcos invertidos.
    """
    inv: dict[str, list[str]] = {n: [] for n in grafo.ady}
    for u, v, _ in grafo.aristas():
        inv[v].append(u)
    saltos = {destino: 0}
    cola = deque([destino])
    while cola:
        x = cola.popleft()
        for y in inv[x]:
            if y not in saltos:
                saltos[y] = saltos[x] + 1
                cola.append(y)
    return saltos


def heuristica_saltos(grafo: Grafo, destino: str) -> Heuristica:
    """Heurística h(n) = s(n) * w_min para A*.

    s(n) es el número mínimo de tramos de n al destino y w_min el menor tiempo
    de un tramo en la red. Es admisible porque cualquier ruta usa al menos s(n)
    tramos y cada uno cuesta al menos w_min. Es consistente porque h baja como
    máximo w_min por cada tramo recorrido.
    """
    w_min = min(w for _, _, w in grafo.aristas())
    saltos = saltos_minimos_hacia(grafo, destino)
    return lambda n: saltos.get(n, float("inf")) * w_min


def a_estrella(grafo: Grafo, origen: str, destino: str, h: Heuristica | None = None) -> Resultado:
    """A* con la heurística de saltos por defecto. Devuelve tiempo, ruta y expandidos."""
    if h is None:
        h = heuristica_saltos(grafo, destino)
    g: dict[str, float] = {origen: 0.0}
    prev: dict[str, str] = {}
    cola: list[tuple[float, float, str]] = [(h(origen), 0.0, origen)]
    hechos: set[str] = set()
    while cola:
        _, gu, u = heapq.heappop(cola)
        if u in hechos:
            continue
        hechos.add(u)
        if u == destino:
            return gu, _reconstruir(prev, origen, destino), len(hechos)
        for v, w in grafo.ady[u].items():
            if v in hechos:
                continue
            ng = gu + w
            if ng < g.get(v, float("inf")):
                g[v] = ng
                prev[v] = u
                heapq.heappush(cola, (ng + h(v), ng, v))
    return float("inf"), None, len(hechos)


def bellman_ford(grafo: Grafo, origen: str, destino: str) -> Resultado:
    """Bellman-Ford. Devuelve tiempo, ruta y número de relajaciones realizadas."""
    dist = {n: float("inf") for n in grafo.ady}
    dist[origen] = 0.0
    prev: dict[str, str] = {}
    aristas = grafo.aristas()
    relajaciones = 0
    for _ in range(len(dist) - 1):
        cambio = False
        for u, v, w in aristas:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                cambio = True
                relajaciones += 1
        if not cambio:
            break
    if dist[destino] == float("inf"):
        return float("inf"), None, relajaciones
    return dist[destino], _reconstruir(prev, origen, destino), relajaciones


def yen(grafo: Grafo, origen: str, destino: str, k: int = 5) -> list[tuple[float, list[str]]]:
    """Las k rutas simples más cortas según el método de Yen.

    Devuelve una lista de pares (tiempo, ruta) ordenada por tiempo.
    """
    t0, r0, _ = dijkstra(grafo, origen, destino)
    if r0 is None:
        return []
    A: list[tuple[float, list[str]]] = [(t0, r0)]
    B: list[tuple[float, list[str]]] = []
    for _ in range(1, k):
        ultima = A[-1][1]
        for i in range(len(ultima) - 1):
            raiz = ultima[: i + 1]
            nodo_desvio = ultima[i]
            vet_aristas = {(r[i], r[i + 1]) for _, r in A if r[: i + 1] == raiz and len(r) > i + 1}
            vet_nodos = set(raiz[:-1])
            _, cola, _ = dijkstra(grafo, nodo_desvio, destino, vet_nodos, vet_aristas)
            if cola is None:
                continue
            candidata = raiz[:-1] + cola
            t = tiempo_ruta(grafo, candidata)
            if t is None:
                continue
            if all(candidata != c[1] for c in B) and all(candidata != a[1] for a in A):
                B.append((t, candidata))
        if not B:
            break
        B.sort(key=lambda x: x[0])
        A.append(B.pop(0))
    return A
