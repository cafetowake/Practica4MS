"""
Universidad del Valle de Guatemala
Facultad de Ingenieria
Ciencia de la Computacion y Tecnologias de la Informacion
CC2017 - Modelacion y Simulacion
Seccion 30
Semestre 2, 2026

Actividad - Practica 4 Ejercicio aplicado, Logistica del Valle
Modulo: simulacion - politicas de ruta y simulacion de eventos discretos

Integrantes:
Nombre: Paula De Leon
Carnet: 23202

Fecha: 22/10/2026

Ejecucion:
Este modulo no se ejecuta solo, lo importan experimentos.py, figuras.py y la
aplicacion. Se prueba con python -m pytest -q test_nucleo.py

Descripcion:
Define las politicas de ruta que se comparan, la generacion de pedidos como un
proceso de Poisson con destinos uniformes y la simulacion de eventos discretos
con cola FIFO. Cada pedido pasa por llegada, espera, viaje de ida, entrega y
regreso del vehiculo a la bodega. La recepcion se cierra a la hora indicada
pero la simulacion continua hasta atender todos los pedidos recibidos. Incluye
un segundo modelo independiente, basado en el instante de disponibilidad de
cada vehiculo, que sirve para verificar la simulacion principal.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from collections.abc import Callable
from typing import TypedDict

import numpy as np

from algoritmos import dijkstra
from grafo import BODEGA, DESTINOS, RUTAS_ACTUALES, Grafo, tiempo_ruta


# politicas
class RutaDestino(TypedDict):
    """Ruta de ida y de regreso asignada a un destino."""
    ida: list[str]
    vuelta: list[str]
    t_ida: float
    t_vuelta: float
    desvio: bool      # la ruta de la hoja de despacho quedó cerrada y se sustituyó
    atendible: bool   # existe camino de ida y de regreso


Politica = dict[str, RutaDestino]
Pedido = tuple[float, str]
Traza = list[tuple[float, int]]


def _camino_minimo(grafo: Grafo, origen: str, destino: str) -> tuple[float, list[str]]:
    """Dijkstra con salida uniforme. Sin camino devuelve infinito y lista vacía."""
    t, ruta, _ = dijkstra(grafo, origen, destino)
    return (t, ruta) if ruta is not None else (float("inf"), [])


def _entrada(ida: list[str], vuelta: list[str], t_ida: float, t_vuelta: float,
             desvio: bool = False) -> RutaDestino:
    return RutaDestino(ida=ida, vuelta=vuelta, t_ida=t_ida, t_vuelta=t_vuelta, desvio=desvio,
                       atendible=math.isfinite(t_ida) and math.isfinite(t_vuelta))


def politica_actual(grafo: Grafo) -> Politica:
    """Rutas de las hojas de despacho con regreso por el mismo camino.

    Si algún tramo de la hoja fue cerrado, el conductor no puede seguirla y se
    le asigna el camino mínimo de la red vigente en ese sentido. Esa ruta se
    marca como desvío.
    """
    pol: Politica = {}
    for d in DESTINOS:
        ida = RUTAS_ACTUALES[d]
        vuelta = ida[::-1]
        t_ida = tiempo_ruta(grafo, ida)
        t_vuelta = tiempo_ruta(grafo, vuelta)
        desvio = False
        if t_ida is None:
            t_ida, ida = _camino_minimo(grafo, BODEGA, d)
            desvio = True
        if t_vuelta is None:
            t_vuelta, vuelta = _camino_minimo(grafo, d, BODEGA)
            desvio = True
        pol[d] = _entrada(ida, vuelta, t_ida, t_vuelta, desvio)
    return pol


def politica_recomendada(grafo: Grafo) -> Politica:
    """Dijkstra independiente para la ida y para el regreso."""
    pol: Politica = {}
    for d in DESTINOS:
        t_ida, ida = _camino_minimo(grafo, BODEGA, d)
        t_vuelta, vuelta = _camino_minimo(grafo, d, BODEGA)
        pol[d] = _entrada(ida, vuelta, t_ida, t_vuelta)
    return pol


def politica_reversible(grafo: Grafo) -> Politica:
    """Mejor ruta cuyo regreso es la misma ruta invertida.

    Se aplica Dijkstra sobre un grafo auxiliar con sólo las calles de doble
    sentido, donde cada tramo pesa la suma de sus tiempos de ida y de regreso.
    """
    ady: dict[str, dict[str, float]] = {n: {} for n in grafo.ady}
    for u, v, w in grafo.aristas():
        w2 = grafo.peso(v, u)
        if w2 is not None:
            ady[u][v] = w + w2
    auxiliar = Grafo(ady)
    pol: Politica = {}
    for d in DESTINOS:
        _, ida = _camino_minimo(auxiliar, BODEGA, d)
        vuelta = ida[::-1]
        t_ida = tiempo_ruta(grafo, ida) if ida else None
        t_vuelta = tiempo_ruta(grafo, vuelta) if vuelta else None
        pol[d] = _entrada(ida, vuelta,
                          float("inf") if t_ida is None else t_ida,
                          float("inf") if t_vuelta is None else t_vuelta)
    return pol


POLITICAS: dict[str, Callable[[Grafo], Politica]] = {
    "actual": politica_actual,
    "recomendada": politica_recomendada,
    "reversible": politica_reversible,
}


# pedidos
def generar_pedidos(lam_hora: float, horas: float, rng: np.random.Generator) -> list[Pedido]:
    """Genera los pedidos de una jornada como un proceso de Poisson.

    Los tiempos entre llegadas son exponenciales y se obtienen por transformada
    inversa, X = -ln(1 - U) / lambda, con lambda en pedidos por minuto. El
    destino se asigna con una uniforme discreta, índice = parte entera de 4V.
    Devuelve una lista de pares (instante de llegada en minutos, destino).
    """
    tasa = lam_hora / 60.0
    limite = horas * 60.0
    pedidos: list[Pedido] = []
    if tasa <= 0:
        return pedidos
    t = 0.0
    while True:
        t += -math.log(1.0 - rng.random()) / tasa
        if t >= limite:
            break
        destino = DESTINOS[int(rng.random() * len(DESTINOS))]
        pedidos.append((t, destino))
    return pedidos


# simulacion
LLEGADA, ENTREGA, REGRESO = 0, 1, 2


def simular(pedidos: list[Pedido], politica: Politica, vehiculos: int, horas: float = 8.0,
            descarga: float = 0.0, umbral: float = 75.0) -> tuple[dict[str, float], Traza]:
    """Simulación de eventos discretos con cola FIFO.

    Eventos
      LLEGADA  el pedido entra a la cola y se intenta despachar.
      ENTREGA  el vehículo llega al destino y completa la entrega.
      REGRESO  el vehículo vuelve a N01 y queda disponible.

    La recepción se cierra en horas * 60 pero la simulación sigue hasta que se
    entregan todos los pedidos recibidos y los vehículos regresan. Los pedidos
    a un destino sin camino de ida y regreso no se atienden y se cuentan aparte.

    Devuelve un diccionario de métricas y la traza de la longitud de la cola.
    """
    cola: deque[int] = deque()
    libres = vehiculos
    eventos: list[tuple[float, int, int, int]] = []
    seq = 0
    n = len(pedidos)
    no_atendidos = 0
    for i, (t, d) in enumerate(pedidos):
        if not politica[d]["atendible"]:
            no_atendidos += 1
            continue
        heapq.heappush(eventos, (t, seq, LLEGADA, i))
        seq += 1

    inicio: dict[int, float] = {}
    entrega: dict[int, float] = {}
    ocupado = 0.0
    ultimo_regreso = 0.0
    ultima_entrega = 0.0

    # longitud de la cola ponderada en el tiempo
    t_prev = 0.0
    area_cola = 0.0
    max_cola = 0
    traza: Traza = [(0.0, 0)]

    def despachar(t: float) -> None:
        nonlocal libres, seq, ocupado
        while cola and libres > 0:
            i = cola.popleft()
            libres -= 1
            ruta = politica[pedidos[i][1]]
            inicio[i] = t
            t_ent = t + ruta["t_ida"] + descarga
            t_reg = t_ent + ruta["t_vuelta"]
            heapq.heappush(eventos, (t_ent, seq, ENTREGA, i))
            seq += 1
            heapq.heappush(eventos, (t_reg, seq, REGRESO, i))
            seq += 1
            ocupado += t_reg - t

    while eventos:
        t, _, tipo, i = heapq.heappop(eventos)
        area_cola += len(cola) * (t - t_prev)
        t_prev = t
        if tipo == LLEGADA:
            cola.append(i)
            max_cola = max(max_cola, len(cola))
            despachar(t)
        elif tipo == ENTREGA:
            entrega[i] = t
            ultima_entrega = max(ultima_entrega, t)
        else:
            libres += 1
            ultimo_regreso = max(ultimo_regreso, t)
            despachar(t)
        traza.append((t, len(cola)))

    atendidos = len(entrega)
    fin = max(horas * 60.0, ultimo_regreso)
    base = {"pedidos": float(n), "atendidos": float(atendidos), "no_atendidos": float(no_atendidos)}
    if atendidos == 0:
        vacio = dict(puntuales=0.0, pct_puntuales=0.0 if n else float("nan"), espera_prom=0.0,
                     espera_max=0.0, solicitud_entrega_prom=0.0, utilizacion=0.0, extra_entrega=0.0,
                     extra_regreso=0.0, cola_prom=0.0, cola_max=0.0)
        return {**base, **vacio}, traza

    servidos = sorted(entrega)
    esperas = np.array([inicio[i] - pedidos[i][0] for i in servidos])
    totales = np.array([entrega[i] - pedidos[i][0] for i in servidos])
    puntuales = int(np.sum(totales <= umbral + 1e-9))
    metricas = {
        **base,
        "puntuales": float(puntuales),
        "pct_puntuales": 100.0 * puntuales / n,   # sobre los pedidos recibidos
        "espera_prom": float(esperas.mean()),
        "espera_max": float(esperas.max()),
        "solicitud_entrega_prom": float(totales.mean()),
        "utilizacion": 100.0 * ocupado / (vehiculos * fin),
        "extra_entrega": max(0.0, ultima_entrega - horas * 60.0),
        "extra_regreso": max(0.0, ultimo_regreso - horas * 60.0),
        "cola_prom": area_cola / fin,
        "cola_max": float(max_cola),
    }
    return metricas, traza


def simular_rapido(pedidos: list[Pedido], politica: Politica, vehiculos: int,
                   descarga: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Verificación independiente con un heap de disponibilidad de vehículos.

    Con cola FIFO cada pedido toma el primer vehículo que quede libre. Devuelve
    las esperas y los tiempos de solicitud a entrega para contrastarlos con los
    de simular().
    """
    libres = [0.0] * vehiculos
    heapq.heapify(libres)
    esperas: list[float] = []
    totales: list[float] = []
    for t, d in pedidos:
        ruta = politica[d]
        if not ruta["atendible"]:
            continue
        disponible = heapq.heappop(libres)
        ini = max(t, disponible)
        ent = ini + ruta["t_ida"] + descarga
        heapq.heappush(libres, ent + ruta["t_vuelta"])
        esperas.append(ini - t)
        totales.append(ent - t)
    return np.array(esperas), np.array(totales)
