"""
Universidad del Valle de Guatemala
Facultad de Ingenieria
Ciencia de la Computacion y Tecnologias de la Informacion
CC2017 - Modelacion y Simulacion
Seccion 30
Semestre 2, 2026

Actividad - Practica 4 Ejercicio aplicado, Logistica del Valle
Modulo: test_nucleo - pruebas del grafo, los algoritmos y la simulacion

Integrantes:
Nombre: Paula De Leon
Carnet: 23202

Fecha: 22/10/2026

Ejecucion:
python -m pytest -q test_nucleo.py

Descripcion:
Pruebas automaticas del nucleo del proyecto. Comprueban el tamano de la red, la
validez de las rutas de las hojas de despacho, que Dijkstra, A* y Bellman-Ford
den el mismo tiempo minimo, que la heuristica sea admisible, el efecto de cerrar
una conexion, el comportamiento de la cola con casos que se calculan a mano, la
reproducibilidad de las semillas, la continuacion de la simulacion despues de las
ocho horas y el aviso cuando un destino queda sin camino. Tambien contrastan la
simulacion de eventos con el modelo independiente de verificacion.
"""

from __future__ import annotations

import numpy as np
import pytest

from algoritmos import a_estrella, bellman_ford, dijkstra, heuristica_saltos, yen
from experimentos import ESCENARIOS, correr, diferencias_pareadas, semillas
from grafo import BODEGA, DESTINOS, RUTAS_ACTUALES, Grafo, tiempo_ruta, validar_ruta
from simulacion import (POLITICAS, generar_pedidos, politica_actual, politica_recomendada,
                        politica_reversible, simular, simular_rapido)


@pytest.fixture(scope="module")
def red() -> Grafo:
    """Red original. Las pruebas que la modifican trabajan sobre una copia."""
    return Grafo.desde_csv()


# grafo
def test_tamano_de_la_red(red: Grafo):
    assert len(red.nodos()) == 40
    assert len(red.aristas()) == 144


def test_asimetrias_de_la_red(red: Grafo):
    un_sentido, tiempos_distintos = red.es_asimetrico()
    assert len(un_sentido) == 18
    assert len(tiempos_distintos) == 14


def test_rutas_actuales_validas_ida_y_regreso(red: Grafo):
    for ruta in RUTAS_ACTUALES.values():
        assert validar_ruta(red, ruta)[0]
        assert validar_ruta(red, ruta[::-1])[0]


def test_tiempos_de_las_rutas_actuales(red: Grafo):
    esperado = {"N08": 59, "N19": 33, "N34": 42, "N40": 93}
    for destino, minutos in esperado.items():
        assert tiempo_ruta(red, RUTAS_ACTUALES[destino]) == minutos


def test_una_ruta_con_tramo_inexistente_es_invalida(red: Grafo):
    valida, tiempo, faltantes = validar_ruta(red, ["N01", "N40"])
    assert not valida and tiempo is None and faltantes == [("N01", "N40")]


def test_cerrar_y_fijar_tiempo_no_tocan_la_red_base(red: Grafo):
    copia = red.copia()
    copia.cerrar("N02", "N03")
    copia.fijar_tiempo("N01", "N02", 30)
    assert red.peso("N02", "N03") == 8 and red.peso("N01", "N02") == 8
    assert copia.peso("N02", "N03") is None and copia.peso("N01", "N02") == 30


# algoritmos
def test_tres_algoritmos_coinciden(red: Grafo):
    for n in red.nodos():
        for origen, destino in ((BODEGA, n), (n, BODEGA)):
            if origen != destino:
                tiempos = [f(red, origen, destino)[0] for f in (dijkstra, a_estrella, bellman_ford)]
                assert max(tiempos) == pytest.approx(min(tiempos))


def test_la_ruta_devuelta_suma_el_tiempo_informado(red: Grafo):
    for destino in DESTINOS:
        tiempo, ruta, _ = dijkstra(red, BODEGA, destino)
        assert ruta is not None and tiempo_ruta(red, ruta) == pytest.approx(tiempo)


def test_la_heuristica_es_admisible(red: Grafo):
    for destino in DESTINOS:
        h = heuristica_saltos(red, destino)
        for n in red.nodos():
            if n != destino:
                assert h(n) <= dijkstra(red, n, destino)[0] + 1e-9


def test_a_estrella_expande_menos_nodos_que_dijkstra(red: Grafo):
    dij = sum(dijkstra(red, BODEGA, d)[2] for d in DESTINOS)
    ast = sum(a_estrella(red, BODEGA, d)[2] for d in DESTINOS)
    assert ast < dij


def test_yen_ordena_las_rutas_y_la_primera_es_el_minimo(red: Grafo):
    for destino in DESTINOS:
        rutas = yen(red, BODEGA, destino, 4)
        tiempos = [t for t, _ in rutas]
        assert tiempos == sorted(tiempos)
        assert tiempos[0] == pytest.approx(dijkstra(red, BODEGA, destino)[0])
        assert len({tuple(r) for _, r in rutas}) == len(rutas)


def test_destino_inalcanzable_devuelve_infinito(red: Grafo):
    copia = red.copia()
    for u, v, _ in [a for a in copia.aristas() if a[1] == "N34"]:
        copia.cerrar(u, v)
    tiempo, ruta, _ = dijkstra(copia, BODEGA, "N34")
    assert tiempo == float("inf") and ruta is None


# politicas
def test_politica_actual_usa_las_hojas_y_regresa_por_el_mismo_camino(red: Grafo):
    pol = politica_actual(red)
    for destino in DESTINOS:
        assert pol[destino]["ida"] == RUTAS_ACTUALES[destino]
        assert pol[destino]["vuelta"] == RUTAS_ACTUALES[destino][::-1]
        assert not pol[destino]["desvio"]


def test_politica_recomendada_no_empeora_a_la_actual(red: Grafo):
    actual, recomendada = politica_actual(red), politica_recomendada(red)
    for destino in DESTINOS:
        assert recomendada[destino]["t_ida"] <= actual[destino]["t_ida"]
        assert recomendada[destino]["t_vuelta"] <= actual[destino]["t_vuelta"]


def test_politica_reversible_regresa_por_la_ruta_invertida(red: Grafo):
    pol = politica_reversible(red)
    for destino in DESTINOS:
        assert pol[destino]["vuelta"] == pol[destino]["ida"][::-1]
        assert validar_ruta(red, pol[destino]["vuelta"])[0]


def test_cierre_desvia_la_ruta_actual_y_la_deja_valida(red: Grafo):
    copia = red.copia()
    copia.cerrar("N02", "N03")
    pol = politica_actual(copia)
    assert pol["N08"]["desvio"] is True
    assert validar_ruta(copia, pol["N08"]["ida"])[0]
    assert validar_ruta(copia, pol["N08"]["vuelta"])[0]


def test_destino_sin_camino_no_es_atendible(red: Grafo):
    copia = red.copia()
    for u, v, _ in [a for a in copia.aristas() if a[1] == "N34"]:
        copia.cerrar(u, v)
    for fabrica in POLITICAS.values():
        assert fabrica(copia)["N34"]["atendible"] is False
        assert fabrica(copia)["N19"]["atendible"] is True


# simulacion
def test_un_pedido_sin_cola(red: Grafo):
    metricas, _ = simular([(10.0, "N19")], politica_recomendada(red), 1)
    assert metricas["espera_prom"] == pytest.approx(0.0)
    assert metricas["solicitud_entrega_prom"] == pytest.approx(18.0)   # ida de 18 minutos
    assert metricas["puntuales"] == 1
    assert metricas["utilizacion"] == pytest.approx(100 * (18 + 24) / 480)


def test_la_descarga_se_suma_al_tiempo_de_entrega(red: Grafo):
    metricas, _ = simular([(10.0, "N19")], politica_recomendada(red), 1, descarga=5.0)
    assert metricas["solicitud_entrega_prom"] == pytest.approx(18.0 + 5.0)


def test_el_segundo_pedido_espera_al_vehiculo(red: Grafo):
    pedidos = [(0.0, "N19"), (1.0, "N19")]
    metricas, _ = simular(pedidos, politica_recomendada(red), 1)
    # el vehiculo regresa en 18 + 24 = 42, el segundo pedido llego en 1 y espera 41
    assert metricas["espera_prom"] == pytest.approx((0.0 + 41.0) / 2)


def test_un_pedido_fuera_del_umbral_no_es_puntual(red: Grafo):
    metricas, _ = simular([(0.0, "N40")], politica_actual(red), 1)    # ida de 93 minutos
    assert metricas["puntuales"] == 0


def test_la_cola_respeta_el_orden_de_llegada(red: Grafo):
    pedidos = [(0.0, "N19"), (1.0, "N40"), (2.0, "N08")]
    metricas, _ = simular(pedidos, politica_recomendada(red), 1)
    # N19 sale en 0 y vuelve en 42. N40 sale en 42 y vuelve en 42 + 47 + 61 = 150.
    # N08 sale en 150. Esperas 0, 41 y 148.
    assert metricas["espera_prom"] == pytest.approx((0 + 41 + 148) / 3)


def test_mismos_pedidos_con_la_misma_semilla():
    a = generar_pedidos(6, 8, np.random.default_rng(7))
    assert a == generar_pedidos(6, 8, np.random.default_rng(7))
    assert a != generar_pedidos(6, 8, np.random.default_rng(8))


def test_los_pedidos_respetan_la_ventana_y_los_destinos():
    pedidos = generar_pedidos(10, 8, np.random.default_rng(3))
    instantes = [t for t, _ in pedidos]
    assert instantes == sorted(instantes) and instantes[-1] < 8 * 60
    assert {d for _, d in pedidos} <= set(DESTINOS)


def test_sin_demanda_no_hay_pedidos():
    assert generar_pedidos(0, 8, np.random.default_rng(1)) == []


def test_la_tasa_de_llegadas_es_poisson():
    conteos = [len(generar_pedidos(6, 8, np.random.default_rng(s))) for s in range(2000)]
    assert abs(np.mean(conteos) - 48) < 1.0          # media teorica 6 por 8
    assert abs(np.var(conteos) - 48) < 5.0           # en Poisson la varianza iguala a la media


def test_los_destinos_son_equiprobables():
    pedidos = generar_pedidos(40, 24, np.random.default_rng(5))
    for destino in DESTINOS:
        proporcion = sum(1 for _, d in pedidos if d == destino) / len(pedidos)
        assert abs(proporcion - 0.25) < 0.05


def test_la_simulacion_continua_despues_de_las_ocho_horas(red: Grafo):
    pedidos = generar_pedidos(10, 8, np.random.default_rng(1))
    metricas, _ = simular(pedidos, politica_actual(red), 8)
    assert metricas["extra_entrega"] > 0
    assert metricas["extra_regreso"] >= metricas["extra_entrega"]
    assert metricas["atendidos"] == len(pedidos)


def test_destino_inalcanzable_se_reporta(red: Grafo):
    copia = red.copia()
    for u, v, _ in [a for a in copia.aristas() if a[1] == "N34"]:
        copia.cerrar(u, v)
    metricas, _ = simular([(1.0, "N34"), (2.0, "N19")], politica_recomendada(copia), 2)
    assert metricas["no_atendidos"] == 1 and metricas["atendidos"] == 1


def test_la_simulacion_coincide_con_el_modelo_de_verificacion(red: Grafo):
    politicas = [fabrica(red) for fabrica in POLITICAS.values()]
    for semilla in range(20):
        pedidos = generar_pedidos(6, 8, np.random.default_rng(semilla))
        for politica in politicas:
            metricas, _ = simular(pedidos, politica, 8)
            esperas, totales = simular_rapido(pedidos, politica, 8)
            assert metricas["espera_prom"] == pytest.approx(esperas.mean())
            assert metricas["solicitud_entrega_prom"] == pytest.approx(totales.mean())


# experimentos
def test_las_semillas_son_reproducibles_y_distintas():
    a, b = semillas(3), semillas(3)
    assert [np.random.default_rng(s).random() for s in a] == [np.random.default_rng(s).random() for s in b]
    assert len({np.random.default_rng(s).random() for s in a}) == 3


def test_las_politicas_reciben_los_mismos_pedidos(red: Grafo):
    df = correr(red, ESCENARIOS["Normal"], 8, reps=5)
    por_politica = df.pivot(index="rep", columns="politica", values="pedidos")
    assert (por_politica["actual"] == por_politica["recomendada"]).all()


def test_correr_es_reproducible(red: Grafo):
    uno = correr(red, 6, 8, reps=5)
    dos = correr(red, 6, 8, reps=5)
    assert uno.equals(dos)


def test_la_politica_recomendada_mejora_la_puntualidad(red: Grafo):
    df = correr(red, ESCENARIOS["Semana Santa"], 8, reps=10)
    dif = diferencias_pareadas(df).set_index("metrica")
    assert dif.loc["pct_puntuales", "ic_sup"] < 0       # actual menos recomendada es negativo
    assert dif.loc["espera_prom", "ic_inf"] > 0
