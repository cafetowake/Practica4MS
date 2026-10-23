"""
Universidad del Valle de Guatemala
Facultad de Ingenieria
Ciencia de la Computacion y Tecnologias de la Informacion
CC2017 - Modelacion y Simulacion
Seccion 30
Semestre 2, 2026

Actividad - Practica 4 Ejercicio aplicado, Logistica del Valle
Modulo: test_app - pruebas de la aplicacion de Streamlit

Integrantes:
Nombre: Paula De Leon
Carnet: 23202

Fecha: 22/10/2026

Ejecucion:
python -m pytest -q test_app.py

Descripcion:
Abre la aplicacion con el probador de Streamlit y recorre sus funciones sin
necesidad de un navegador. Comprueba la carga inicial, los escenarios, la
edicion de la tasa de pedidos, el cierre de una conexion, el cambio de tiempo,
el restablecimiento de la red, el aviso cuando un destino queda sin camino y la
ejecucion de las repeticiones.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent / "app.py")


def abrir() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    return at


def boton(at: AppTest, etiqueta: str):
    return next(b for b in at.sidebar.button if b.label == etiqueta)


def selector(at: AppTest, etiqueta: str):
    return next(s for s in at.sidebar.selectbox if s.label == etiqueta)


def tabla_de_rutas(at: AppTest) -> pd.DataFrame:
    return next(d.value for d in at.dataframe if "Ruta de ida" in d.value.columns)


def total(rutas: pd.DataFrame, destino: str, politica: str) -> float:
    """Total en minutos de la fila que corresponde al destino y a la política."""
    fila = (rutas["Destino"] == destino) & (rutas["Política"].str.startswith(politica))
    return float(rutas.loc[fila, "Total en minutos"].iloc[0])


def cerrar(at: AppTest, desde: str, hacia: str) -> None:
    selector(at, "Desde").select(desde).run()
    selector(at, "Hacia").select(hacia).run()
    boton(at, "Cerrar conexión").click().run()
    assert not at.exception


def test_la_aplicacion_carga_sin_errores():
    at = abrir()
    assert at.title[0].value == "Logística del Valle. Rutas y flota"
    assert len(tabla_de_rutas(at)) == 8      # cuatro destinos por dos políticas
    assert at.session_state["cambios"] == []


def test_el_escenario_fija_la_tasa_de_pedidos():
    at = abrir()
    selector(at, "Escenario").select("Navidad").run()
    assert at.sidebar.number_input(key="lam").value == 10.0
    selector(at, "Escenario").select("Semana Santa").run()
    assert at.sidebar.number_input(key="lam").value == 6.0


def test_editar_la_tasa_cambia_el_escenario():
    at = abrir()
    at.sidebar.number_input(key="lam").set_value(7.0).run()
    assert at.sidebar.selectbox(key="escenario").value == "Personalizado"
    at.sidebar.number_input(key="lam").set_value(4.0).run()
    assert at.sidebar.selectbox(key="escenario").value == "Normal"


def test_cerrar_una_conexion_desvia_la_ruta_actual():
    at = abrir()
    cerrar(at, "N02", "N03")
    cambios = at.session_state["cambios"]
    assert len(cambios) == 1 and cambios[0]["accion"] == "cerrar"
    rutas = tabla_de_rutas(at)
    assert "Actual con desvío" in set(rutas["Política"])
    fila = (rutas["Destino"] == "N08") & (rutas["Política"].str.startswith("Actual"))
    assert "N02 → N03" not in str(rutas.loc[fila, "Ruta de ida"].iloc[0])


def test_cambiar_el_tiempo_recalcula_las_rutas():
    at = abrir()
    antes = total(tabla_de_rutas(at), "N34", "Recomendada")
    selector(at, "Desde").select("N26").run()
    selector(at, "Hacia").select("N34").run()
    entrada = next(n for n in at.sidebar.number_input if n.label == "Nuevo tiempo en minutos")
    entrada.set_value(60.0).run()
    boton(at, "Cambiar tiempo").click().run()
    assert not at.exception
    assert total(tabla_de_rutas(at), "N34", "Recomendada") > antes


def test_restablecer_la_red_borra_los_cambios():
    at = abrir()
    cerrar(at, "N02", "N03")
    boton(at, "Restablecer red").click().run()
    assert at.session_state["cambios"] == []


def test_un_destino_sin_camino_muestra_aviso_y_se_puede_simular():
    at = abrir()
    cerrar(at, "N26", "N34")
    cerrar(at, "N33", "N34")
    assert any("N34" in w.value for w in at.warning)
    at.sidebar.number_input(key="reps").set_value(2).run()
    next(b for b in at.button if b.label == "Simular").click().run()
    assert not at.exception
    resumen = next(d.value for d in at.dataframe if "Sin ruta" in d.value.columns)
    assert (resumen["Sin ruta"] > 0).all()


def test_simular_muestra_resultados_y_grafica():
    at = abrir()
    at.sidebar.number_input(key="reps").set_value(3).run()
    next(b for b in at.button if b.label == "Simular").click().run()
    assert not at.exception
    resumen = next(d.value for d in at.dataframe if "Entregas puntuales" in d.value.columns)
    assert list(resumen.index) == ["Actual", "Recomendada"]
    assert resumen.loc["Recomendada", "Puntuales en porcentaje"] > resumen.loc["Actual", "Puntuales en porcentaje"]
    assert len(at.get("vega_lite_chart")) == 1
