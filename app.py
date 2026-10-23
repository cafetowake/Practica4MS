"""
Universidad del Valle de Guatemala
Facultad de Ingenieria
Ciencia de la Computacion y Tecnologias de la Informacion
CC2017 - Modelacion y Simulacion
Seccion 30
Semestre 2, 2026

Actividad - Practica 4 Ejercicio aplicado, Logistica del Valle
Modulo: app - aplicacion configurable para estudiar rutas y flota

Integrantes:
Nombre: Paula De Leon
Carnet: 23202

Fecha: 22/10/2026

Ejecucion:
streamlit run app.py

Descripcion:
Aplicacion web hecha con Streamlit. Permite elegir el escenario de demanda o
fijar los pedidos por hora, la cantidad de vehiculos, la duracion de la
recepcion de pedidos y la semilla aleatoria. Tambien permite cerrar una
conexion o cambiar su tiempo para estudiar un incidente vial. Los cambios se
acumulan sobre la red, las rutas se recalculan con Dijkstra y se compara la
politica actual contra la recomendada usando los mismos pedidos en cada
repeticion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import pandas as pd
import streamlit as st

from experimentos import ESCENARIOS, SEMILLA_BASE, semillas
from grafo import DESTINOS, RUTA_MATRIZ, Grafo
from simulacion import POLITICAS, Politica, Traza, generar_pedidos, simular

NOMBRE_POLITICA = {"actual": "Actual", "recomendada": "Recomendada"}

COLUMNAS_RESULTADO = {
    "pedidos": "Pedidos recibidos",
    "no_atendidos": "Sin ruta",
    "puntuales": "Entregas puntuales",
    "pct_puntuales": "Puntuales en porcentaje",
    "espera_prom": "Espera promedio en minutos",
    "solicitud_entrega_prom": "Solicitud a entrega en minutos",
    "utilizacion": "Utilización en porcentaje",
    "extra_entrega": "Tiempo adicional en minutos",
    "cola_max": "Cola máxima",
}


class Cambio(TypedDict):
    """Modificación de una conexión de la red."""
    desde: str
    hacia: str
    accion: str        # "cerrar" o "tiempo"
    minutos: float     # sólo se usa con la acción "tiempo"
    ambos: bool        # aplicar también al sentido contrario


@dataclass
class Parametros:
    """Valores que el usuario fija en la barra lateral."""
    lam: float
    horas: float
    vehiculos: int
    semilla: int
    reps: int
    descarga: float


@st.cache_resource
def cargar_red() -> Grafo:
    """Lee la matriz una sola vez. Nunca se modifica, se trabaja sobre copias."""
    return Grafo.desde_csv(RUTA_MATRIZ)


def aplicar_cambios(base: Grafo, cambios: list[Cambio]) -> Grafo:
    """Devuelve una copia de la red con todos los cambios aplicados en orden."""
    red = base.copia()
    for c in cambios:
        if c["accion"] == "cerrar":
            red.cerrar(c["desde"], c["hacia"], c["ambos"])
        else:
            red.fijar_tiempo(c["desde"], c["hacia"], c["minutos"], c["ambos"])
    return red


def describir(c: Cambio) -> str:
    """Texto breve de un cambio para mostrarlo en la lista de cambios activos."""
    sentido = " y su sentido contrario" if c["ambos"] else ""
    if c["accion"] == "cerrar":
        return f"Cierre de {c['desde']} a {c['hacia']}{sentido}"
    return f"El tramo de {c['desde']} a {c['hacia']}{sentido} pasa a {c['minutos']:g} minutos"


def registrar(cambio: Cambio) -> None:
    """Agrega un cambio. Si ya había uno sobre la misma conexión, lo reemplaza."""
    cambios: list[Cambio] = st.session_state["cambios"]
    clave = (cambio["desde"], cambio["hacia"])
    cambios[:] = [c for c in cambios if (c["desde"], c["hacia"]) != clave]
    cambios.append(cambio)


def tabla_rutas(politicas: dict[str, Politica]) -> pd.DataFrame:
    """Tabla de rutas y tiempos por destino para cada política."""
    filas: list[dict[str, str | float | None]] = []
    for destino in DESTINOS:
        for clave, nombre in NOMBRE_POLITICA.items():
            r = politicas[clave][destino]
            if not r["atendible"]:
                filas.append({"Destino": destino, "Política": nombre, "Ruta de ida": "Sin ruta",
                              "Ida en minutos": None, "Ruta de regreso": "Sin ruta",
                              "Regreso en minutos": None, "Total en minutos": None})
                continue
            filas.append({
                "Destino": destino,
                "Política": nombre + (" con desvío" if r["desvio"] else ""),
                "Ruta de ida": " → ".join(r["ida"]),
                "Ida en minutos": r["t_ida"],
                "Ruta de regreso": " → ".join(r["vuelta"]),
                "Regreso en minutos": r["t_vuelta"],
                "Total en minutos": r["t_ida"] + r["t_vuelta"],
            })
    return pd.DataFrame(filas)


def serie_cola(traza: Traza) -> pd.Series:
    """Longitud de la cola en el tiempo, con el eje en horas."""
    serie = pd.Series([n for _, n in traza], index=[t / 60.0 for t, _ in traza], dtype=float)
    return serie.groupby(level=0).last()


def ejecutar(politicas: dict[str, Politica], p: Parametros) -> tuple[pd.DataFrame, dict[str, Traza]]:
    """Repite la simulación. Los pedidos de cada repetición son los mismos para ambas políticas."""
    filas: list[dict[str, float | int | str]] = []
    primera: dict[str, Traza] = {}
    for r, semilla in enumerate(semillas(p.reps, p.semilla)):
        pedidos = generar_pedidos(p.lam, p.horas, np.random.default_rng(semilla))
        for clave, politica in politicas.items():
            metricas, traza = simular(pedidos, politica, p.vehiculos, p.horas, p.descarga)
            if r == 0:
                primera[clave] = traza
            filas.append({"rep": r + 1, "politica": clave, **metricas})
    return pd.DataFrame(filas), primera


# callbacks
def _al_elegir_escenario() -> None:
    escenario = st.session_state["escenario"]
    if escenario in ESCENARIOS:
        st.session_state["lam"] = float(ESCENARIOS[escenario])


def _al_editar_tasa() -> None:
    tasa = float(st.session_state["lam"])
    for nombre, valor in ESCENARIOS.items():
        if tasa == float(valor):
            st.session_state["escenario"] = nombre
            return
    st.session_state["escenario"] = "Personalizado"


# barra lateral
def barra_demanda() -> Parametros:
    """Escenario, demanda, flota, duración de la recepción, semilla y repeticiones."""
    st.sidebar.header("Demanda y flota")
    st.sidebar.selectbox("Escenario", [*ESCENARIOS, "Personalizado"], key="escenario",
                         on_change=_al_elegir_escenario)
    lam = st.sidebar.number_input("Pedidos por hora", min_value=0.5, max_value=40.0, step=0.5,
                                  key="lam", on_change=_al_editar_tasa)
    horas = st.sidebar.number_input("Horas de recepción de pedidos", min_value=1.0, max_value=24.0,
                                    value=8.0, step=0.5, key="horas")
    vehiculos = st.sidebar.number_input("Vehículos", min_value=1, max_value=60, value=8, step=1,
                                        key="vehiculos")
    semilla = st.sidebar.number_input("Semilla aleatoria", min_value=0, max_value=10**9,
                                      value=SEMILLA_BASE, step=1, key="semilla")
    reps = st.sidebar.number_input("Repeticiones", min_value=1, max_value=200, value=10, step=1,
                                   key="reps")
    descarga = st.sidebar.number_input("Minutos de descarga por entrega", min_value=0.0,
                                       max_value=30.0, value=0.0, step=1.0, key="descarga")
    return Parametros(float(lam), float(horas), int(vehiculos), int(semilla), int(reps), float(descarga))


def barra_incidente(base: Grafo) -> None:
    """Controles para cerrar una conexión o cambiar su tiempo."""
    st.sidebar.header("Incidente vial")
    desde = str(st.sidebar.selectbox("Desde", base.nodos(), key="desde"))
    hacia = str(st.sidebar.selectbox("Hacia", list(base.ady[desde])))
    tiempo_base = base.peso(desde, hacia)
    minutos = float(st.sidebar.number_input(
        "Nuevo tiempo en minutos", min_value=1.0, max_value=300.0,
        value=float(tiempo_base) if tiempo_base is not None else 1.0, step=1.0))
    ambos = bool(st.sidebar.checkbox("Aplicar también al sentido contrario", key="ambos"))
    izquierda, derecha = st.sidebar.columns(2)
    if izquierda.button("Cambiar tiempo"):
        registrar(Cambio(desde=desde, hacia=hacia, accion="tiempo", minutos=minutos, ambos=ambos))
    if derecha.button("Cerrar conexión"):
        registrar(Cambio(desde=desde, hacia=hacia, accion="cerrar", minutos=0.0, ambos=ambos))
    if st.sidebar.button("Restablecer red"):
        st.session_state["cambios"] = []


# zona principal
def mostrar_cambios(cambios: list[Cambio]) -> None:
    st.subheader("Cambios activos en la red")
    if cambios:
        st.dataframe(pd.DataFrame({"Cambio": [describir(c) for c in cambios]}),
                     width="stretch", hide_index=True)
    else:
        st.caption("La red se usa tal como viene en la matriz.")


def mostrar_rutas(politicas: dict[str, Politica]) -> None:
    st.subheader("Rutas por destino")
    sin_ruta = [d for d in DESTINOS if not politicas["recomendada"][d]["atendible"]]
    if sin_ruta:
        st.warning(f"No existe camino de ida y regreso hacia {', '.join(sin_ruta)}. "
                   "Los pedidos a esos destinos no se atienden y se cuentan aparte.")
    st.dataframe(tabla_rutas(politicas), width="stretch", hide_index=True)
    with st.expander("Cómo se actualizan las rutas cuando cambia la red"):
        st.markdown(
            "1. Cada cambio se aplica sobre una copia de la red y la matriz original no se toca.\n"
            "2. Dijkstra se ejecuta de nuevo desde N01 hacia cada destino y desde cada destino "
            "hacia N01, y así se obtienen las rutas de ida y de regreso vigentes.\n"
            "3. En la política recomendada se adoptan esas rutas.\n"
            "4. En la política actual se vuelven a sumar los tiempos de las hojas de despacho. "
            "Si una hoja usa un tramo cerrado, se sustituye por el camino mínimo y se marca con desvío.\n"
            "5. Si un destino queda sin camino, sus pedidos no se atienden y se informan aparte.")


def mostrar_simulacion(politicas: dict[str, Politica], p: Parametros) -> None:
    resultados, primera = ejecutar(politicas, p)
    st.subheader(f"Promedio de {p.reps} repeticiones con los mismos pedidos para ambas políticas")
    resumen = pd.DataFrame(resultados.groupby("politica")[list(COLUMNAS_RESULTADO)].mean())
    resumen = resumen.rename(columns=COLUMNAS_RESULTADO, index=NOMBRE_POLITICA).round(1)
    st.dataframe(resumen, width="stretch")

    st.subheader("Cola de la primera repetición")
    cola = pd.DataFrame({NOMBRE_POLITICA[k]: serie_cola(t) for k, t in primera.items()}).ffill()
    st.line_chart(cola, x_label="Horas desde el inicio", y_label="Pedidos en cola")

    st.subheader("Resultados por repetición")
    etiquetas = {"rep": "Repetición", "politica": "Política", **COLUMNAS_RESULTADO}
    detalle = resultados.loc[:, list(etiquetas)].rename(columns=etiquetas)
    detalle["Política"] = detalle["Política"].replace(NOMBRE_POLITICA)
    st.dataframe(detalle.round(2), width="stretch", hide_index=True)


def principal() -> None:
    st.set_page_config(page_title="Logística del Valle", layout="wide")
    st.session_state.setdefault("escenario", "Normal")
    st.session_state.setdefault("lam", float(ESCENARIOS["Normal"]))
    st.session_state.setdefault("cambios", [])

    base = cargar_red()
    st.title("Logística del Valle. Rutas y flota")
    st.caption("Compara las rutas de las hojas de despacho con las rutas calculadas por Dijkstra "
               "usando los mismos pedidos en cada repetición.")

    parametros = barra_demanda()
    barra_incidente(base)

    cambios: list[Cambio] = st.session_state["cambios"]
    red = aplicar_cambios(base, cambios)
    politicas = {clave: POLITICAS[clave](red) for clave in NOMBRE_POLITICA}

    mostrar_cambios(cambios)
    mostrar_rutas(politicas)
    if st.button("Simular", type="primary"):
        mostrar_simulacion(politicas, parametros)


principal()
