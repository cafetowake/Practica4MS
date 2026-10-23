"""
Universidad del Valle de Guatemala
Facultad de Ingenieria
Ciencia de la Computacion y Tecnologias de la Informacion
CC2017 - Modelacion y Simulacion
Seccion 30
Semestre 2, 2026

Actividad - Practica 4 Ejercicio aplicado, Logistica del Valle
Modulo: experimentos - corridas pareadas, barridos e incidentes viales

Integrantes:
Nombre: Paula De Leon
Carnet: 23202

Fecha: 22/10/2026

Ejecucion:
python experimentos.py
python experimentos.py 50     (opcion que indica el numero de repeticiones)

Descripcion:
Ejecuta los experimentos del informe y guarda los resultados como CSV en la
carpeta salida. Para cada repeticion los pedidos se generan una sola vez y se
entregan a todas las politicas, de modo que la comparacion es pareada. Incluye
los tres escenarios con ocho vehiculos, un barrido de la cantidad de vehiculos,
un barrido de la demanda y tres incidentes viales en Semana Santa. Las
semillas de las repeticiones se derivan con SeedSequence a partir de una
semilla base, por lo que los resultados son reproducibles.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats

from grafo import DIR_SALIDA, RUTA_MATRIZ, Grafo
from simulacion import POLITICAS, generar_pedidos, simular

ESCENARIOS: dict[str, int] = {"Normal": 4, "Semana Santa": 6, "Navidad": 10}
SEMILLA_BASE = 2026
METRICAS = ["pedidos", "atendidos", "no_atendidos", "puntuales", "pct_puntuales", "espera_prom",
            "solicitud_entrega_prom", "utilizacion", "extra_entrega", "extra_regreso",
            "cola_prom", "cola_max"]


def semillas(n: int, base: int = SEMILLA_BASE) -> list[np.random.SeedSequence]:
    """Semillas independientes para n repeticiones."""
    return np.random.SeedSequence(base).spawn(n)


def correr(grafo: Grafo, lam: float, vehiculos: int, horas: float = 8.0, reps: int = 30,
           politicas: tuple[str, ...] = ("actual", "recomendada"), base: int = SEMILLA_BASE,
           descarga: float = 0.0) -> pd.DataFrame:
    """Repite la simulación y devuelve una fila por repetición y política.

    Los pedidos de cada repetición se generan una sola vez y se reutilizan en
    todas las políticas.
    """
    pols = {k: POLITICAS[k](grafo) for k in politicas}
    filas: list[dict[str, float | int | str]] = []
    for r, ss in enumerate(semillas(reps, base)):
        pedidos = generar_pedidos(lam, horas, np.random.default_rng(ss))
        for nombre, pol in pols.items():
            metricas, _ = simular(pedidos, pol, vehiculos, horas, descarga)
            filas.append({"rep": r + 1, "politica": nombre, **metricas})
    return pd.DataFrame(filas)


def resumen(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Media, desviación estándar y semiamplitud del intervalo de confianza del 95 por ciento."""
    g = df.groupby("politica")[METRICAS]
    media, desv = g.mean(), g.std(ddof=1)
    n = df.groupby("politica").size()
    t = stats.t.ppf(0.975, n - 1)
    ic = desv.mul(t / np.sqrt(n), axis=0)
    return pd.DataFrame(media), pd.DataFrame(desv), pd.DataFrame(ic)


def diferencias_pareadas(df: pd.DataFrame, a: str = "actual", b: str = "recomendada") -> pd.DataFrame:
    """Diferencia a - b por repetición con intervalo de confianza del 95 por ciento."""
    filas = []
    for m in METRICAS:
        p = df.pivot(index="rep", columns="politica", values=m)
        d = p[a] - p[b]
        n = len(d)
        h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
        filas.append({"metrica": m, "dif_media": d.mean(), "ic_inf": d.mean() - h, "ic_sup": d.mean() + h})
    return pd.DataFrame(filas)


def incidentes(reps: int = 30, lam: float = 6, vehiculos: int = 8) -> pd.DataFrame:
    """Tres incidentes viales en el escenario de Semana Santa."""
    base = Grafo.desde_csv(RUTA_MATRIZ)
    casos = {
        "Sin incidente": lambda h: None,
        "Cierre de N09 a N01": lambda h: h.cerrar("N09", "N01"),
        "Cierre de N02 a N03": lambda h: h.cerrar("N02", "N03"),
        "Congestion N01 a N17 (9 a 27 min)": lambda h: h.fijar_tiempo("N01", "N17", 27),
    }
    filas = []
    for nombre, modificar in casos.items():
        h = base.copia()
        modificar(h)
        df = correr(h, lam, vehiculos, reps=reps)
        for pol, sub in df.groupby("politica"):
            filas.append({"incidente": nombre, "politica": pol, **sub[METRICAS].mean().to_dict()})
    return pd.DataFrame(filas)


def main(reps: int = 30) -> None:
    DIR_SALIDA.mkdir(exist_ok=True)
    g = Grafo.desde_csv(RUTA_MATRIZ)

    # 1. escenarios base con 8 vehiculos
    todos, difs = [], []
    for nombre, lam in ESCENARIOS.items():
        df = correr(g, lam, 8, reps=reps, politicas=("actual", "recomendada", "reversible"))
        df.insert(0, "escenario", nombre)
        todos.append(df)
        d = diferencias_pareadas(df)
        d.insert(0, "escenario", nombre)
        difs.append(d)
    pd.concat(todos).to_csv(DIR_SALIDA / "escenarios_corridas.csv", index=False)
    pd.concat(difs).to_csv(DIR_SALIDA / "escenarios_diferencias.csv", index=False)

    # 2. barrido de flota
    filas = []
    for nombre, lam in ESCENARIOS.items():
        for v in range(4, 25):
            df = correr(g, lam, v, reps=reps)
            for pol, sub in df.groupby("politica"):
                filas.append({"escenario": nombre, "lam": lam, "vehiculos": v, "politica": pol,
                              **sub[METRICAS].mean().to_dict()})
    pd.DataFrame(filas).to_csv(DIR_SALIDA / "barrido_flota.csv", index=False)

    # 3. barrido de demanda con 8 vehiculos
    filas = []
    for lam in range(2, 15):
        df = correr(g, lam, 8, reps=reps)
        for pol, sub in df.groupby("politica"):
            filas.append({"lam": lam, "politica": pol, **sub[METRICAS].mean().to_dict()})
    pd.DataFrame(filas).to_csv(DIR_SALIDA / "barrido_demanda.csv", index=False)

    # 4. incidentes viales
    out = incidentes(reps=reps)
    out.to_csv(DIR_SALIDA / "incidentes.csv", index=False)
    print("Experimentos terminados. Resultados en", DIR_SALIDA)
    print(out.round(1).to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
