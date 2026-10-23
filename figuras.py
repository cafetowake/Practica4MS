"""
Universidad del Valle de Guatemala
Facultad de Ingenieria
Ciencia de la Computacion y Tecnologias de la Informacion
CC2017 - Modelacion y Simulacion
Seccion 30
Semestre 2, 2026

Actividad - Practica 4 Ejercicio aplicado, Logistica del Valle
Modulo: figuras - graficas del informe

Integrantes:
Nombre: Paula De Leon
Carnet: 23202

Fecha: 22/10/2026

Ejecucion:
python experimentos.py
python figuras.py

Descripcion:
Genera las figuras del informe en la carpeta salida. Dibuja el grafo de la red con
sus sentidos de circulacion, la comparacion de rutas por destino y las graficas de
los escenarios, de la cola, de la cantidad de vehiculos y de la demanda. Las
graficas de resultados se leen de los CSV que escribe experimentos.py, por lo que
ese modulo debe ejecutarse primero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from numpy.typing import ArrayLike
from scipy import stats

from grafo import BODEGA, DESTINOS, DIR_SALIDA, RUTAS_ACTUALES, Grafo
from simulacion import politica_actual, politica_recomendada, generar_pedidos, simular
from experimentos import semillas, ESCENARIOS

AZUL, NARANJA, GRIS, TINTA = "#2a78d6", "#eb6834", "#c9c8c3", "#0b0b0b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#8a8984", "axes.grid": True,
                     "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.axisbelow": True})
g = Grafo.desde_csv()


def figura_grafo() -> None:
    G = nx.Graph()
    for u, v, w in g.aristas():
        G.add_edge(u, v, weight=w)
    pos = nx.kamada_kawai_layout(G, weight=None)
    fig, ax = plt.subplots(figsize=(15, 11))
    sentido_unico, doble = [], []
    for u, v, w in g.aristas():
        if g.peso(v, u) is None:
            sentido_unico.append((u, v))
        elif u < v:
            doble.append((u, v))
    nx.draw_networkx_edges(G, pos, edgelist=doble, edge_color=GRIS, width=1.4, ax=ax)
    D = nx.DiGraph(sentido_unico)
    nx.draw_networkx_edges(D, pos, edgelist=sentido_unico, edge_color="#7a5fd0", width=1.4,
                           arrows=True, arrowsize=14, node_size=520, ax=ax,
                           connectionstyle="arc3,rad=0.0")
    rec = politica_recomendada(g)
    actual_e = set()
    for d in DESTINOS:
        r = RUTAS_ACTUALES[d]
        actual_e |= set(frozenset(p) for p in zip(r[:-1], r[1:]))
    rec_e = set()
    for d in DESTINOS:
        r = rec[d]["ida"]
        rec_e |= set(frozenset(p) for p in zip(r[:-1], r[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=[tuple(e) for e in actual_e], edge_color=AZUL,
                           width=3.2, alpha=0.75, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=[tuple(e) for e in rec_e], edge_color=NARANJA,
                           width=2.4, style="dashed", ax=ax)
    colores = []
    for n in G.nodes:
        colores.append("#0b0b0b" if n == BODEGA else "#1baf7a" if n in DESTINOS else "#f3f2ee")
    nx.draw_networkx_nodes(G, pos, node_color=colores, edgecolors="#52514e", node_size=520, ax=ax)
    for n in G.nodes:
        x, y = pos[n]
        destacado = n == BODEGA or n in DESTINOS
        ax.text(x, y, n[1:], ha="center", va="center", fontsize=8,
                color="white" if n == BODEGA else "#0b0b0b",
                fontweight="bold" if destacado else "normal")
    # etiquetas de minutos en aristas
    etiq = {}
    for u, v in doble:
        a, b = g.peso(u, v), g.peso(v, u)
        etiq[(u, v)] = f"{a:g}" if a == b else f"{a:g}/{b:g}"
    for u, v in sentido_unico:
        etiq[(u, v)] = f"{g.peso(u, v):g}"
    nx.draw_networkx_edge_labels(G, pos, edge_labels=etiq, font_size=6.5, font_color="#52514e",
                                 bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8),
                                 ax=ax)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    leyenda = [Line2D([0], [0], color=GRIS, lw=2, label="Calle de doble sentido (minutos ida/vuelta)"),
               Line2D([0], [0], color="#7a5fd0", lw=2, marker=">", label="Calle de un solo sentido"),
               Line2D([0], [0], color=AZUL, lw=3.2, label="Rutas actuales de despacho"),
               Line2D([0], [0], color=NARANJA, lw=2.4, ls="--", label="Rutas de ida recomendadas"),
               Patch(fc="#0b0b0b", label="Bodega N01"), Patch(fc="#1baf7a", label="Destinos")]
    ax.legend(handles=leyenda, loc="lower left", frameon=True, fontsize=9)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(DIR_SALIDA / "fig_grafo.png", dpi=170)
    plt.close(fig)


def figura_rutas() -> None:
    act, rec = politica_actual(g), politica_recomendada(g)
    x = np.arange(len(DESTINOS))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    w = 0.36
    for k, (pol, col, nom) in enumerate([(act, AZUL, "Actual"), (rec, NARANJA, "Recomendada")]):
        ida = [pol[d]["t_ida"] for d in DESTINOS]
        vue = [pol[d]["t_vuelta"] for d in DESTINOS]
        xs = x + (k - 0.5) * w
        ax.bar(xs, ida, w - 0.03, color=col, label=f"{nom} ida")
        ax.bar(xs, vue, w - 0.03, bottom=ida, color=col, alpha=0.45, label=f"{nom} regreso")
        for xi, a, b in zip(xs, ida, vue):
            ax.text(xi, a + b + 2, f"{a + b:g}", ha="center", fontsize=8.5)
    ax.axhline(75, color="#52514e", lw=0.9, ls=":")
    ax.text(3.45, 77, "75 min", fontsize=8, color="#52514e", ha="right")
    ax.set_xticks(x, DESTINOS)
    ax.set_ylabel("Minutos")
    ax.legend(ncol=2, frameon=False, fontsize=8.5, loc="upper left")
    ax.set_ylim(0, 215)
    fig.tight_layout()
    fig.savefig(DIR_SALIDA / "fig_rutas.png", dpi=170)
    plt.close(fig)


def _ic(valores: ArrayLike) -> float:
    """Semiamplitud del intervalo de confianza del 95 por ciento para la media."""
    x = np.asarray(valores, dtype=float)
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def figura_escenarios() -> None:
    d = pd.read_csv(DIR_SALIDA / "escenarios_corridas.csv")
    d = d[d.politica.isin(["actual", "recomendada"])]
    metricas = [("pct_puntuales", "Entregas puntuales (%)"),
                ("espera_prom", "Espera promedio (min)"),
                ("solicitud_entrega_prom", "Solicitud a entrega (min)"),
                ("utilizacion", "Utilizacion de vehiculos (%)"),
                ("extra_entrega", "Tiempo adicional (min)")]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.8))
    escen = list(ESCENARIOS)
    for ax, (m, tit) in zip(axes, metricas):
        for k, (pol, col, nom) in enumerate([("actual", AZUL, "Actual"), ("recomendada", NARANJA, "Recomendada")]):
            medias = [d[(d.escenario == e) & (d.politica == pol)][m].mean() for e in escen]
            ics = [_ic(d[(d.escenario == e) & (d.politica == pol)][m]) for e in escen]
            xs = np.arange(3) + (k - 0.5) * 0.38
            ax.bar(xs, medias, 0.35, yerr=ics, color=col, label=nom, error_kw=dict(lw=0.9, capsize=2))
        ax.set_xticks(range(3), ["Normal", "S. Santa", "Navidad"], fontsize=8.5)
        ax.set_title(tit, fontsize=9.5)
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(DIR_SALIDA / "fig_escenarios.png", dpi=170)
    plt.close(fig)


def figura_cola() -> None:
    ss = semillas(30)[0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), sharey=False)
    act, rec = politica_actual(g), politica_recomendada(g)
    for ax, (nombre, lam) in zip(axes, ESCENARIOS.items()):
        ped = generar_pedidos(lam, 8, np.random.default_rng(ss))
        for pol, col, nom in [(act, AZUL, "Actual"), (rec, NARANJA, "Recomendada")]:
            _, tr = simular(ped, pol, 8)
            t = [a / 60 for a, _ in tr]
            q = [b for _, b in tr]
            ax.step(t, q, where="post", color=col, lw=1.4, label=nom)
        ax.axvline(8, color="#52514e", lw=0.9, ls=":")
        ax.set_title(f"{nombre} ({lam} pedidos por hora)", fontsize=10)
        ax.set_xlabel("Horas desde el inicio")
        ax.set_ylabel("Pedidos en cola")
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(DIR_SALIDA / "fig_cola.png", dpi=170)
    plt.close(fig)


def figura_flota() -> None:
    f = pd.read_csv(DIR_SALIDA / "barrido_flota.csv")
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.4), sharex=True)
    for j, (nombre, lam) in enumerate(ESCENARIOS.items()):
        for pol, col, nom in [("actual", AZUL, "Actual"), ("recomendada", NARANJA, "Recomendada")]:
            s = f[(f.escenario == nombre) & (f.politica == pol)]
            axes[0, j].plot(s.vehiculos, s.pct_puntuales, color=col, lw=1.8, label=nom)
            axes[1, j].plot(s.vehiculos, s.espera_prom, color=col, lw=1.8, label=nom)
        axes[0, j].set_title(f"{nombre} ({lam} pedidos por hora)", fontsize=10)
        axes[0, j].axvline(8, color="#52514e", lw=0.8, ls=":")
        axes[1, j].axvline(8, color="#52514e", lw=0.8, ls=":")
        axes[1, j].set_xlabel("Vehiculos")
        axes[1, j].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        axes[1, j].set_yscale("symlog", linthresh=10)
    axes[0, 0].set_ylabel("Entregas puntuales (%)")
    axes[1, 0].set_ylabel("Espera promedio (min)")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(DIR_SALIDA / "fig_flota.png", dpi=170)
    plt.close(fig)


def figura_demanda() -> None:
    f = pd.read_csv(DIR_SALIDA / "barrido_demanda.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for pol, col, nom in [("actual", AZUL, "Actual"), ("recomendada", NARANJA, "Recomendada")]:
        s = f[f.politica == pol]
        axes[0].plot(s.lam, s.pct_puntuales, color=col, lw=1.8, marker="o", ms=4, label=nom)
        axes[1].plot(s.lam, s.espera_prom, color=col, lw=1.8, marker="o", ms=4, label=nom)
    for ax in axes:
        ax.set_xlabel("Pedidos por hora")
    # capacidad teorica con 8 vehiculos
    for ax in axes:
        ax.axvline(8 / 113.5 * 60, color=AZUL, lw=0.8, ls=":")
        ax.axvline(8 / 79.5 * 60, color=NARANJA, lw=0.8, ls=":")
    axes[0].set_ylabel("Entregas puntuales (%)")
    axes[1].set_ylabel("Espera promedio (min)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(DIR_SALIDA / "fig_demanda.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    figura_grafo()
    figura_rutas()
    figura_escenarios()
    figura_cola()
    figura_flota()
    figura_demanda()
    print("figuras listas")
