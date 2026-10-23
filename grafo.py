"""
Universidad del Valle de Guatemala
Facultad de Ingenieria
Ciencia de la Computacion y Tecnologias de la Informacion
CC2017 - Modelacion y Simulacion
Seccion 30
Semestre 2, 2026

Actividad - Practica 4 Ejercicio aplicado, Logistica del Valle
Modulo: grafo - construccion y validacion del grafo dirigido de la red vial

Integrantes:
Nombre: Paula De Leon
Carnet: 23202

Fecha: 22/10/2026

Ejecucion:
Este modulo no se ejecuta solo, lo importan los demas modulos. Para verlo
en accion se puede correr python -m pytest -q test_nucleo.py

Descripcion:
Lee la matriz de adyacencia de 40 nodos y la convierte en un grafo dirigido
ponderado, donde cada arco indica los minutos de viaje de un origen a un
destino. Ofrece operaciones para cerrar una conexion o cambiar su tiempo, y
funciones para sumar el tiempo de una ruta y comprobar si existe tramo por
tramo. Tambien define la bodega, los cuatro destinos y las rutas que usan hoy
los conductores segun las hojas de despacho.
"""

from __future__ import annotations

import copy
import csv
from pathlib import Path

DIR_BASE = Path(__file__).resolve().parent
RUTA_MATRIZ = DIR_BASE / "datos" / "matriz.csv"
DIR_SALIDA = DIR_BASE / "salida"

BODEGA = "N01"
DESTINOS: list[str] = ["N08", "N19", "N34", "N40"]

# Rutas de las hojas de despacho. El regreso es la misma ruta en sentido inverso.
RUTAS_ACTUALES: dict[str, list[str]] = {
    "N08": ["N01", "N02", "N03", "N04", "N05", "N06", "N07", "N08"],
    "N19": ["N01", "N02", "N03", "N11", "N19"],
    "N34": ["N01", "N09", "N17", "N25", "N33", "N34"],
    "N40": ["N01", "N02", "N03", "N04", "N05", "N06", "N07", "N08",
            "N16", "N24", "N32", "N40"],
}


class Grafo:
    """Grafo dirigido y ponderado. ``ady[u][v]`` son los minutos de u a v."""

    def __init__(self, ady: dict[str, dict[str, float]] | None = None) -> None:
        self.ady: dict[str, dict[str, float]] = ady if ady is not None else {}

    @staticmethod
    def desde_csv(ruta: str | Path = RUTA_MATRIZ) -> "Grafo":
        """Construye el grafo a partir de la matriz de adyacencia.

        Cada fila es un origen y cada columna un destino. Una celda vacía o con
        el signo de guion largo significa que no hay conexión directa y el
        valor cero identifica el mismo punto.
        """
        with open(ruta, newline="", encoding="utf-8-sig") as f:
            filas = list(csv.reader(f))
        columnas = filas[0][1:]
        ady: dict[str, dict[str, float]] = {n: {} for n in columnas}
        for fila in filas[1:]:
            if not fila or not fila[0]:
                continue
            origen = fila[0]
            if origen not in ady:
                raise ValueError(f"La fila {origen} no tiene columna correspondiente")
            for destino, valor in zip(columnas, fila[1:]):
                valor = valor.strip()
                if valor in ("", "—") or origen == destino:
                    continue
                ady[origen][destino] = float(valor)
        return Grafo(ady)

    def copia(self) -> "Grafo":
        """Copia profunda, útil para simular un incidente sin tocar la red base."""
        return Grafo(copy.deepcopy(self.ady))

    def nodos(self) -> list[str]:
        return list(self.ady.keys())

    def aristas(self) -> list[tuple[str, str, float]]:
        """Lista de arcos como tuplas (origen, destino, minutos)."""
        return [(u, v, w) for u, vs in self.ady.items() for v, w in vs.items()]

    def peso(self, u: str, v: str) -> float | None:
        """Minutos del arco u a v, o None si no existe."""
        return self.ady.get(u, {}).get(v)

    def cerrar(self, u: str, v: str, ambos_sentidos: bool = False) -> None:
        """Elimina el arco u a v y, si se pide, también el arco v a u."""
        self.ady[u].pop(v, None)
        if ambos_sentidos:
            self.ady[v].pop(u, None)

    def fijar_tiempo(self, u: str, v: str, minutos: float, ambos_sentidos: bool = False) -> None:
        """Cambia el tiempo del arco u a v. En sentido contrario sólo si ya existe."""
        if v in self.ady[u] or not ambos_sentidos:
            self.ady[u][v] = float(minutos)
        if ambos_sentidos and u in self.ady[v]:
            self.ady[v][u] = float(minutos)

    def es_asimetrico(self) -> tuple[list[tuple[str, str, float]], list[tuple[str, str, float, float]]]:
        """Separa las conexiones asimétricas de la red.

        Devuelve dos listas. La primera tiene los arcos de un solo sentido y la
        segunda los pares de doble sentido con tiempos distintos por dirección.
        """
        solo_un_sentido: list[tuple[str, str, float]] = []
        tiempos_distintos: list[tuple[str, str, float, float]] = []
        for u, v, w in self.aristas():
            inverso = self.peso(v, u)
            if inverso is None:
                solo_un_sentido.append((u, v, w))
            elif u < v and inverso != w:
                tiempos_distintos.append((u, v, w, inverso))
        return solo_un_sentido, tiempos_distintos


def tiempo_ruta(grafo: Grafo, ruta: list[str]) -> float | None:
    """Suma de minutos de una ruta, o None si algún tramo no existe."""
    total = 0.0
    for u, v in zip(ruta[:-1], ruta[1:]):
        w = grafo.peso(u, v)
        if w is None:
            return None
        total += w
    return total


def validar_ruta(grafo: Grafo, ruta: list[str]) -> tuple[bool, float | None, list[tuple[str, str]]]:
    """Comprueba una ruta tramo por tramo.

    Devuelve si es válida, su tiempo total y la lista de tramos inexistentes.
    """
    faltantes = [(u, v) for u, v in zip(ruta[:-1], ruta[1:]) if grafo.peso(u, v) is None]
    if faltantes:
        return False, None, faltantes
    return True, tiempo_ruta(grafo, ruta), []
