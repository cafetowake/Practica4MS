# Practica 4 
## Universidad del Valle de Guatemala
## Facultad de Ingenieria
## CC2017 Modelacion y Simulacion, seccion 30
## semestre 2, 2026.
### Paula De Leon, carnet 23202.

La Practica construye el grafo dirigido de la red de Logistica del Valle a partir de la matriz de
tiempos, calcula rutas de tiempo minimo y compara con una simulacion de eventos discretos las rutas
actuales de las hojas de despacho contra las rutas recomendadas por el algoritmo.

## Estructura

| Archivo | Contenido |
|---|---|
| `grafo.py` | Lectura de la matriz, clase del grafo dirigido, cierre y cambio de tiempo de una conexion, suma y validacion de rutas. |
| `algoritmos.py` | Dijkstra, A* con heuristica de tramos, Bellman-Ford y el metodo de Yen para rutas alternativas. |
| `simulacion.py` | Politicas de ruta, generacion de pedidos como proceso de Poisson y simulacion de eventos discretos con cola FIFO. |
| `experimentos.py` | Corridas pareadas por escenario, barridos de flota y de demanda e incidentes viales. Escribe CSV en `salida/`. |
| `figuras.py` | Graficas del informe a partir de los CSV. |
| `app.py` | Aplicacion web configurable hecha con Streamlit. |
| `test_nucleo.py` | Pruebas del grafo, los algoritmos y la simulacion. |
| `test_app.py` | Pruebas de la aplicacion con el probador de Streamlit. |
| `datos/matriz.csv` | Matriz de adyacencia del anexo A, con tiempos en minutos. |
| `salida/` | CSV y figuras generados. No se sube a GitHub. |

## Instalacion

```
pip install -r requirements.txt
```

## Uso

Los pasos se ejecutan en este orden desde la carpeta del proyecto.

```
python experimentos.py        # 30 repeticiones por escenario, o python python experimentos.py 50
python figuras.py
streamlit run app.py
python -m pytest -q
```

## Que permite la aplicacion

1. Elegir el escenario normal, Semana Santa o Navidad, o fijar los pedidos por hora.
2. Cambiar la cantidad de vehiculos, la duracion de la recepcion de pedidos, la semilla aleatoria, las
   repeticiones y el tiempo de descarga.
3. Cerrar una conexion o cambiar su tiempo para estudiar un incidente vial. Los cambios se acumulan y se
   pueden restablecer.
4. Ver las rutas de ida y de regreso por destino. Las rutas se recalculan con Dijkstra cada vez que cambia la red.
5. Simular y comparar la politica actual con la recomendada usando los mismos pedidos en cada repeticion.

## Supuestos principales

- Los tiempos de la matriz son fijos y la descarga en el destino dura cero minutos salvo que se indique otro valor.
- Los pedidos llegan como un proceso de Poisson y el destino es uniforme entre N08, N19, N34 y N40.
- Cada vehiculo lleva un solo pedido y queda libre al regresar a N01. La cola es FIFO.
- Una entrega es puntual si se completa en 75 minutos o menos desde la solicitud, incluida la espera.
- La recepcion termina a la hora indicada y la simulacion continua hasta atender todos los pedidos recibidos.

## Reproducibilidad

Las semillas de las repeticiones se derivan con `SeedSequence` a partir de una semilla base, que por
defecto es 2026. Con la misma semilla se obtienen los mismos pedidos y los mismos resultados.
