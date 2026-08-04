# Mapa del proyecto

Una pagina para no tener que recordar donde esta cada cosa.

## Que hace esto

Un foton que pasa cerca de un agujero negro se desvia. Calcular cuanto se desvia
requiere integrar una ecuacion diferencial, y eso es lento. Entrenamos una red
neuronal que da la misma respuesta al instante, para poder renderizar el agujero
negro en tiempo real.

## La idea que lo simplifica todo

Todo se mide en unidades del radio de Schwarzschild `r_s = 2GM/c^2`:

    beta = b / r_s        parametro de impacto (la entrada)

En esas unidades la ecuacion no contiene la masa. Un agujero negro de 10 masas
solares y Sagitario A* producen la misma imagen, solo cambia la escala. Por eso
**un unico dataset sirve para cualquier masa** y la red nunca ve `M`.

Valor clave: `beta_crit = 3*sqrt(3)/2 = 2.598076`. Por debajo el foton cae, por
encima escapa. Es exacto y no depende de la masa.

## Flujo

    1. generate_data.py   ->  data/processed/geodesics_dataset.npz
    2. verify_dataset.py  ->  20 comprobaciones (deben pasar todas)
    3. explore_dataset.py ->  results/figures/*.png
    4. [pendiente] entrenar la red      (beta, phi) -> u_tilde
    5. [pendiente] renderer clasico     sombra + fondo + disco
    6. [pendiente] renderer neural      mismo renderer, red en vez de RK45

## Archivos

### Nucleo (hay que entenderlo)

| Archivo | Lineas | Que hace |
|---|---|---|
| `src/utils/constants.py` | 17 | Unidades naturales G=c=M=1 y los radios caracteristicos. |
| `src/physics/schwarzschild.py` | 60 | La metrica. `r_s`, `b_crit` y los radios los usa todo el pipeline. `f_aux`/`g_tt` los necesita el mapeo pixel->beta del renderer. `V_eff` es de donde sale `b_crit`. Los simbolos de Christoffel quedan para el dia que haga falta la ecuacion geodesica completa en 4 coordenadas. |
| `src/physics/integrator.py` | 103 | Integra la ecuacion de Binet con RK45. Es la verdad-terreno contra la que se compara todo. |
| `src/physics/disk.py` | 229 | Disco de acrecion: corrimiento al rojo `g`, perfil de temperatura y angulos de cruce con el plano ecuatorial. Lo consume el renderer. |
| `scripts/generate_data.py` | 366 | Elige que rayos calcular y guarda el resultado. Ver "muestreo" abajo. |

### Herramientas (se ejecutan y se olvidan)

| Archivo | Lineas | Que hace |
|---|---|---|
| `scripts/verify_dataset.py` | 230 | Comprueba el dataset contra formulas conocidas. Si sale verde, los datos son correctos. |
| `scripts/explore_dataset.py` | 308 | Genera las cuatro figuras de `results/figures/`. |
| `tests/test_disk.py` | 121 | 43 tests del modulo de disco contra valores en forma cerrada. |

## Que hay dentro del dataset

`data/processed/geodesics_dataset.npz`. Dos compartimentos:

**Por punto de trayectoria** — lo que come la red: `beta`, `phi`, `u_tilde`,
`ray_index`. Con `phi = 0` en el periastro, asi que la curva no depende de donde
se empezo a integrar y es simetrica en `phi`.

Hace falta la trayectoria entera, y no solo la desviacion total, porque para
saber si un rayo atraviesa el disco hay que saber por donde paso, no solo donde
acabo.

**Por rayo** — resumenes utiles para verificar y para la cabeza de clasificacion:

| Llave | Que es |
|---|---|
| `ray_beta` | El parametro de impacto. La entrada de la red. |
| `ray_delta_phi` | Cuanto se desvio el rayo, en radianes. La salida. `NaN` si el foton cayo. |
| `ray_status` | 0 escapo, 1 capturado, 2 truncado. |
| `ray_r_min_over_rs` | Lo mas cerca que paso del agujero. |

Ademas `config_json` guarda la semilla y todos los parametros, para poder
reproducir el dataset exactamente.

## Muestreo: por que no son rayos al azar

La desviacion se dispara cuando `beta` se acerca a `beta_crit`. Si se eligieran
rayos uniformemente, casi todos caerian en la zona aburrida donde la desviacion
es casi cero, y la red no aprenderia la parte interesante.

Por eso los rayos se reparten en cuatro grupos: dos concentrados a ambos lados
del valor critico, uno en campo lejano y uno de captura profunda. Se muestrea en
escala logaritmica de la distancia al critico, que es la variable en la que la
desviacion crece de forma regular.

## Comandos

    python scripts/generate_data.py --n-rays 20000 -j 8
    python scripts/verify_dataset.py
    python scripts/explore_dataset.py
    PYTHONPATH=src python -m pytest tests/ -q

`data/` y `results/` no se versionan: se regeneran con estos comandos.

## Notas

- `docs/dataset_generation.md` describe una version anterior del muestreo y esta
  desactualizado respecto a este mapa.
- Editar en Windows convierte los finales de linea y ensucia los diffs. Se evita
  con un `.gitattributes` que contenga `* text=auto eol=lf`.
