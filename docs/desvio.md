# Desvio: registro completo y por que hay que corregir el rumbo

Documento de cierre de una sesion larga. Recoge TODO lo construido, con sus
metricas y sus errores, y termina con una valoracion honesta: parte del trabajo
se alejo del objetivo del proyecto.

## El objetivo

> Usar redes neuronales para **resolver las ecuaciones diferenciales de las
> geodesicas** (las trayectorias de los fotones), de modo que calcular una
> imagen de un agujero negro sea mucho mas rapido. Mas adelante, animaciones,
> tambien potenciadas con aprendizaje profundo.

La palabra que importa es **resolver las EDO**. La red sustituye al integrador
numerico, no al renderer.

---

# PARTE 1 — Lo que SI cumple el objetivo

## 1.1 Red de geodesicas de Schwarzschild

Aprende `(beta, phi) -> u_tilde`, que es exactamente la solucion de la ecuacion
de Binet `d2u/dphi2 = -u + 3M u^2`. Es una red que integra la EDO.

| | |
|---|---|
| Arquitectura | MLP 64x3, tanh, **8 897 parametros** |
| Entrenamiento | 6 000 epocas en GPU, 53 min |
| Dataset | 20 000 rayos x 192 puntos = 3 840 000 muestras |
| **MAE en u** | **0.00004** |
| Peor caso | 0.0069 |
| MSE de prueba | 9.6e-9 |
| Velocidad | 3.2 M pts/s en CPU, **95 M pts/s en GPU fp16** |
| vs RK45 | **51x mas rapida por punto** |

Particion **por rayo**, nunca por punto: los 128 puntos de un rayo son casi la
misma curva, y repartirlos filtraria informacion dando una validacion falsa.

### Progresion del error

| Etapa | MAE | Peor |
|---|---|---|
| Version inicial | 0.0146 | 0.54 |
| Tras arreglar la precision | 0.00058 | 0.034 |
| 64x3 con 1 500 epocas | 0.00015 | 0.013 |
| GPU + 20k rayos + 6 000 epocas | **0.00004** | **0.0069** |

**365 veces mejor** que el punto de partida, con la misma red de 8.9k parametros.

### El hallazgo que valio por todo lo demas: float32 cerca del critico

Cerca de `beta_crit = 2.598076` el ULP de float32 vale 2.4e-7, del mismo orden
que la distancia al critico de los rayos que forman el anillo de fotones.
Medido sobre el dataset:

- float32 colapsaba **576 valores distintos** de `|beta - beta_crit|` **en 22**
- dejaba **10 112 muestras en CERO exacto**
- metia hasta **2.34 de error** en la entrada `s`, cuyo rango es -7 a 1

Manteniendo `beta` en float64 por toda la cadena, el MSE de validacion mejoro
**128 veces**. Mas que todos los cambios de arquitectura juntos.

### Decisiones de diseño, cada una salida de medir donde fallaba

1. **Predecir la razon `u/u_max`, no `u`.** `u_max` se conoce exacto: el
   periastro para los rayos que escapan (raiz cerrada de `u^2(1-u) = 1/beta^2`,
   rama trigonometrica k=1, verificada contra `np.roots`) y el horizonte, u=1,
   para los que caen. Sin esto el peor error estaba en el **campo lejano** y no
   cerca del critico, porque un rayo con beta=15 tiene todos sus valores por
   debajo de u=0.07 y una perdida absoluta los ignoraba.
2. **Entrada `s = log10|beta - beta_crit|`**, que es la variable en la que la
   deflexion crece de forma regular y en la que `generate_data.py` ya muestrea.
3. **Entradas `|phi|/beta` y `log10(1+|phi|)`.** Cada rayo barre un rango de phi
   distintisimo (captura profunda +-0.039, cerca del critico +-18), asi que una
   sola normalizacion global aplastaba los casi radiales al 0.2% del rango.
4. **`|phi|` y no `phi`**: la trayectoria es simetrica respecto al periastro
   (comprobado, `|u(phi) - u(-phi)| < 1.6e-4` sobre 303 rayos).
5. **La captura no se aprende**: `beta_crit` se conoce en forma cerrada.

### Capacidad frente a tiempo de entrenamiento

| Config | Params | MAE | Velocidad |
|---|---|---|---|
| 128x4, 400 epocas | 50 689 | 0.00025 | 1.1 M/s |
| **64x3, 1 500 epocas** | **8 897** | **0.00015** | **3.3 M/s** |

La red pequeña entrenada mas tiempo gana a la grande en **precision y
velocidad**. El limite era optimizacion, no capacidad.

## 1.2 Fisica de Kerr

`src/physics/kerr.py` y `kerr_integrator.py`. Todo validado.

| Prueba | Resultado |
|---|---|
| Reduccion a Schwarzschild con a=0 | **0.0e+00** componente a componente |
| `g . g^-1 = I` | 3.3e-16 |
| `det g = -Sigma^2 sin^2(theta)` | 1.7e-15 |
| Horizonte, fotones (3M/M/4M), ISCO (6M/M/9M) | exactos |
| Ergosfera (2M en el ecuador, cualquier espin) | exacta |
| Borde de sombra trazado vs analitico | **1e-8** |
| Test extremo a=M: borde plano en alpha=-2M | 8.9e-10 |
| Hamiltoniano `H = 1/2 g^ab p_a p_b` (debe ser 0) | ~1e-7 |
| Corrimiento del disco vs `disk.py` con a=0 | 2e-16 |

Formulacion **hamiltoniana** y no las ecuaciones separadas con la constante de
Carter, porque aquellas llevan raices cuadradas cuyo signo hay que voltear a
mano en cada punto de retorno: el bug clasico de los trazadores de Kerr, y falla
justo en el anillo de fotones.

### Optimizacion del integrador: 110x

| Cambio | Ganancia |
|---|---|
| Derivadas analiticas de `g^ab` + floats de Python sin matrices | 5.4x |
| Integracion por lotes (Dormand-Prince vectorizado, paso por rayo) | 20.8x |
| **Total** | **110x** (11 -> 0.10 ms/rayo) |

Y **mas preciso**: el borde de la sombra paso de 1e-7 a 1e-8 al desaparecer el
error de truncamiento de las diferencias finitas.

## 1.3 Bugs de fisica encontrados y corregidos

**El corrimiento al rojo de Kerr podia explotar.** El denominador de
`g = 1/[u^t (1 - Omega xi)]` pasa por cero, y como el brillo va con `g^4` un
solo pixel se llevaba toda la escala (llegaba a `g = 1e9`). Ese cero significa
combinacion `(r, xi)` **imposible**: ese elemento de disco no pudo emitir ese
foton hacia nosotros. Ahora devuelve emision nula. En rayos trazados de verdad
no ocurre nunca porque `R(r) >= 0` lo garantiza: g medido queda en
**[0.026, 1.49]**.

**Los renders de Kerr salian espejados verticalmente.** `beta` entra como
`p_theta`, y theta crece hacia el SUR, asi que `beta > 0` es una fuente al sur;
el renderer ponia el beta mayor arriba. Comprobado de tres formas: una estrella
puesta 8 grados al norte aparecia abajo; en campo lejano la direccion asintotica
con beta>0 tiene componente negativa sobre el eje de giro; y el render con
espin 0 solo coincidia con `render_classical.py` al espejarlo.

**Fuga de memoria en el trazado por lotes.** `trace_batch` reservaba las siete
etapas de Dormand-Prince de golpe, `(7, N, 5)`: con 1.2M rayos por proceso eran
336 MB cada uno, y un render de 2560x1440 con supersampling moria tras 38
minutos. Troceado en sub-lotes de 150k, y ademas quedo **mas rapido** (21.5 min
frente a 38) por localidad de cache.

**La comparacion de espines mentia.** `disk.temperature_profile` normaliza a
maximo 1, y ese maximo esta en `(49/36) x_in`; al subir el espin el ISCO baja,
el pico se mete hacia dentro y el disco exterior parece apagarse. Medido a
r=18M, donde `g` es igual para todos los espines, el brillo caia **factor 64**
solo por la normalizacion. En realidad la eficiencia radiativa **sube** del 5.7%
al 32%. De ahi `--norm-r`.

## 1.4 El radio de Einstein: por que el lensing no se veia

Con la fuente en el infinito y la camara a `r_obs`:

    4M/b = b/r_obs   ->   b_E = sqrt(4 M r_obs) = 63.2 M

Medido poniendo una estrella sintetica justo detras: anillo limpio de **64.7 M**
(2.4% por encima; campo debil subestima).

| | radio |
|---|---|
| sombra | 5.20 M |
| **encuadre que se usaba** | **26 M** |
| anillo de Einstein | 64.7 M |

Los 26 M caian en **tierra de nadie**. Con `--half-width 80` aparecen los arcos.
Y `b_E` crece con `sqrt(r_obs)`, asi que acercar la camara junta las dos escalas:
la razon `b_E/b_sombra` vale 27x a r_obs=5000 M pero solo **5.4x a 200 M**.

## 1.5 Cielo real de Gaia

| Catalogo | Estrellas |
|---|---|
| G<10 | 482 106 |
| G<12 | 3 087 821 |

**Panorama frente a fuentes puntuales.** Un panorama tiene resolucion angular
fija: en un primer plano el encuadre cubre 1/8240 del cielo, asi que uno de
4096x2048 aporta 648 pixeles propios para llenar 3.7 millones — **151 veces mas
grueso que el render**. Una estrella como punto no tiene resolucion.

Trampa al usar puntos: el sigma debe ser comparable al pixel (por defecto 1.1
px). Con sigma menor que un pixel las estrellas caen **entre** centros y
desaparecen; el primer intento con 6 arcsec contra pixeles de 16.8 dio un cielo
completamente negro.

## 1.6 Rendimiento medido

| | |
|---|---|
| RK45 Schwarzschild (2 EDO) | 2.24 ms/rayo |
| Kerr hamiltoniano, tras optimizar | 0.094-0.12 ms/rayo |
| Red 64x3 en CPU / GPU fp32 / GPU fp16 | 3.2 / 58.9 / **95.0** M pts/s |
| Plano general 2560x1440 ss2 (14.7 M rayos) | 29.2 min |

**GPU:** `torch 2.13.0+cu126`, RTX 3050 Laptop 4 GB. fp32 4.39 TFLOPS frente a
fp64 **0.11** — las GeForce capan la doble precision a 1/64.

**El trazado de rayos NO se puede acelerar en esta GPU.** Se porto
`trace_batch` a CUDA y quedo funcionalmente correcto (clasificacion identica en
40 000 rayos, direcciones a 2e-6 grados), pero: en float64 va igual que la CPU
por el capado, y en float32 la tolerancia `rtol=1e-6` queda al borde de la
precision simple, el paso adaptativo se atasca y tarda **5 veces mas**. Un
integrador adaptativo necesita precision, y esta GPU solo es rapida en la que no
alcanza.

## 1.7 Imagenes producidas

En `results/figures/`:

- `render_clasico.png`, `render_disco.png` — renderer clasico de Schwarzschild
- `render_gaia_real.png`, `render_gaia_sharp.png`, `render_gaia_hq.png` — cielo
  real de Gaia, panorama y puntual, hasta 1920x1080 con supersampling
- `kerr_sombra.png` — contorno analitico de la sombra segun espin e inclinacion
- `kerr_hero.png` — 2560x1440 ss2, 14 745 600 rayos, 29.2 min
- `kerr_suite/` — 24 renders comparando espin, inclinacion, sentido del disco,
  tamaño, encuadre, centro galactico y ganancia, mas sus laminas
- `render_neural.png` y `render_neural_comparacion.png` — neural contra clasico

Benchmarks crudos con entorno y fecha en `results/benchmarks/*.json`.

---

# PARTE 2 — El desvio

## 2.1 Que se hizo mal

Para Kerr **se cambio el problema**. En vez de que la red resolviera la ecuacion
geodesica, se la puso a predecir las **salidas del renderer**:

    KerrSkyNet:  (alpha, beta, espin, inclinacion) -> direccion de salida
    KerrDiskNet: (alpha, beta, espin, inclinacion, k) -> (r, phi, existe)

Eso **no es resolver una EDO**. Es aprender un mapa imagen-a-imagen de extremo a
extremo. Se justifico con "el renderer solo usa tres cosas de cada rayo, asi que
reconstruir la geodesica entera seria pagar de mas". Como optimizacion era
razonable; como rumbo del proyecto, no: la red dejo de ser un integrador y paso
a ser un interpolador de imagenes.

## 2.2 Lo que costo ese desvio

**Un muro de precision que no es de entrenamiento sino estructural.** El mapa
`(alpha, beta) -> direccion` es **caotico** cerca del anillo de fotones: el rayo
da muchas vueltas y la salida cambia a saltos. Una red suave no puede
representarlo por muchas epocas que se le den.

| Distancia al borde de la sombra | Error angular |
|---|---|
| < 0.001 M | **45 grados** |
| 0.1 - 1 M | 16 |
| > 20 M | 2.7 |

**Precision final, evaluada sobre pixeles reales** (espin 0.777, inclinacion
63.3, nunca vistos):

| | absoluto | residual |
|---|---|---|
| media | 0.148 | 0.162 |
| mediana | 0.124 | **0.063** |
| p95 | 0.336 | **0.178** |
| sin red (recta) | 7.25 | — |

Un pixel a 1920 de ancho abarca 0.0048 grados, asi que la mediana de 0.063 son
**13 pixeles de desplazamiento**. Comparado con el **0.00004** de Schwarzschild,
que si resuelve la EDO, la diferencia de calidad es abismal.

La formulacion residual (predecir la desviacion respecto a la recta) mejoro la
mediana y los percentiles ~2x, pero **no** reprodujo el factor 128 que el truco
analogo dio en Schwarzschild. Alli el problema era de **escala**; aqui todos los
objetivos ya son vectores unitarios, asi que no habia desbalance que corregir.

**Parches que no existirian si la red resolviera la EDO:**

- Hubo que añadirle a la red del disco una salida de "**existe el cruce?**". Sin
  ella, los pixeles que no cruzan el disco caian en la extrapolacion, devolvian
  radios plausibles y el disco salia como un **borron blanco sobre toda la
  imagen**. Con la salida nueva acierta el 98.84%, error 0.236 M en radio y 6.18
  grados en angulo — pero es una pregunta que un integrador contesta solo.
- Hubo que **precalcular por configuracion**: la geometria depende de (espin,
  inclinacion), asi que mover la camara obligaba a rehacerla. Una red que
  resolviera la EDO serviria para cualquier camara sin reentrenar.
- Una reentrenada del cielo **regreso a 7.4 grados**, practicamente igual a no
  usar red (7.25), sin explicacion todavia. Sintoma de que el problema esta mal
  planteado, no solo mal ajustado.

## 2.3 Lo que si funciono del renderer neural

Para que quede constancia, porque la parte de ingenieria vale:

| | Tiempo | fps |
|---|---|---|
| Trazado clasico de rayos | 216 s | 0.005 |
| Neural sin optimizar | 18.5 s | 0.05 |
| Neural en GPU, sin tabla | 0.124 s | 8 |
| **Neural con tabla precalculada** | **0.027 s** | **38** |
| Motor solo cielo, GPU | 0.011 s | **90** |
| Motor con disco, GPU | 0.084 s | 12 |

Hasta **8 131 veces** mas rapido que trazar rayos. Y el cuello de botella nunca
fue la red: a 640x360 el KD-tree costaba 1017 ms y las entradas en float64 otros
358, mientras la red tardaba 12. Llevarlo todo a GPU en float32 fue lo que abrio
el tiempo real.

Aviso honesto sobre esa cifra: los 38-90 fps son del **motor en GPU**. El visor
completo con matplotlib baja a ~8 fps, porque traer la imagen a CPU, aplicar el
tonemap y redibujar se lleva el resto. Medir el motor y presentarlo como fps del
visor fue engañoso.

## 2.4 Como se hace bien

Kerr tiene una estructura que vuelve esto **tan limpio como Schwarzschild**, y
no se uso. En **tiempo de Mino** (`dlambda = dtau/Sigma`) las ecuaciones se
**desacoplan exactamente**:

    (dr/dlambda)^2     = R(r; xi, eta)
    (dtheta/dlambda)^2 = Theta(theta; xi, eta)

Ya no estan mezcladas por `Sigma`. La trayectoria depende de **solo dos
constantes** (`xi = L_z/E`, `eta = Q/E^2`), igual que en Schwarzschild dependia
de una sola (`beta`). Asi que la red correcta es

    red(xi, eta, lambda) -> (r, theta)

Tres entradas, dos salidas, y **es la solucion de la ecuacion diferencial**:
el analogo exacto de la de Schwarzschild. `phi` y `t` salen por cuadratura.

Ventajas frente a lo que hay hoy:

- Es una EDO **suave**, no un mapa caotico
- El caos del anillo de fotones queda donde debe: en cuantas vueltas da el
  integrador, no dentro de la red
- Sirve para **cualquier camara, inclinacion y encuadre** sin reentrenar
- Se valida contra el integrador punto a punto, como se hizo con Schwarzschild
- Desaparecen los parches: no hay que preguntar "existe el cruce", ni
  precalcular por configuracion

## 2.5 Que conservar y que tirar

**Conservar:**

- Toda la fisica: `kerr.py`, `kerr_integrator.py`, validaciones incluidas
- La red de Schwarzschild y su metodologia entera (particion por rayo,
  normalizacion por el periastro, cuidado con float32 cerca del critico)
- Los renderers clasicos y la infraestructura de benchmarks
- El motor en GPU (`neural_live.py`): la ingenieria de tiempo real sirve igual,
  solo cambia que la red debajo sea otra
- El generador de datos de Kerr, reaprovechable cambiando que se guarda

**Rehacer:**

- `KerrSkyNet` y `KerrDiskNet`: el planteamiento es incorrecto
- `generate_kerr_data.py`: debe guardar **trayectorias** `(xi, eta, lambda) ->
  (r, theta)`, no salidas del renderer
- `train_kerr_model.py`: en consecuencia

---

## Conclusion

La mitad de Schwarzschild cumple el objetivo y lo cumple bien: una red de 8.9k
parametros que resuelve la ecuacion geodesica con **MAE 4e-5** y va **51 veces**
mas rapida que RK45.

La mitad de Kerr se desvio. Se construyo un interpolador de imagenes en vez de
un integrador, y todos los sintomas posteriores — el muro de 45 grados en el
borde, la salida de "existe el cruce", la geometria precalculada por
configuracion, la regresion inexplicada a 7.4 grados — son consecuencias de esa
decision, no problemas independientes.

La correccion no es entrenar mas ni una red mas grande: es **volver a plantear
el problema como lo que es**, resolver la EDO en tiempo de Mino con `(xi, eta)`
como parametros.
