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
    5. [hecho]     renderer clasico     sombra + fondo + disco
    6. [pendiente] renderer neural      mismo renderer, red en vez de RK45
    7. [en curso]  Kerr                 agujero negro con espin

## Archivos

### Nucleo (hay que entenderlo)

| Archivo | Lineas | Que hace |
|---|---|---|
| `src/utils/constants.py` | 17 | Unidades naturales G=c=M=1 y los radios caracteristicos. |
| `src/physics/schwarzschild.py` | 60 | La metrica. `r_s`, `b_crit` y los radios los usa todo el pipeline. `f_aux`/`g_tt` los necesita el mapeo pixel->beta del renderer. `V_eff` es de donde sale `b_crit`. Los simbolos de Christoffel NO son decorativos: se usaron para validar el integrador (ver "la ecuacion que se integra"). |
| `src/physics/integrator.py` | 103 | Integra la ecuacion de Binet con RK45. Es la verdad-terreno contra la que se compara todo. |
| `src/physics/disk.py` | 229 | Disco de acrecion: corrimiento al rojo `g`, perfil de temperatura y angulos de cruce con el plano ecuatorial. Lo consume el renderer. |
| `src/physics/kerr.py` | 250 | Metrica de Kerr en Boyer-Lindquist: componentes, inversa en forma cerrada, horizontes, ergosfera, orbitas de fotones, ISCO de Bardeen y el contorno analitico de la sombra. Con `a=0` se reduce identicamente a `schwarzschild.py`. |
| `src/physics/kerr_integrator.py` | 175 | Geodesicas nulas de Kerr en forma hamiltoniana. Ver "por que hamiltoniana" abajo. |
| `src/rendering/classical_renderer.py` | 483 | El renderer. Camara, tabla de rayos por simetria radial, disco, cielo, bloom y tonemap. |
| `src/rendering/starfield.py` | 203 | Proyecta un catalogo de estrellas puntuales a traves de la lente, con imagenes multiples y magnificacion. |
| `scripts/generate_data.py` | 366 | Elige que rayos calcular y guarda el resultado. Ver "muestreo" abajo. |

### Herramientas (se ejecutan y se olvidan)

| Archivo | Lineas | Que hace |
|---|---|---|
| `scripts/verify_dataset.py` | 230 | Comprueba el dataset contra formulas conocidas. Si sale verde, los datos son correctos. |
| `scripts/explore_dataset.py` | 308 | Genera las cuatro figuras de `results/figures/`. |
| `scripts/download_gaia.py` | 141 | Descarga Gaia DR3, o convierte a npz el CSV bajado a mano del archivo web (`--from-csv`). |
| `scripts/gaia_to_panorama.py` | 123 | Rasteriza el catalogo a un panorama equirectangular. Ver "puntos vs fuente extendida". |
| `scripts/render_classical.py` | 102 | Render de Schwarzschild con la tabla. Es el camino rapido. |
| `scripts/render_starfield.py` | 139 | Render del campo de estrellas lensado. |
| `scripts/render_bruteforce.py` | 210 | Un rayo por pixel, sin simetria. Mide el error de la tabla y es el esqueleto de Kerr. |
| `scripts/render_kerr.py` | 175 | Primera simulacion de Kerr sobre el cielo de Gaia, en paralelo. |
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

## La ecuacion que se integra (no es campo debil)

`integrator.py` resuelve `d2u/dphi2 = -u + 3*M*u^2`. Eso NO es una aproximacion:
es la ecuacion de Binet exacta para geodesicas nulas, que sale de reducir la
ecuacion geodesica completa usando los dos vectores de Killing (E y L
conservados) y la simetria esferica, que hace el movimiento exactamente plano.
Los Christoffel quedan absorbidos en las constantes de movimiento.

Comprobado: integrando el sistema completo de 6 EDO
`d2x/dl2 + Gamma dx dx = 0` con `SMetric.christoffel_symbols()` se obtiene la
misma trayectoria **a 1e-11**. Son la misma ecuacion.

`SMetric.weak_field_deflection` (alpha = 2 r_s / b) existe pero **no se llama
desde ningun sitio**. La deflexion exacta diverge cerca del critico: en
beta = 2.59808 vale 13.04 rad, mas de dos vueltas completas, contra 0.77 rad
del campo debil. Esa divergencia logaritmica es el anillo de fotones.

## Puntos vs fuente extendida (por que el lensing "no se veia")

Una fuente PUNTUAL no muestra deformacion: un punto lensado sigue siendo un
punto, solo cambia de sitio y de brillo. Da igual cuantas estrellas haya. Para
que el alabeo se vea hace falta una fuente EXTENDIDA y continua.

Por eso `gaia_to_panorama.py` convierte el catalogo en un panorama: mismas
estrellas reales, pero como campo continuo. Dos sutilezas que costaron:

- El suavizado en longitud tiene que escalar como `blur/sin(theta)`. Cerca de
  los polos los pixeles cubren mucho menos cielo en longitud, y un sigma fijo
  en pixeles deja bandas negras.
- Hay que dividir por `sin(theta)` para pasar de flujo a brillo superficial.

Para validar el lensing sin depender de nada de esto: `--grid` pinta una
rejilla de coordenadas y la deformacion es inmediata de ver.

### Pero el panorama tiene un limite duro: los primeros planos

Un panorama tiene resolucion angular FIJA, y eso lo mata en cuanto la camara
se acerca. Medido en el render heroe de Kerr: el encuadre cubre 2.98 grados en
2560 px, o sea 4.2 arcsec/pixel, mientras que un panorama de 2048x1024 da
633 arcsec/pixel. El cielo queda **151 veces mas grueso que el render**: cada
pixel del panorama se estira a ~150, y las estrellas salen como manchas.

No se arregla agrandando el panorama. El encuadre abarca 1/8240 del cielo, asi
que incluso uno de 8192x4096 aporta solo 2592 pixeles propios para llenar 3.7
millones; harian falta ~310000 px de ancho.

La solucion es pintar las estrellas como **fuentes puntuales** (`--stars`).
Una estrella como punto no tiene resolucion, es una coordenada: da igual cuanto
se acerque la camara. Son los MISMOS datos de Gaia, solo que sin pasar por una
imagen intermedia.

Ojo con dos cosas al usarlo:

- Cada estrella se trata como un disco gaussiano en el PLANO FUENTE, no en la
  imagen. Asi la magnificacion sale sola: donde la lente amplia, mas pixeles
  caen dentro del mismo disco. No hay que calcular ningun jacobiano.
- El sigma tiene que ser comparable al pixel del render (por defecto 1.1
  pixeles). Con sigma mas chico que un pixel las estrellas caen ENTRE centros
  de pixel y desaparecen: es el mismo motivo por el que un telescopio sin PSF
  no registraria nada.

Cuando conviene cada uno: panorama para planos generales (barato y muestra el
alabeo de estructura extendida), puntos para primeros planos (nitidos a
cualquier zoom). Ademas los puntos evitan la distorsion de proyectar una esfera
en un plano, que es lo que ensucia el panorama cerca de los polos.

### El radio de Einstein: por que el lensing de las estrellas no se veia

Durante un tiempo parecio que las estrellas no se lenseaban. No era asi: el
problema era el ENCUADRE.

Con la fuente en el infinito y la camara a r_obs, hay anillo de Einstein cuando
la deflexion iguala al angulo con que sale el rayo:

    4M/b = b/r_obs      ->      b_E = sqrt(4 M r_obs)

Con r_obs = 1000 M eso da 63.2 M. Comprobado poniendo UNA estrella sintetica
justo detras del agujero: sale un anillo limpio de radio medido **64.7 M**, un
2.4% por encima (campo debil subestima la deflexion).

Y ahi esta el problema:

| | radio |
|---|---|
| sombra | 5.20 M |
| **encuadre que se venia usando** | **26 M** |
| anillo de Einstein | 64.7 M |

O sea que 26 M cae en TIERRA DE NADIE: pasado el anillo de fotones y muy corto
del de Einstein. Todo el lensing visible quedaba fuera del cuadro.

Con `--half-width 80` aparecen los arcos concentricos de estrellas estiradas
tangencialmente, que es la firma del lensing fuerte.

Notar que b_E crece con sqrt(r_obs), asi que acercar la camara encoge el anillo
de Einstein y lo acerca a la sombra. Para que ambos entren comodos en el mismo
cuadro hay que elegir r_obs a conciencia, no dejarlo en 1000 M por inercia.

### Que parche del cielo queda detras

`--behind-galactic-center` gira el cielo para que Sgr A* (RA 266.417,
DEC -29.008) quede JUSTO DETRAS del agujero:

    centro galactico  ->  agujero negro  ->  camara

Importa porque la lente solo puede deformar lo que tenga detras. El centro
galactico tiene **245 estrellas/deg^2** contra 74.9 de media del cielo, o sea
3.3 veces mas material que lensear. Sin orientar, la zona lensada cae en
cualquier parte, normalmente vacia.

`--behind RA DEC` permite apuntar a cualquier otro punto.

Nota aparte: que el borde de la sombra salga circular en Schwarzschild es
CORRECTO, no un bug. La asimetria requiere espin, y aun asi es debil (ver Kerr).

## Kerr: que se rompe con el espin

1. La simetria pasa de esferica a AXIAL. El movimiento deja de ser plano, asi
   que la reduccion de Binet no existe y la tabla 1D del renderer no sirve: la
   trayectoria ya no depende solo de beta. Hay que integrar por pixel.
2. Hacen falta TRES constantes: E, L_z y la de CARTER `Q`, que no viene de una
   simetria evidente sino de que Hamilton-Jacobi separa en estas coordenadas.
3. `g_tphi != 0`: estacionaria pero no estatica. De ahi el arrastre de marcos y
   la ergosfera, que queda FUERA del horizonte.
4. La sombra deja de ser circular: se desplaza y se aplana.

### Por que formulacion hamiltoniana

Las ecuaciones separadas de primer orden llevan `+- sqrt(R(r))` y
`+- sqrt(Theta(theta))`, con un signo que hay que voltear A MANO en cada punto
de retorno. Equivocarse de rama ahi es el bug clasico de los trazadores de
Kerr, y falla justo donde mas importa: cerca del anillo de fotones, donde el
rayo da varias vueltas y cruza muchos retornos.

La forma hamiltoniana no tiene raices, los retornos salen solos porque `p_r`
pasa por cero de forma continua. Cuesta 5 EDO en vez de 3. No se calculan
Christoffel: `p_t` y `p_phi` ya son constantes por los Killing.

Comprobacion interna gratis: para rayos nulos `H = 1/2 g^ab p_a p_b` vale CERO
siempre, y no se impone en ningun sitio. Si se aleja de cero, la integracion se
degrado. (En rayos capturados `H` se dispara, pero eso es porque se evalua en
el horizonte donde `Delta -> 0` y Boyer-Lindquist es singular: es esperado.)

### El disco en Kerr

No basta con reusar `disk.py`: el disco cambia de tres formas.

1. **El borde interior lo fija el espin.** El ISCO va de 6M (a=0) a 1.24M
   (a*=0.998). Un disco alrededor de un agujero que gira rapido llega mucho mas
   adentro, donde el gas orbita mas rapido y el corrimiento es brutal: `g` en el
   ISCO cae de 0.71 a 0.093.
2. **La velocidad orbital lleva el arrastre**:
   `Omega = +-sqrt(M)/(r^{3/2} +- a sqrt(M))`. A r=6M con a*=0.9 el gas prograde
   gira MAS DESPACIO que en Schwarzschild (0.0641 vs 0.0680) y el retrogrado
   mas rapido (0.0725).
3. **El corrimiento no se separa en factores.** En Schwarzschild
   `g = sqrt(1-3M/r)/(1-Omega b)` separa gravedad y Doppler. Aqui
   `g_tphi != 0` los mezcla y hay que sacar `u^t` de la normalizacion completa:
   `g = 1/[u^t (1 - Omega xi)]`. Comprobado que con a=0 coincide con `disk.py`
   a 2e-16.

Cuidado con un detalle que quema la imagen si se pasa por alto: el denominador
`u^t (1 - Omega xi)` es la energia del foton medida por el gas, y puede pasar
por CERO. Cuando sale <= 0 esa combinacion (r, xi) es imposible (ese elemento
de disco no pudo emitir ese foton hacia nosotros) y hay que devolver emision
nula, no un `g` gigante: el brillo va con `g^4`, asi que un solo pixel mal
tratado se lleva toda la escala. En rayos trazados de verdad la condicion
`R(r) >= 0` lo garantiza sola: medido sobre un render completo, `g` se queda en
[0.026, 1.49] y no aparece ni un caso prohibido.

### Trampa al comparar espines: la normalizacion del disco

`disk.temperature_profile` esta normalizada a MAXIMO 1, y ese maximo cae en
`x_pico = (49/36) x_in`. Al subir el espin el ISCO baja, el pico se mete hacia
dentro y se vuelve mucho mas caliente, asi que TODO el resto del disco queda
pequeño en relacion a el. Renderizando varios espines sin cuidado sale que el
disco se APAGA al girar mas rapido, y eso es falso: la eficiencia radiativa de
un disco delgado sube del 5.7% (a=0) al 32% (a*=0.998), o sea que a igual tasa
de acrecion un agujero que gira rapido brilla MAS.

Medido: a r=18 M, donde `g` es practicamente igual para todos los espines, el
brillo relativo caia de 0.19 a 0.003 (factor 64) solo por la normalizacion.

Por eso `render_kerr.py --norm-r 12` renormaliza el perfil en un radio fijo en
vez de en el maximo. Con eso el disco exterior queda igual entre espines y las
diferencias se ven donde de verdad estan: en la region interior. Para comparar
espines hay que usarlo siempre.

### Cuanto varia la sombra

Vista de canto, en unidades de M:

| a* | ancho | alto | aplanamiento | desplazamiento |
|---|---|---|---|---|
| 0.0 | 10.392 | 10.392 | 0% | 0 |
| 0.6 | 10.148 | 10.392 | 2.4% | 1.24 |
| 0.9 | 9.669 | 10.392 | 7.0% | 1.99 |
| 0.998 | 9.105 | 10.392 | 12.4% | 2.44 |

El aplanamiento es DEBIL: 12% en el espin casi extremo. El desplazamiento del
centro se nota mucho mas. El alto en beta no cambia nunca: el espin solo afecta
al eje alpha. Y de polo la sombra vuelve a ser circular sea cual sea el espin.

Convenio de Bardeen: `alpha` es PERPENDICULAR al eje de rotacion proyectado
(sale de `xi = L_z/E`) y `beta` PARALELO (sale de la constante de Carter).

## Rendimiento: para que sirve la red en realidad

Medido en 6 nucleos, CPU:

| | coste |
|---|---|
| RK45 Schwarzschild (Binet, 2 EDO) | 2.24 ms/rayo |
| Kerr hamiltoniano, version inicial | 55 ms/rayo (11 en paralelo) |
| Kerr tras optimizar (ver abajo) | **0.10 ms/rayo** |
| MLP 64x3 / 128x4 / 256x4 | 3.3 / 1.1 / 0.1 M puntos/s |
| `build_table` actual | 1.05 M puntos/s efectivos |

### Como se acelero Kerr 110x

El perfil mostraba dos cosas: `inverse_metric_matrix` se llamaba 5 veces por
evaluacion del lado derecho (1 + 4 para las derivadas numericas), y
`np.asarray` aparecia 825k veces por un puñado de rayos. O sea, casi todo el
tiempo era overhead de numpy sobre escalares y derivadas por diferencias.

1. **Derivadas analiticas** de `g^ab` en r y theta (`inverse_components_and_
   derivs`): elimina 4 de las 5 evaluaciones. Validadas contra diferencias
   centradas de cuarto orden a 4e-11.
2. **Floats de Python en vez de numpy** y sin construir matrices 4x4: el
   contraido `g^ab p_a p_b` se escribe a mano aprovechando que Kerr solo tiene
   5 componentes no nulas.
   → juntas dan **5.4x**, y de paso MAS precision (el borde de la sombra pasa
   de 1e-7 a 1e-8, porque desaparece el error de truncamiento del paso finito).
3. **Integracion por lotes** (`trace_batch`): Dormand-Prince vectorizado con
   paso adaptativo POR RAYO, avanzando todos los rayos vivos a la vez y
   sacando del lote a los que terminan. Asi el overhead del interprete se
   reparte entre miles de rayos en vez de pagarse uno por uno.
   → **20.8x adicional**. Clasificacion captura/escape identica a la version
   serie en 400/400 rayos, y direcciones que difieren 1.4e-3 grados como mucho.

Resultado: un fotograma de 1920x1080 con supersampling x2 pasa de ~26 horas a
**~14 minutos**. Ojo con la conclusion: eso NO quita el argumento de la red,
pero si lo cambia. Kerr offline ya es viable sin red; lo que la red sigue
habilitando es tiempo real, GPU y las derivadas exactas.

### Benchmark medido (`scripts/benchmark_kerr.py`)

Intel de 12 hilos, 12 procesos, a*=0.9, con disco y fondo de Gaia. Los tiempos
en bruto quedan en `results/benchmarks/kerr_benchmark.json` junto al entorno y
la fecha, para poder detectar regresiones despues de tocar el integrador.

| resolucion | rayos | tiempo | ms/rayo |
|---|---|---|---|
| 320x180 | 57 600 | 13.7 s | 0.238 |
| 480x270 | 129 600 | 17.6 s | 0.136 |
| 640x360 | 230 400 | 25.1 s | 0.109 |
| 960x540 | 518 400 | 49.9 s | 0.096 |
| 1280x720 | 921 600 | 86.6 s | 0.094 |

El coste por rayo BAJA con la resolucion y se estabiliza cerca de 0.094 ms: a
lotes pequeños el arranque de los procesos y el overhead fijo pesan mas que el
trazado. O sea que a alta resolucion es cuando mejor rinde, justo al reves que
la intuicion. Extrapolado: 1920x1080 ~3.6 min, con supersampling x2 ~14 min,
4K con supersampling x2 ~57 min.

**Resultado contraintuitivo:** para el renderer de Schwarzschild que ya existe,
una red en CPU casi no gana. Un MLP 128x4 empata con RK45+tabla y uno de 256x4
es 10x mas LENTO. El motivo es que la simetria radial ya amortiza de maravilla:
un rayo de 2.24 ms sirve para miles de pixeles.

Donde la red si gana por goleada es SIN simetria, o sea Kerr: por pixel a
1920x1080 con supersampling son ~5.2 h en Schwarzschild y **~26 h en Kerr**
incluso paralelizado, contra ~18 s de una red. Mas dos ganancias que no son
velocidad: GPU (estos numeros son CPU, y scipy no paraleliza en GPU) y
derivadas EXACTAS por autodiff, que `starfield.py` necesita para la
magnificacion y que con RK45 solo salen por diferencias finitas ruidosas.

Implicacion de diseno: conviene una red PEQUENA (64x3 va 3x mas rapido que
128x4) y suficientemente precisa, no grande.

**La tabla solo compensa a alta resolucion.** Su ventaja es el factor de
amortizacion (pixeles / rayos de tabla). A 160x90 con `n_radial=12000` es mas
LENTA que la fuerza bruta; a 1920x1080 con supersampling es ~200x mas rapida.
Su error cae como ~1/n_radial (la busqueda en beta es de vecino mas cercano) y
se concentra en el borde de la sombra, donde la imagen es discontinua de verdad.

## Comandos

    python scripts/generate_data.py --n-rays 20000 -j 8
    python scripts/verify_dataset.py
    python scripts/explore_dataset.py
    PYTHONPATH=src python -m pytest tests/ -q

    # cielo real
    python scripts/download_gaia.py --from-csv data/raw/tu_descarga.csv
    python scripts/gaia_to_panorama.py

    # renders
    python scripts/render_classical.py --sky-image data/raw/gaia_panorama.png \
        --inc 88 --n-radial 12000 --ss 2 -W 1920 -H 1080
    python scripts/render_classical.py --grid --no-disk      # validar el lensing
    python scripts/render_bruteforce.py --compare -W 160 -H 90
    python scripts/render_kerr.py --compare-spin 0.0 0.9 0.998 --outline

`data/` y `results/` no se versionan: se regeneran con estos comandos.

## Notas

- `docs/dataset_generation.md` describe una version anterior del muestreo y esta
  desactualizado respecto a este mapa.
- Editar en Windows convierte los finales de linea y ensucia los diffs. Se evita
  con un `.gitattributes` que contenga `* text=auto eol=lf`.
