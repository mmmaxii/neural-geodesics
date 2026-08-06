# Kerr en tiempo de Mino: qué se aprendió

Notas de la etapa que rehizo el surrogate de Kerr. Registra sobre todo los
resultados NEGATIVOS y los bugs, que es lo que no se deduce leyendo el código.

## El punto de partida

El primer surrogate de Kerr aprendía el mapa píxel → salida del renderer
(dirección de escape, cruce con el disco). Ese mapa es caótico cerca del anillo
de fotones, necesitaba una cabeza-parche para decidir "¿existe el cruce?", y se
quedaba en ~13 px de error. No resolvía la ecuación diferencial: interpolaba
renders.

La corrección era usar tiempo de Mino (`dλ = dτ/Σ`), donde las ecuaciones de
Kerr para `r` y `θ` se desacoplan exactamente y la trayectoria depende solo de
`(ξ, η, a)` — sin cámara, sin encuadre.

## El resultado, medido

| Camino | Error de dirección (mediana) | Tiempo |
|---|---|---|
| Baseline anterior (imagen→imagen) | ~13 px | — |
| Mino, con red para `r`, Φ_r y μ | 9955 px (con bugs) → 32 px | 0.27 s |
| Mino, red para `r` + Φ_r y G exactos | **1.1 px** | 0.27 s |
| Mino, **todo exacto** (μ por Jacobi sn) | **0.001 px** | 0.27 s |

Clasificación de la sombra: **100 % de píxeles idénticos** al trazador clásico
en todos los casos, porque sale en forma cerrada de las raíces del cuártico y
no de ninguna red.

## El hallazgo principal: en Mino, Kerr es analíticamente resoluble

Esta es la conclusión que más cuesta admitir y la más útil.

- `θ(λ)` tiene forma cerrada **exacta**: `cos θ = √u₊ · sn(K(m)(1−4x), m)`.
- `φ` se separa en dos cuadraturas: la polar es una integral elíptica de
  tercera especie (Carlson `R_J`), y la radial se reduce a una cuadratura de
  Gauss-Legendre en una variable bien elegida (ver abajo).
- `r(λ)` es lo único sin forma cerrada elemental, pero se obtiene invirtiendo
  `λ(r)` — que sí es cerrada (Carlson `R_F`) — con 6 pasos de Newton desde una
  semilla cruda, hasta precisión de máquina.

Conclusión honesta: **la red neuronal no hace falta para Kerr**. El camino
exacto es 3500× más preciso y cuesta lo mismo (0.27 s en ambos casos, medido).
El papel legítimo que le queda a la red es la velocidad — sembrar la inversión
de Newton, o alimentar un visor en vivo donde importa el fotograma y no el
último dígito — no la precisión.

Eso NO invalida la red de Schwarzschild (MAE 4.3e-5, 51× más rápida que RK45),
donde el integrador clásico sí es el cuello de botella real.

## La sustitución que hace fácil la cuadratura radial

`quad` adaptativo NO converge en `∫dr/√R` pegado a la curva crítica: hay una
casi-singularidad interior en `r₃ ≈ r₄`. La solución no es más subdivisión sino
cambiar de variable, escribiendo `R = (r−r₁)(r−r₂)·Q(r)` con `Q` el par de
raíces grandes:

| caso | sustitución | resultado |
|---|---|---|
| escapa (`r₃,r₄` reales, ancla en `r₄`) | `r = r₄ + δ sinh²w` | `dr/√Q = 2 dw` |
| cae, par complejo `μ ± iν` | `r = μ + ν sinh w` | `dr/√Q = dw` |
| cae, `r₃,r₄` reales | `r = μ + κ cosh w` | `dr/√Q = dw` |

En los tres casos `dr/√Q` es **constante por `dw`**: la casi-singularidad se
absorbe exactamente, el integrando queda suave y una Gauss-Legendre fija de 96
nodos basta. El `λ` así calculado coincide con el de Carlson a **1e-15** — dos
vías completamente independientes dando precisión de máquina.

## Bugs que costaron tiempo

Todos se encontraron comparando contra algo ya validado, nunca leyendo el
código nuevo buscando el fallo.

1. **`G_phi_exacto` sólo valía en la rama principal.** La fórmula de Π vía las
   funciones de Jacobi pliega `am(v)` dentro de `[−π/2, π/2]`, así que para
   `|x| > 1/2` perdía los cuasi-períodos. Se notaba como `φ` desviado en
   múltiplos de π, y sólo en los rayos que dan varias vueltas. Se arregla
   reduciendo con la periodicidad exacta del integrando (período 1/2 en `x`).

2. **Convención de features distinta entre entrenar e inferir.** El
   entrenamiento construía `2x−1` a mano mientras `predecir()` usaba
   `norm.features()`, que hace su propio `2·(·)−1`. La red veía medio dominio de
   entrada al entrenar y otro al evaluar. No lanza ningún error: sólo da peores
   resultados, que es la peor clase de bug.

3. **`sigma_max` con una resta donde va una suma.** Un rayo que escapa recorre
   la entrada (`lam_cam`) *más* la salida (`lam_esc`). Escribir la resta daba
   2.3 rad de error angular.

4. **Escala de compresión de Φ_r mal puesta.** Se fijó a `max|residuo|` = 572
   en vez de `asinh(max|residuo|)` = 7.04, así que la salida vivía encerrada en
   ±0.012 en lugar de ±1: la MSE apenas tenía gradiente ahí y al descomprimir
   cualquier error de red se multiplicaba por 572.

5. **Pedirle a la red lo que se sabe exacto.** Tres veces seguidas:
   - `r` en el punto de escape vale `r_esc` por construcción — y es justo donde
     peor resuelve la red (`ratio_u` cae a ~5e-4 y el error relativo estalla).
   - `r` en la cámara vale `r_obs`.
   - `G_φ` y `Φ_r` tienen forma cerrada, y entran en `φ` multiplicados por `ξ`
     (~7), así que un error de 1e-3 se convertía en 9 px.

   Cada vez que se sustituyó una salida de red por su forma exacta, el error
   bajó un orden de magnitud. Esa fue la lección estructural de la etapa.

6. **Más capacidad no arregla un target difícil.** La red de `θ` pasó de 8.7k a
   50k parámetros y sólo mejoró de 3.1e-3 a 2.6e-3 — y en el render acabó
   siendo *peor* (3.5 px frente a 1.1 px de la pequeña). Con `m → −∞` la `sn`
   se vuelve casi una onda cuadrada; el problema no era capacidad.

## Limitación abierta: espín extremo cerca del horizonte

Medido con `validate_kerr_mino_net.py --mu-exacto` sobre 12 configuraciones:

| Métrica | Mediana | p99 | Objetivo |
|---|---|---|---|
| Dirección de escape | 2.98e-8 rad | 6.0e-7 | < 2e-4 ✅ |
| `\|dr\|` de los cruces | 2.1e-12 M | 3.6e-9 M | < 1e-3 ✅ |
| `\|dφ\|` de los cruces | 4.1e-13 rad | 8.8e-2 rad | < 1e-3 ⚠️ |
| Speedup | 27–33× | | ≥ 4× ✅ |

La mediana de `|dφ|` está en precisión de máquina, pero **el 2.4 % de los cruces
supera 1e-3 rad**, con un peor caso de 0.18 rad. El patrón es inequívoco:
**todos** los outliers son de `a = 0.998` y en cruces de radio pequeño (1.4–7 M),
donde los dos horizontes casi se fusionan (`r₊ = 1.063`, `r₋ = 0.937`) y `f_r`
tiene un polo casi doble junto al límite inferior de integración.

Lo que ya se descartó como causa:
- **No es la cuadratura**: converge a 96 nodos (de 96 a 2048 nodos la respuesta
  cambia 1e-11).
- **No es la referencia**: el integrador consigo mismo, entre rtol 1e-12 y
  1e-14, es autoconsistente a 1e-12 en esos mismos rayos.
- **No es Newton**: subir de 3 a 6 iteraciones no movió los números.

Queda por investigar. Afecta solo al régimen extremal y no toca la imagen
(la dirección de escape y la sombra son exactas ahí), pero está sin resolver y
no conviene presentarlo como cerrado.

## Umbrales de validación y por qué son los que son

`validate_kerr_mino.py --etapa 0` compara contra el integrador hamiltoniano ya
validado. Dos matices que conviene no re-litigar:

- El oráculo de la cuadratura radial es la **EDO**, no `quad`: `quad` devuelve
  `nan/inf` justo donde más importa (δ pequeño). Se comprobó que Carlson
  coincide con la EDO a 1e-13 en todo el rango.
- Los umbrales de dirección y θ se fijaron en 1e-7 porque a rtol 1e-12 el que
  se movía era el **integrador de referencia**: al apretarlo a 1e-14 la
  predicción de Mino coincidía exactamente. Medir por debajo de eso mide el
  ruido de la referencia, no el error propio.
