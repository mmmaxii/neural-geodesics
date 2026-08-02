# Investigación Inicial — Neural Geodesics

## 1. Fundamento Físico

### 1.1 La Métrica de Schwarzschild
La métrica de Schwarzschild describe el campo gravitacional en el exterior de una masa esféricamente simétrica, estática y sin carga (un agujero negro no rotatorio). Derivada por Karl Schwarzschild en 1916, es la primera solución exacta de las ecuaciones de campo de Einstein.

$$ ds^2 = -\left(1 - \frac{r_s}{r}\right) c^2 dt^2 + \left(1 - \frac{r_s}{r}\right)^{-1} dr^2 + r^2 \left(d\theta^2 + \sin^2\theta d\phi^2\right) $$

Donde:
- $r_s = \frac{2GM}{c^2}$ es el radio de Schwarzschild.
- $G$ es la constante de gravitación universal.
- $M$ es la masa del objeto.
- $c$ es la velocidad de la luz.

Convención de unidades naturales: Frecuentemente se asume $G = c = 1$, por lo que $r_s = 2M$.

### 1.2 Geodésicas Nulas (Trayectorias de Fotones)
Las partículas sin masa, como los fotones, siguen trayectorias en el espaciotiempo conocidas como geodésicas nulas ($ds^2 = 0$). La ecuación geodésica general está dada por:

$$ \frac{d^2x^\mu}{d\lambda^2} + \Gamma^\mu_{\alpha\beta} \frac{dx^\alpha}{d\lambda} \frac{dx^\beta}{d\lambda} = 0 $$

Debido a las simetrías de la métrica de Schwarzschild (independencia del tiempo $t$ y del ángulo acimutal $\phi$), existen vectores de Killing asociados que implican la conservación de dos cantidades a lo largo de la trayectoria del fotón:
- **Energía $E$**
- **Momento Angular $L$**

### 1.3 El Potencial Efectivo
La dinámica radial del fotón puede formularse en términos de un potencial efectivo. Partiendo de $ds^2 = 0$ y usando las constantes de movimiento, obtenemos:

$$ \left(\frac{dr}{d\lambda}\right)^2 + V_{\text{eff}}(r) = E^2 $$

Para fotones, el potencial efectivo es:

$$ V_{\text{eff}}(r) = \frac{L^2}{r^2} \left(1 - \frac{r_s}{r}\right) $$

Interpretación física: Este potencial tiene un máximo que corresponde a una órbita circular inestable para los fotones. Esta órbita se conoce como la **esfera de fotones**.

### 1.4 La Ecuación de Binet
Para realizar ray tracing, es más útil conocer la forma geométrica de la órbita $r(\phi)$ en lugar de su evolución temporal $r(t)$ o parametrizada $r(\lambda)$. Realizando el cambio de variable $u = \frac{1}{r}$, derivamos la ecuación de Binet para fotones en Schwarzschild:

$$ \frac{d^2u}{d\phi^2} + u = \frac{3GM}{c^2}u^2 $$

Esta es la ecuación fundamental que debe integrarse numéricamente en el ray tracing clásico.
Se puede reescribir como un sistema de primer orden para integración (ej. RK45):
$$ \frac{du}{d\phi} = p $$
$$ \frac{dp}{d\phi} = -u + 3Mu^2 $$

### 1.5 Parámetro de Impacto Crítico
El parámetro de impacto de un fotón se define como $b = \frac{L}{E}$. Representa la distancia perpendicular desde el centro del agujero negro a la trayectoria asintótica del fotón entrante.

El fotón será capturado por el agujero negro si su parámetro de impacto es menor que un valor crítico $b_{\text{crit}}$. En este punto, la energía del fotón es igual al máximo del potencial efectivo (situado en $r = 1.5 r_s = 3M$).
Resolviendo para este máximo, el parámetro de impacto crítico es:

$$ b_{\text{crit}} = \frac{3\sqrt{3}}{2} r_s \approx 2.598 r_s $$

La sombra del agujero negro que un observador lejano percibe tiene un radio angular determinado por este $b_{\text{crit}}$.

### 1.6 Clasificación de Trayectorias
Dependiendo de $b$, tenemos tres escenarios:
- **$b < b_{\text{crit}}$**: El fotón entra en la esfera de fotones y es inexorablemente **capturado** (cruza el horizonte de eventos).
- **$b = b_{\text{crit}}$**: El fotón entra en una órbita inestable en la esfera de fotones ($r = 3M$) y orbita infinitamente (en teoría).
- **$b > b_{\text{crit}}$**: El fotón es **desviado** por la gravedad pero logra escapar hacia el infinito.

En el límite de campo débil ($b \gg b_{\text{crit}}$), el ángulo de deflexión $\alpha$ se aproxima a:
$$ \alpha \approx \frac{4GM}{c^2b} $$

### 1.7 Extensión a Kerr (Futuro)
La métrica de Kerr describe un agujero negro en rotación. Es considerablemente más compleja ya que introduce momento angular intrínseco (spin $a$).
Además de la Energía y el Momento Angular, aparece una tercera constante de movimiento: la **constante de Carter**, debido a simetrías ocultas asociadas a tensores de Killing-Yano.
El enfoque neuronal desarrollado para Schwarzschild se escala a Kerr aprendiendo un mapeo más multidimensional, esquivando la extremadamente costosa integración acoplada.

## 2. Estado del Arte — Machine Learning para Geodésicas

### 2.1 PINNs (Physics-Informed Neural Networks)
Introducidas ampliamente por Raissi et al. (2019), las PINNs incorporan las ecuaciones diferenciales (PDEs/ODEs) directamente en la función de pérdida.
- **Aplicación a geodésicas:** Penalizan a la red si sus predicciones no satisfacen la Ecuación de Binet.
- **Pros y contras:** Excelente generalización y respeto a la física, pero a veces difíciles de entrenar por competencia entre el término de datos y el término físico.

### 2.2 Neural ODEs
Propuestas por Chen et al. (2018, arXiv:1806.07366) y popularizadas a través de `torchdiffeq`. Tratan las capas de la red como un proceso continuo e integran usando un ODE solver.
- **Limitación en este proyecto:** Para ray tracing en tiempo real, queremos *evitar* la integración explícita durante la inferencia para ahorrar cómputo, por lo que este enfoque no es el ideal para la pasada final.

### 2.3 DeepONet
Lu et al. (2021, Nature Machine Intelligence) propuso el aprendizaje directo de operadores no lineales usando redes "branch" (codifican la condición inicial) y "trunk" (codifican la variable independiente).
- Representa una evolución natural para mapear $b \to \text{trayectoria completada}$ sin paso iterativo.

### 2.4 GravLensX (ICCV 2025)
- **Referencia:** arXiv:2507.15775
- El paper más relevante a la fecha para la aplicación de este proyecto. Demuestra aceleraciones de más de $15\times$ respecto a métodos clásicos. Sustituye la integración de geodésicas (Binet y equivalentes en Kerr) por una red neuronal optimizada que fue validada con métricas de agujeros negros rotatorios (Kerr).

### 2.5 Otros Enfoques
- **BH-NeRF:** Neural Radiance Fields adaptados al espaciotiempo curvo para renderizar discos de acreción alrededor de agujeros negros.
- Modelos de difusión latente (Latent Diffusion) para generar imágenes (arXiv:2602.07786).
- **MANet:** Regresión directa de parámetros astrofísicos (arXiv:2507.15910).

## 3. Arquitectura Elegida: PIMLP

### 3.1 Justificación
Se optó por un **Physics-Informed Multi-Layer Perceptron (PIMLP)** por ser el balance óptimo:
| Método | Precisión | Velocidad Inferencia | Costo Entrenamiento | Respeto a Física |
|--------|-----------|----------------------|---------------------|------------------|
| ODE (Clásico) | Alta | Lenta (paso a paso) | N/A | Total |
| MLP Puro | Media | Muy Rápida | Bajo | Bajo (puede divergir) |
| PINN/PIMLP | Alta | Muy Rápida | Medio-Alto | Alto |
| Neural ODE | Alta | Lenta | Alto | Total |

El PIMLP es rápido durante la inferencia (solo multiplicaciones matriciales) pero se beneficia de la regularización física durante el entrenamiento.

### 3.2 Función de Pérdida Híbrida
El entrenamiento de PIMLP utiliza una función de pérdida compuesta:
$$ \mathcal{L} = \lambda_{\text{data}} \mathcal{L}_{\text{MSE}} + \lambda_{\text{phys}} \mathcal{L}_{\text{ODE}} + \lambda_{\text{cls}} \mathcal{L}_{\text{BCE}} $$

- **Data loss (MSE):** Error cuadrático medio respecto a las trayectorias generadas por RK45.
- **Physics loss (ODE residual):** Cuánto viola la salida de la red la ecuación de Binet ($d^2u/d\phi^2 + u - 3Mu^2 = 0$).
- **Classification loss (BCE):** Pérdida de entropía cruzada binaria para predecir si el fotón será capturado o escapará (crucial cerca de $b_{\text{crit}}$).
- **Lambda scheduling:** Las ponderaciones $\lambda$ se ajustan dinámicamente durante el entrenamiento (p. ej. empezando con MSE y aumentando ODE gradualmente).

### 3.3 Datos de Entrenamiento
Se emplea integración RK45 para generar el "ground truth".
- **Estrategia de Muestreo:** En lugar de un muestreo uniforme de parámetros de impacto $b$, se sobre-muestrea densamente alrededor del umbral $b_{\text{crit}}$ donde la dinámica es caótica e inestable.
- Normalización de las entradas y salidas para evitar gradientes explosivos en la red.

## 4. Ray Tracing Clásico

### 4.1 Códigos Existentes
Existen múltiples soluciones numéricas clásicas pesadas utilizadas en la academia, por ejemplo:
- **GYOTO, RAPTOR, IPOLE, AART, BlackRay**
Suelen basarse en C++ o Fortran y son capaces de trazar radiación polarizada a través de plasmas densos, pero son demasiado lentos para simulaciones interactivas ligeras.

### 4.2 Implementaciones Web
Ejemplos interactivos modernos que usan WebGL o integradores simplificados:
- oseiskar/black-hole
- MisterPrada/singularity
- Adriwin06/black-hole
- Chaotic Curiosity

## 5. Stack Tecnológico

### 5.1 Backend
- **PyTorch** y **torchdiffeq**: Entrenamiento del modelo PIMLP y cálculo de gradientes.
- **scipy** (`scipy.integrate.solve_ivp` RK45): Generación precisa de ground truth numérico.

### 5.2 Export para Web
- **ONNX Runtime Web:** Para correr el PIMLP directamente en el navegador del cliente sin depender de un servidor de inferencia backend.
- Aceleración mediante **WebGPU**.
- Cuantización de pesos a **INT8** para minimizar el peso del modelo transmitido a la web.

### 5.3 Frontend (Fase Futura)
- **React** + **react-three-fiber** para la escena y renderizado 3D.
- **GLSL Shaders** acoplados con los tensores de ONNX para ray tracing final.
- **Leva** para el panel de controles UI.

## 6. Referencias Bibliográficas

- **Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).** *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics. URL: [https://doi.org/10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045) — Fundacional para PINNs.
- **Chen, R. T., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K. (2018).** *Neural ordinary differential equations.* arXiv:1806.07366. URL: [https://arxiv.org/abs/1806.07366](https://arxiv.org/abs/1806.07366) — Trabajo clave de modelos continuos en ML.
- **Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021).** *Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators.* Nature Machine Intelligence. URL: [https://doi.org/10.1038/s42256-021-00302-5](https://doi.org/10.1038/s42256-021-00302-5) — Deep operator networks.
- **GravLensX (2025).** *Neural rendering of gravitational lensing.* arXiv:2507.15775. URL: [https://arxiv.org/abs/2507.15775](https://arxiv.org/abs/2507.15775) — Referencia principal para aceleración de renderizado de agujeros negros usando ML.
- **MANet (2025).** arXiv:2507.15910. URL: [https://arxiv.org/abs/2507.15910](https://arxiv.org/abs/2507.15910) — Regresión de parámetros.
- **Latent Diffusion for BH (2026).** arXiv:2602.07786. URL: [https://arxiv.org/abs/2602.07786](https://arxiv.org/abs/2602.07786) — Últimas aplicaciones generativas a astrofísica.
