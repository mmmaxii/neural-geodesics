# Generacion del Dataset de Entrenamiento

Referencia de formulas para `scripts/generate_data.py`.

## Objetivo

Generar N = 100,000 geodesicas integradas numericamente y guardarlas en
`data/processed/geodesics_dataset.npz` para entrenar la red neuronal.

---

## Estrategia de Muestreo (Smart Sampling)

La funcion delta_phi(b) diverge logaritmicamente en b -> b_crit.
Sin sobremuestreo en esa zona, la red nunca aprende la transicion critica.

### Banda 1: Uniforme (40% = 40,000 puntos)

b_i ~ U(b_min, b_max)

donde b_min = 0.1 * r_s, b_max = 15.0 * r_s.

### Banda 2: Logaritmico (30% = 30,000 puntos)

b_i = 10^(x_i),  x_i ~ U(log10(b_min), log10(b_max))

Cubre mejor los ordenes de magnitud pequenos.

### Banda 3: Concentrado en b_crit (30% = 30,000 puntos)

b_i ~ U(b_crit - epsilon, b_crit + epsilon)

con epsilon = 0.2 * r_s.

### Concatenacion final

b = shuffle( b_uniform || b_log || b_crit_zone )

Mezclar aleatoriamente para evitar sesgo de orden en el entrenamiento.

---

## Extraccion de Features por Rayo

Para cada b_i, llamar a integrator.integrate_ray(b=b_i) y extraer:

- delta_phi_i = res['delta_phi']       (angulo de deflexion total)
- r_min_i     = min(res['r'])           (radio de maximo acercamiento)
- captured_i  = 1 si res['status'] == 'captured', 0 si no

---

## Formato de Salida (.npz)

| Llave      | Shape  | Descripcion                        |
|------------|--------|------------------------------------|
| b_values   | (N,)   | Parametros de impacto              |
| delta_phi  | (N,)   | Deflexion total                    |
| r_min      | (N,)   | Radio de maximo acercamiento       |
| captured   | (N,)   | Booleano: capturado o no           |
| b_crit     | scalar | Valor de b_crit para referencia    |

Guardar con np.savez_compressed().

---

## Notas

- El script deberia ser ~40 lineas.
- Archivo de salida: data/processed/geodesics_dataset.npz
- Tiempo estimado: depende de la maquina (~minutos con N=100k y RK45).
