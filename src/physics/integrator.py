# Definimos el integrador de manera iterativa
from scipy.integrate import solve_ivp
from physics.schwarzschild import SMetric
import numpy as np

def binet_rhs(phi: float, state: list, M: float) -> list:
    u, p = state
    du_dphi = p 
    dp_dphi = -u + 3.0 * M * (u**2)
    return [du_dphi, dp_dphi]


# Condiciones para el solver
def event_capture(phi: float, state: list, M: float, r_s: float) -> float:
    u, p = state
    return u - (1.0 / r_s)

event_capture.terminal = True

def event_escape(phi: float, state: list, u_min: float) -> float:
    u, p = state
    return u - u_min

event_escape.terminal = True


class GeodesicIntegrator:
    def __init__(self, metric: SMetric):
        self.metric = metric

    def integrate_ray(
        self,
        b: float,
        r0: float = 100.0,
        max_revolutions: float = 8.0 * np.pi,
        rtol: float = 1e-8,
        atol: float = 1e-10,
        dense_output: bool = False,
    ) -> dict:
        """Integra un rayo con parámetro de impacto b.

        Con dense_output=True el diccionario incluye 'sol_dense', un interpolador
        de orden alto sol_dense(phi) -> (u, p) que permite muestrear la trayectoria
        en cualquier malla de phi sin volver a integrar. Es lo que usa
        scripts/generate_data.py para construir el dataset de trayectorias.
        """
        # Primero definimos las condiciones iniciales (para fotón entrante, u aumenta -> p0 > 0)
        u0 = 1.0 / r0
        p0 = np.sqrt(np.max([0.0, (1.0 / b**2) - u0**2 * (1.0 - self.metric.r_s * u0)]))
        state_0 = [u0, p0]

        # Definimos la distancia máxima del escape (r > 1.05 * r0)
        u_escape = 1.0 / (1.05 * r0)

        cap_event = lambda phi, state, *args: event_capture(phi, state, self.metric.M, self.metric.r_s)
        cap_event.terminal = True

        esc_event = lambda phi, state, *args: event_escape(phi, state, u_escape)
        esc_event.terminal = True
        esc_event.direction = -1
        
        sol = solve_ivp(
            fun=binet_rhs, 
            t_span=(0.0, max_revolutions),
            y0=state_0,
            args=(self.metric.M,),
            events=[cap_event, esc_event],
            rtol=rtol,
            atol=atol,
            dense_output=dense_output,
        )

        r_array = 1.0 / sol.y[0]

        # Clasificación del resultado
        if len(sol.t_events[0]) > 0:
            status = 'captured'
            delta_phi = 0.0
        elif len(sol.t_events[1]) > 0:
            status = 'escaped'
            # sol.t[-1] solo cubre el tramo r0 -> r_min -> r_end, no el viaje
            # completo desde/hasta el infinito. Fuera de r0 el rayo es casi una
            # recta de parámetro de impacto b, con r(phi) = b / cos(phi), así que
            # cada tramo que falta barre pi/2 - arccos(b/r) = arcsin(b/r).
            # Sin esto delta_phi depende de r0 y llega a salir negativa.
            r_end = r_array[-1]
            tail = (np.arcsin(np.clip(b / r0, -1.0, 1.0))
                    + np.arcsin(np.clip(b / r_end, -1.0, 1.0)))
            delta_phi = sol.t[-1] - np.pi + tail
        else:
            status = 'orbit'
            delta_phi = 0.0

        out = {
            'phi': sol.t,
            'u': sol.y[0],
            'r': r_array,
            'delta_phi': delta_phi,
            'status': status
        }
        if dense_output:
            out['sol_dense'] = sol.sol
        return out
