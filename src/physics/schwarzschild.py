# Formulas de la metrica de schwarzschild
import utils.constants as c
import numpy as np



class SMetric:
    def __init__(M: float = c.DEFAULT_M):
        self.M = M
        self.r_s = c.RS_FACTOR * self.M
        
        self.r_horizon = self.r_s
        self.r_photon = 1.5 * self.r_s
        self.r_isco = 3 * self.r_s
        self.b_crit = (3.0 * np.sqrt(3.0) / 2.0) * self.r_s
    
    def f_aux(self, r: float) -> float:
        return 1 - (self.r_s / r)

    def g_tt(self, r: float) -> float:
        return -self.f_aux(r)

    def g_rr(self, r: float) -> float:
        return 1 / self.f_aux(r)
    
    def g_thetatheta(self, r: float) -> float:
        return r**2

    def g_phiphi(self, r: float, theta: float) -> float:
        return r**2 * np.sin(theta)

    # reunimos todas las componentes y las retornamos siempre en una tupla (t, r, thetha, phi) 
    def smetric_components(self, r: float, thetha: float) -> tuple:
        return (self.g_tt(r), self.g_rr(r), self.g_thetatheta(r), self.g_phiphi(r,thetha))


    

  