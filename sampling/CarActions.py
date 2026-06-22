import numpy as np
from sampling.ActionsDistribution import ActionsDistribution as ad

class CarActions:
    def generate() -> np.ndarray:
        steer = CarActions.__steer(ad.steer())
        gas = CarActions.__pedal(ad.gas())
        brake = CarActions.__pedal(ad.brake())
        return np.array([steer, gas, brake])
        
    def __steer(param: tuple) -> float:
        sample = np.random.normal(param[0],param[1])
        if sample < -1:
            sample = -1
        if sample > 1:
            sample = 1
        return sample 

    def __pedal(param: tuple) -> float:
        sample = np.random.normal(param[0],param[1])
        if sample < 0:
            sample = 0
        if sample > 1:
            sample = 1
        return sample