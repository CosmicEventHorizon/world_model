import numpy as np
from sampling.ActionsDistribution import ActionsDistribution


class CarActions:
    def generate(no_actions: int) -> tuple[np.ndarray, int]:
        modes = [0, 1, 2, 3]
        no_actions_per_mode = no_actions // 4
        total_no_actions = no_actions_per_mode * 4
        a = np.zeros(shape=(total_no_actions, 3))
        ad = ActionsDistribution()
        i = 0
        for mode in modes:
            ad.setMode(mode)
            for _ in range(no_actions_per_mode):
                a[i] = CarActions.__sample(ad)
                i += 1

        return (a, total_no_actions)

    def __sample(ad: ActionsDistribution) -> np.ndarray:
        steer = CarActions.__steer(ad.steer())
        gas = CarActions.__pedal(ad.gas())
        brake = CarActions.__pedal(ad.brake())
        return np.array([steer, gas, brake])

    def __steer(param: tuple) -> float:
        sample = np.random.normal(param[0], param[1])
        if sample < -1:
            sample = -1
        if sample > 1:
            sample = 1
        return sample

    def __pedal(param: tuple) -> float:
        sample = np.random.normal(param[0], param[1])
        if sample < 0:
            sample = 0
        if sample > 1:
            sample = 1
        return sample
