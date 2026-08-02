"""
Actions
[-1,+1]: steer left, steer right
[0,1]: gas
[0,1]: brake
"""

"""
Mode
0: Full gas
1: gas + brake
2: gas + left + right
3: random
"""


class ActionsDistribution:
    def steer(self) -> tuple:
        return (self.steer_mean, self.steer_sd)

    def gas(self) -> tuple:
        return (self.gas_mean, self.gas_sd)

    def brake(self) -> tuple:
        return (self.brake_mean, self.brake_sd)

    def setMode(self, mode: int):
        self.mode = mode
        self.__setupMode()

    def __setupMode(self):
        self.__resetVariables()
        match self.mode:
            case 0:
                self.__gas()
            case 1:
                self.__gas()
                self.__brake()
            case 2:
                self.__gas()
                self.__steer()
            case 3:
                self.__gas()
                self.__steer()
                self.__brake()

    def __resetVariables(self):
        self.gas_mean = 0
        self.gas_sd = 0
        self.steer_mean = 0
        self.steer_sd = 0
        self.brake_mean = 0
        self.brake_sd = 0

    def __gas(self):
        self.gas_mean = 0.2
        self.gas_sd = 0.1

    def __steer(self):
        self.steer_mean = 0
        self.steer_sd = 1

    def __brake(self):
        self.brake_mean = 0.01
        self.brake_sd = 0.1
