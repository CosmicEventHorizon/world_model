'''
[-1,+1]: steer left, steer right
[0,1]: gas
[0,1]: brake
'''
class ActionsDistribution:
    def steer() -> tuple:
        steer_mean = 0
        steer_sd = 0
        return (steer_mean,steer_sd)

    def gas() -> tuple:
        gas_mean = 0.2
        gas_sd = 0.1
        return (gas_mean,gas_sd)

    def brake() -> tuple:
        brake_mean = 0
        brake_sd = 0
        return (brake_mean,brake_sd)
