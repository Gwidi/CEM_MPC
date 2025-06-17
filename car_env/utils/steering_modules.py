import torch
from car_env.utils.delay_fifo import DelayBufforFIFO
from car_env.utils.time_delay_first_order import TimeDelayFirstOrder as tdfo


class SteeringWheelModule:
    def __init__(self, dt, num_envs, delay=0.00, tau=0.04) -> None:
        self.dt = dt  # s
        self.delay = delay # s
        self.tau = tau  # s
        self.tdf = tdfo(self.dt, self.delay, self.tau, num_envs)
        self.y = torch.zeros(num_envs)
        self.zero_counter = 0

    def _rk4_step(self, f, xu, dt):
        k1 = f(xu)
        k2 = f(xu + dt / 2 * k1)
        k3 = f(xu + dt / 2 * k2)
        k4 = f(xu + dt * k3)
        return dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4) + xu

    def _euler_step(self, f, xu, dt):
        return xu + dt * f(xu)

    def update(self, delta_ctrl):
        with torch.inference_mode():
            self.y = self._rk4_step(self.tdf, self.y, self.dt)

        self.tdf.update_delayed_input(delta_ctrl)

        return self.y


class CurrentControlModule:
    def __init__(self, dt, num_envs, delay=0.05) -> None:
        self.dt = dt  # s
        self.delay = delay # s
        self.fifo = DelayBufforFIFO(int(self.delay / self.dt), num_envs)

    def update(self, current_ctrl):
        delay_current_ctrl = self.fifo(current_ctrl)

