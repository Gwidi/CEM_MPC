import torch
from car_env.utils.delay_fifo import DelayBufforFIFO


class TimeDelayFirstOrder(torch.nn.Module):
    def __init__(self, dt, delay_time, tau, num_envs):
        super(TimeDelayFirstOrder, self).__init__()

        self.dt = dt
        self.tau = tau
        self.delay_buffor = DelayBufforFIFO(int(delay_time / dt), num_envs)
        self.x_delayed = torch.tensor([0.0] * num_envs)

    def forward(self, x):
        dxdt = (self.x_delayed - x) / self.tau
        return dxdt

    def update_delayed_input(self, u):
        self.x_delayed = self.delay_buffor(u)

    def copy(self):
        new = TimeDelayFirstOrder(
            self.dt, self.delay_buffor.buffor.shape[-1] * self.dt, self.tau)
        new.delay_buffor = self.delay_buffor.copy()
        new.x_delayed = self.x_delayed.clone()
        return new

    def get_internal_state(self):
        return [*self.delay_buffor.get_internal_state(), self.x_delayed]

    def set_internal_state(self, state):
        self.delay_buffor.set_internal_state(state[:-1])
        self.x_delayed = state[-1].clone()
