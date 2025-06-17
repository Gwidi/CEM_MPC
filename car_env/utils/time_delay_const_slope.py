import torch
from car_env.utils.delay_fifo import DelayBufforFIFO


class TimeDelayConstSlope(torch.nn.Module):
    def __init__(self, dt, delay_time, slope, num_envs):
        super(TimeDelayConstSlope, self).__init__()

        self.dt = dt
        self.slope = slope
        self.delay_buffor = DelayBufforFIFO(int(delay_time / dt), num_envs)
        self.x_delayed = torch.zeros(num_envs)

    def forward(self, x):
        return torch.tanh(25 * (self.x_delayed - x)) * self.slope

    def update_delayed_input(self, u):
        self.x_delayed = self.delay_buffor(u)

    def copy(self):
        new = TimeDelayConstSlope(
            self.dt, self.delay_buffor.buffor.shape[-1] * self.dt, self.slope
        )
        new.delay_buffor = self.delay_buffor.copy()
        new.x_delayed = self.x_delayed.clone()
        return new

    def get_internal_state(self):
        return [*self.delay_buffor.get_internal_state(), self.x_delayed]

    def set_internal_state(self, state):
        self.delay_buffor.set_internal_state(state[:-1])
        self.x_delayed = state[-1].clone()
