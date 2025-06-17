import torch


class DelayBufforFIFO(torch.nn.Module):
    def __init__(self, delay_time_in_samples, num_envs):
        super(DelayBufforFIFO, self).__init__()
        self.delay_time_in_samples = delay_time_in_samples
        self.buffor = torch.zeros(num_envs, delay_time_in_samples)
        self.i = torch.zeros(num_envs, dtype=torch.int64)

    def forward(self, x: torch.tensor):
        ans = self.buffor.gather(1, self.i.unsqueeze(1)).squeeze(1).clone()
        self.buffor[:, self.i] = x.clone()
        self.i = (self.i + 1) % (self.delay_time_in_samples)

        return ans

    # def copy(self):
    #     new = DelayBufforFIFO(len(self.buffor))
    #     new.buffor = self.buffor.clone()
    #     new.i = self.i
    #     return new

    # def get_internal_state(self):
    #     return self.buffor.clone(), self.i.clone()

    # def set_internal_state(self, state):
    #     buffor, i = state
    #     assert buffor.shape == self.buffor.shape, "Buffor shape mismatch"
    #     self.buffor = buffor.clone()
    #     self.i = i.clone()
