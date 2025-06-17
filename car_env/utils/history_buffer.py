import torch


class HistoryBuffer(torch.nn.Module):
    def __init__(self, history_length, num_envs):
        super(HistoryBuffer, self).__init__()
        self.history_length = history_length
        self.num_envs = num_envs
        self.buffer = torch.zeros(num_envs, history_length)
        self.i = torch.zeros(num_envs, dtype=torch.int64)
        self.last_input = torch.zeros(num_envs, 1)

    def add(self, x: torch.Tensor):
        # Remove the last dimension if x has shape (num_envs, 1)
        x = x.squeeze(-1)
        indices = torch.arange(self.num_envs)
        # Correctly index the buffer using advanced indexing
        self.buffer[indices, self.i] = x
        self.last_input = x
        self.i = (self.i + 1) % self.history_length

    def get_history(self):
        # Double the buffer to handle wrap-around
        double_buffer = torch.cat([self.buffer, self.buffer], dim=1)
        indices = self.i.unsqueeze(1) + torch.arange(self.history_length)
        env_indices = torch.arange(self.num_envs).unsqueeze(1)
        # Retrieve the history using advanced indexing
        history = double_buffer[env_indices, indices]
        # reverse the history to get the correct order
        history = history.flip(1)

        return history.clone()
    
    def reset(self, env_reset):
        # zero out the history buffer for the specified environments
        self.buffer = torch.where(env_reset.unsqueeze(1), torch.zeros_like(self.buffer), self.buffer)
        self.last_input = torch.where(env_reset, torch.zeros_like(self.last_input), self.last_input)

    def get_last_input(self):
        return self.last_input.clone()


if __name__ == "__main__":
    # Test the history buffer
    history_length = 3
    num_envs = 2

    history_buffer = HistoryBuffer(history_length, num_envs)

    for i in range(20):
        x = torch.tensor([[i], [i + 1]], dtype=torch.float32)
        history_buffer.add(x)
        if i % 5 == 0:
            env_reset = torch.tensor([0, 1], dtype=torch.bool)
            history_buffer.reset(env_reset)
        print(f"Step {i+1}:")
        print(history_buffer.get_history())
        print(history_buffer.get_last_input())
