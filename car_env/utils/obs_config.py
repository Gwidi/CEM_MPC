import os
import numpy as np
import yaml
import wandb
import gymnasium as gym


class ObservationInfo:
    def __init__(
        self,
        name: str,
        dim: int,
        min_val: float,
        max_val: float,
        rand: float,
        length: int = None,
        sample_rate: int = None,
    ):
        self.name = name
        self.dim = dim
        self.min_val = min_val
        self.max_val = max_val
        self.rand = rand

        if length is None and sample_rate is None:
            self.iterable = False
        else:
            self.iterable = True
            self.length = length
            self.sample_rate = sample_rate


class ObservationConfig:
    def __init__(self):
        self.observation_list = []
        self.obs_max = np.array([])
        self.obs_min = np.array([])
        self.obs_dim = 0
        self.names = []

    def add_observation(self, obs: ObservationInfo):
        self.observation_list.append(obs)
        self.obs_dim += obs.dim
        self.obs_max = np.append(self.obs_max, np.array([obs.max_val] * obs.dim))
        self.obs_min = np.append(self.obs_min, np.array([obs.min_val] * obs.dim))

        for i in range(obs.dim):
            self.names.append(obs.name)
        # create obs pos wioch is dict with name and pos

    def get_observation_space(self):
        high = self.obs_max
        low = self.obs_min
        observation_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=(self.obs_dim,),
            dtype=np.float64,
        )
        return observation_space

    def get_info(self):
        info = []
        for oba in self.observation_list:
            if oba.iterable:
                info.append(
                    {
                        "name": oba.name,
                        "dim": oba.dim,
                        "length": oba.length,
                        "sample_rate": oba.sample_rate,
                        "min_val": oba.min_val,
                        "max_val": oba.max_val,
                    }
                )
            else:
                info.append(
                    {
                        "name": oba.name,
                        "dim": oba.dim,
                        "min_val": oba.min_val,
                        "max_val": oba.max_val,
                    }
                )
        return info

    def load_from_file(self, name):
        # load from yaml
        path = os.path.join(os.path.dirname(__file__), "../config/obs/" + name + ".yaml")

        with open(path, "r") as file:
            data = yaml.load(file, Loader=yaml.FullLoader)

        for ob in data:
            # check if iterable
            rand = None
            if "rand" in ob:
                rand = ob["rand"]

            if "length" in ob:
                self.add_observation(
                    ObservationInfo(
                        name=ob["name"],
                        dim=ob["dim"],
                        min_val=ob["min_val"],
                        max_val=ob["max_val"],
                        length=ob["length"],
                        sample_rate=ob["sample_rate"],
                        rand=rand,
                    )
                )
            else:
                self.add_observation(
                    ObservationInfo(
                        name=ob["name"],
                        dim=ob["dim"],
                        min_val=ob["min_val"],
                        max_val=ob["max_val"],
                        rand=rand,
                    )
                )
