from stable_baselines3 import PPO
import logging
import matplotlib.pyplot as plt
import numpy as np
import pygame
import csv
import pandas as pd



class DataCollector:
    def __init__(self, env):
        self.dt = env.dt
        self.sates = []
        self.state_names = env.simulator.state_names
        self.obs = []
        self.obs_names = env.simulator.obs_config.names
        self.actions = []
        self.action_names = env.simulator.control_names
        self.t = []

        self.ferent = []
        self.ferent_names = ["s", "closest_dist", "heading_diff"]

        self.cartisian = []
        self.cartisian_names = ["x", "y", "yaw"]

    def collect(self, state, obs, action):
        self.sates.append(state)
        self.obs.append(obs)
        self.actions.append(action)

        if self.t == []:
            self.t.append(0)
        else:
            self.t.append(self.t[-1] + self.dt)

    def plot_data(self, data_type, do_not_plot=[]):
        if data_type == "obs":
            data = self.obs
            names = self.obs_names
            title = "Observations"
        elif data_type == "state":
            data = self.sates
            names = self.state_names
            title = "States"
        elif data_type == "actions":
            data = self.actions
            names = self.action_names
            title = "Actions"
        else:
            raise ValueError(
                "Invalid data type. Choose from 'obs', 'state', or 'actions'."
            )

        suplots_n = len(names) - sum(1 for i in names if i in do_not_plot)
        fig, axs = plt.subplots(suplots_n, 1, figsize=(10, 10))
        fig.suptitle(title)

        ax_idx = 0

        for i, name in enumerate(names):
            if name in do_not_plot:
                continue
            axs[ax_idx].plot(self.t, [d[i] for d in data], label=f"{name}")
            axs[ax_idx].legend()
            axs[ax_idx].grid()
            ax_idx += 1

    def save_data(self, file_name):
        # crete pd dataframe
        data = {
            "t": self.t,
        }

        for i, name in enumerate(self.ferent_names):
            data[name] = [d[i] for d in self.ferent]

        for i, name in enumerate(self.state_names):
            data[name] = [d[i] for d in self.sates]

        # for i, name in enumerate(self.obs_names):
        #     data["obs_" + name] = [d[i] for d in self.obs]

        df = pd.DataFrame(data)

        # save to csv
        path = "./data/tests_sim/" + file_name
        df.to_csv(path, index=False)


class LapTimer:
    def __init__(self, dt):
        self.dt = dt
        self.lap_times = []
        self.iter_since_last_lap = 0
        self.last_s = None

    def update(self, s):
        if self.last_s is None:
            self.last_s = s
        elif (self.last_s - s) > 1.0:  # 1.0 is arbitraty
            lap_time = self.iter_since_last_lap * self.dt
            self.lap_times.append(lap_time)
            print(f"Lap time: {lap_time} s")
            self.iter_since_last_lap = 0

        self.last_s = s
        self.iter_since_last_lap += 1

    def reset_lap(self):
        self.iter_since_last_lap = 0
        self.last_s = None

    def get_lap_times(self):
        return self.lap_times


class Tester:
    def __init__(
        self,
        env,
        drive_manual=False,
        n_steps=512,
        control_delay=0,
        control_noise=0,
        obs_noise=0,
        render=True,
        friction_spline_y=None,
        friction_spline_x=None,
    ):
        self.env = env

        self.drive_manual = drive_manual
        self.n_steps = n_steps
        self.control_noise = control_noise
        self.obs_noise = obs_noise
        self.render = render

        self.data_collector = DataCollector(env)

        self.lap_timer = LapTimer(env.dt)

        self.control_array = np.zeros((control_delay + 1, 2))

        self.friction_spline_y = friction_spline_y
        self.friction_spline_x = friction_spline_x


    def run(self, model, reccurent=False, start_s=None):

        if self.render:
            self.env.start_render()

        obs = self.env.reset()

        if self.friction_spline_y is not None and self.friction_spline_x is not None:
            self.env.set_friction_curve(
            0,
            self.friction_spline_x,
            self.friction_spline_y,
            )

        if start_s is not None:
            self.env.simulator._initialize_state(start_at_zero=True)
            s = self.env.simulator.get_s()

        _states = None

        for _ in range(self.n_steps):
            if self.drive_manual:
                action = self.get_keyboard_control()
            else:
                if not reccurent:
                    action, _states = model.predict(obs, deterministic=True)
                else:
                    action, _states = model.predict(
                        obs, deterministic=True, state=_states
                    )

            action = self.add_control_noise(action)

            # add control delay
            self.control_array = np.roll(self.control_array, 1, axis=0)
            self.control_array[0] = action[0]
            action[0] = self.control_array[-1]

            obs, rewards, dones, info = self.env.step(action)

            if dones[0]:
                self.lap_timer.reset_lap()
                self.env.simulator._initialize_state(start_at_zero=True)
                s = self.env.simulator.get_s()

            obs = self.add_obs_noise(obs)
            s = self.env.simulator.get_s()

            self.data_collector.collect(
                state=self.env.state[0], obs=obs[0], action=action[0]
            )
            self.data_collector.ferent.append(
                [
                    s,
                    self.env.simulator.get_clostest_dist(),
                    self.env.simulator.get_heading_diff(),
                ]
            )
            self.data_collector.cartisian.append(
                [
                    self.env.simulator.get_x(),
                    self.env.simulator.get_y(),
                    self.env.simulator.get_yaw(),
                ]
            )

            self.lap_timer.update(s)

            if self.render:
                self.env.render()

        lap_times = self.lap_timer.get_lap_times()
        self.lap_timer.reset_lap()
        return lap_times

    def get_keyboard_control(self):
        action = np.array([[0.0, 0.0]], dtype=np.float32)
        events = pygame.event.get()
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_UP]:
            action[0][1] = 1.0
        if pressed[pygame.K_DOWN]:
            action[0][1] = -1.0
        if pressed[pygame.K_LEFT]:
            action[0][0] = 1.0
        if pressed[pygame.K_RIGHT]:
            action[0][0] = -1.0
        return action

    def add_obs_noise(self, obs):
        obs = obs * np.random.normal(1, self.obs_noise, obs.shape)
        return obs

    def add_control_noise(self, action):
        action = action * np.random.normal(1, self.control_noise, action.shape)
        return action.astype(np.float32)

    def plot_data(self, actions=False, states=False, obs=False):
        if actions:
            self.data_collector.plot_data("actions")
        if states:
            self.data_collector.plot_data("state", do_not_plot=["x", "y", "friction"])
        if obs:
            self.data_collector.plot_data("obs", do_not_plot=["curvatures", "widths"])

        plt.show()


