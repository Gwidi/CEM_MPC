import os
import torch
import yaml
import logging
import numpy as np
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter1d


from car_env.utils.track_reader import TrackReader
from car_env.utils.utils import Info
from car_env.utils.history_buffer import HistoryBuffer
from car_env.utils.state_wrapper import StateWrapper, STATE_DEF_LIST
from car_env.utils.obs_config import ObservationConfig
from car_env.utils.steering_modules import SteeringWheelModule
from car_env.envs.simulators.robot_models.single_track_params import VehicleParameters
from car_env.envs.simulators.robot_models.single_track import SingleTrackPacejkaModel, SingleTrackPacejkaModelNoDelay, SingleTrackPacejkaModelVelocityControls

import matplotlib.pyplot as plt
 
class Simulator(torch.nn.Module):
    @torch.no_grad()
    def __init__(
        self,
        vehicle_config: str,
        tire_config: str,
        obs_config: ObservationConfig,
        dt: float,
        integration_method: str,
        num_envs: int,
        tracks: list,
        device: torch.device,
        constants,
        reward_type: str = "rl",
        rand_config: str = None,
        normalize_observations: bool = False,
        reset_if_off_track: bool = True,
        two_way_tracks: bool = False,
        compile: bool = True,
    ) -> None:
        super().__init__()
        logging.info("Initializing simulator")

         # Config
        self.device = device
        self.compile = compile
        torch.set_default_device(self.device)
        self.dt = torch.tensor(dt)
        self.sim_constants = constants
        self.state_dim = self.sim_constants.DEFAULT_STATE_DIMENSION
        self.num_envs = num_envs
        self.reset_if_off_track = reset_if_off_track
        self.two_way_tracks = two_way_tracks

        reward_types = {
            "rl": self._compute_rl_reward,
            "mppi": self._compute_mppi_reward,
        }
        if reward_type in reward_types.keys():
            self._compute_reward = reward_types[reward_type]
        else:
            raise ValueError(f"Unknown reward type: {reward_type}. Available types: {list(reward_types.keys())}")

        
        self.single_track_model = SingleTrackPacejkaModel()  # SingleTrackPacejkaModel SingleTrackPacejkaModelNoDelay 
        # Compile methods
        if self.compile:
            self.single_track_model = torch.compile(
                self.single_track_model,
                mode='max-autotune-no-cudagraphs',
                # fullgraph=True
            )
            self.forward = torch.compile(
                self.forward,
                mode='max-autotune-no-cudagraphs',
                # fullgraph=True
            )
            self._calculate_observation = torch.compile(
                self._calculate_observation,
                mode='max-autotune-no-cudagraphs',
                # fullgraph=True
            )
            logging.info("Methods compiled")

        # Constants
        self.min_speed = torch.tensor(self.sim_constants.MIN_VELOCITY)
        self.max_speed = torch.tensor(self.sim_constants.MAX_VELOCITY)
        self.progres_reward = torch.tensor(self.sim_constants.PROGRESS_REWARD_SCALE)
        self.off_track_reward = torch.tensor(self.sim_constants.OFF_TRACK_PENALTY)
        self.off_track_scaler = torch.tensor(self.sim_constants.OFF_TRACK_SCALER)
        self.slip_reward = torch.tensor(self.sim_constants.SLIP_REWARD)
        self.centerline_reward = torch.tensor(self.sim_constants.CENTERLINE_REWARD)
        self.heading_diff_reward = torch.tensor(self.sim_constants.HEADIND_DIFF_REWARD)

        # State [x, y, yaw, v_x, v_y, r, omega_wheels, omega_wheels_ref, delta, friction, delta_ref, omega_dot]
        self.state = torch.zeros(
            (num_envs, self.state_dim), 
            dtype=torch.float32,
        )
        self.state_names = STATE_DEF_LIST
        self.control_names = ["delta", "omega_dot"]

        self.iteration_count = 0
        self.eps = torch.tensor(1e-6)
        self.integration_method = integration_method

        self.obs_config = obs_config
        self.normalize_observations = normalize_observations
        self.randomize = rand_config is not None

        self._load_params(
            vehicle_config=vehicle_config,
            tire_config=tire_config,
            rand_config=rand_config,
        )

            
        self._load_tracks(tracks)
        self._initialize_state()
        self._initialize_history()

        if self.randomize:
            self._randomize_params()

        logging.info("Simulator initialized")

    @torch.no_grad()
    def forward(self, u):
        """Simulates one step of the vehicle dynamics given control inputs.

        Args:
            u (torch.tensor): Control inputs tensor of shape [batch_size, 2] containing:
                - u[:,0]: Normalized steering angle in range [-1,1]
                - u[:,1]: Normalized motor current in range [-1,1]

        Returns:
            tuple:
                - state (torch.tensor): Current vehicle state [batch_size, state_dim]
                - reward (torch.tensor): Reward value for current step [batch_size]
                - observation (torch.tensor): Current observation of vehicle state [batch_size, obs_dim]
                - off_track (torch.tensor): Boolean tensor indicating if vehicle is off track [batch_size]
        """
        # Delay steering and current control
        steering = u[:, 0] * self.sim_constants.MAX_STEERING
        current = u[:, 1]  * self.sim_constants.MAX_OMEGA_DOT # TODO reverse to self.sim_constants.MAX_OMEGA_DOT

        self.delta_ref_history.add(steering)
        self.omega_dot_history.add(current)

        u = torch.stack([steering, current], dim=1)

        friction = self.get_friction()

        self._step(u, friction)

        # Update state history
        self.vx_history.add(self.state[:, 3])
        self.vy_history.add(self.state[:, 4])
        self.r_history.add(self.state[:, 5])
        self.omega_history.add(self.state[:, 6])
        self.omega_ref_history.add(self.state[:, 7])
        self.delta_history.add(self.state[:, 8])

        observation = self._calculate_observation()
        reward = self._compute_reward()

        self.iteration_count += 1
        if self.reset_if_off_track and torch.any(self.off_track):
            self._reset_off_track()

        return self.state, reward, observation, self.off_track

    def _step(self, u, friction):
        """
        u: [delta, omega_dot]
        friction: [batch_size]
        """
        self.state[:, 9] = friction
        self.state[:, 10] = u[:, 0]  # delta [rad] 
        self.state[:, 11] = u[:, 1]  #  TODO revers 7 to 11 omega_dot - wheel speed derivitive [m/s] 

        self.state = self._integrate(self.single_track_model, self.state, self.dt)

        self.state[:, 3] = torch.clamp(self.state[:, 3], self.min_speed, self.max_speed)
        self.state[:, 6] = torch.clamp(self.state[:, 6], self.min_speed+self.eps, 2. * self.max_speed)
        self.state[:, 7] = torch.clamp(self.state[:, 7], self.min_speed+self.eps, 2. * self.max_speed)
        self.state[:, 2] = torch.atan2(
            torch.sin(self.state[:, 2]), torch.cos(self.state[:, 2])
        )


    def _compute_rl_reward(self):
        omega_dot_hist = self.omega_dot_history.get_history()
        delta_hist = self.delta_ref_history.get_history()
        reg_omega_dot = (torch.pow(omega_dot_hist[:, 0], 2) * self.sim_constants.INPUT_REGURGITATION 
                 + torch.pow(omega_dot_hist[:, 1] - omega_dot_hist[:, 0], 2)
                 * self.sim_constants.INPUT_CHANGE_REGURGITATION) / self.sim_constants.MAX_OMEGA_DOT
        reg_delta = (torch.pow(delta_hist[:, 0], 2) * self.sim_constants.INPUT_REGURGITATION
                       + torch.pow(delta_hist[:, 1] - delta_hist[:, 0], 2) 
                       * self.sim_constants.INPUT_CHANGE_REGURGITATION) / self.sim_constants.MAX_STEERING
        
        off_track_reward = - self.off_track_scaler * self.dist_to_track**2
        dist_centerline_reward = self.centerline_reward * self.dist_to_centerline**2
        heading_diff_reward = self.heading_diff_reward * self.heading_diff**2

        slip_diff = (self.beta_kin - self.beta_dyn)**2
        slip_reward = slip_diff * self.slip_reward

        progress_reward = self.progres_reward * (1 + (self.steps_no_off_track * self.dt / self.sim_constants.SEC_TO_DOUBLE_REWARD))

        reward = self.progress * progress_reward / self.dt

        reward = reward / 10

        reward = torch.where(self.off_track_small, off_track_reward, reward)

        if self.reset_if_off_track:
           off_track_r = self.off_track_reward
           reward = torch.where(self.off_track, off_track_r, reward)

        self.all_reward += reward
        return reward

    def _compute_mppi_reward(self):
        progress_reward = self.v_f * self.progres_reward
        velocity_reward = -abs(self.v_f - 1.0) * self.progres_reward
        slip_diff = -(self.beta_kin - self.beta_dyn)**2 
        heading_diff_reward = -self.heading_diff_reward * self.heading_diff**2 * 10
        dist_centerline_reward = -self.centerline_reward * self.dist_to_centerline**2 * 10
        slip_angle_reward = -100. * torch.maximum(torch.atan2(self.state[:, 4], self.state[:, 3]).abs() - 0.3, torch.zeros(1))
        off_track_constraint = self.current_track_width / 2 - self.dist_to_centerline
        off_track_barrier = 100. * torch.log(1. + torch.exp(-100. * off_track_constraint))
        off_track_barrier = torch.minimum(off_track_barrier, 1000. * torch.ones(1))
        off_track_reward = -off_track_barrier
        #reward = progress_reward + off_track_reward + slip_diff + heading_diff_reward + dist_centerline_reward 
        #reward = progress_reward + off_track_reward + slip_angle_reward
        reward = velocity_reward + off_track_reward + slip_angle_reward
        #reward = torch.where(self.off_track, off_track_reward, reward)
        #reward = torch.where(self.off_track, self.off_track_reward, reward)

        self.all_reward += reward
        return reward

    @torch.no_grad()
    def _calculate_observation(self):
        # Get state components
        pos = self.state[:, :2]  # [batch_size, 2]
        yaw = self.state[:, 2]   # [batch_size]
        
        # Pre-allocate track points tensor
        track_points = torch.stack([
            self.track_x, 
            self.track_y
        ], dim=1)  # [batch_size, 2, track_size]

        pos = pos.unsqueeze(1)  # [batch_size, 1, 2]
        track_points = track_points.permute(0, 2, 1)  # [batch_size, track_size, 2]
        
        # Vectorized distance calculation
        distances = torch.cdist(
            pos, 
            track_points,
            p=2.0
        )  # [batch_size, track_size]
        distances = distances.squeeze(1)
        car_pos = pos.squeeze(1) # [batch_size, 2]


        # Find closest points
        closest_dists, closest_idx = torch.min(distances, dim=1)
        self.closest_idx  = closest_idx.clone()

        # Second closest point
        batch_idx = torch.arange(self.num_envs)

        next_idx = (closest_idx + 1) % self.track_size
        prev_idx = (closest_idx - 1) % self.track_size

        adjacent_dists = torch.stack([
        distances[batch_idx, next_idx],
        distances[batch_idx, prev_idx]
        ], dim=1)
        # dists_tesnot = dists_tesnot.permute(1, 0)
        is_next = torch.argmin(adjacent_dists, dim=1) == 0
        closest_dists_2 = torch.where(is_next, next_idx, prev_idx)

        idx1 = batch_idx, closest_idx
        idx2 = batch_idx, closest_dists_2

        # Calculate track segment vector and position vector
        track_vec = torch.stack([
            self.track_x[idx2] - self.track_x[idx1],
            self.track_y[idx2] - self.track_y[idx1]
        ], dim=1)
        
        pos_vec = torch.stack([
            car_pos[:, 0] - self.track_x[idx1],
            car_pos[:, 1] - self.track_y[idx1]
        ], dim=1)
        
        # Calculate projection coefficient
        s_dist = self.track_s[idx2] - self.track_s[idx1]
        t = (torch.sum(track_vec * pos_vec, dim=1) / 
            (s_dist * s_dist + 1e-6)).clamp(0, 1)
        
        # Interpolate s and heading
        now_s = (1 - t) * self.track_s[idx1] + t * self.track_s[idx2]
        heading = (1 - t) * self.track_heading[idx1] + t * self.track_heading[idx2]
        local_curvature = (1 - t) * self.track_curvature[idx1] + t * self.track_curvature[idx2]

        self.progress = now_s - self.last_s

        # Handle track loop (if progress is big and negative car looped around)
        self.progress = torch.where(
            self.progress < -0.95 * self.track_lengths,
            self.progress + self.track_lengths,
            self.progress
        )
        self.last_s = now_s
        
        # Calculate cross product to determine track side
        next_idx = (closest_idx + 1) % self.track_size
        closest_pos = torch.stack([
            self.track_x[batch_idx, closest_idx],
            self.track_y[batch_idx, closest_idx]
        ], dim=1)
        next_pos = torch.stack([
            self.track_x[batch_idx, next_idx],
            self.track_y[batch_idx, next_idx]
        ], dim=1)
        
        car_to_closest = closest_pos - car_pos
        closest_to_next = next_pos - closest_pos
        cross = car_to_closest[:, 0] * closest_to_next[:, 1] - car_to_closest[:, 1] * closest_to_next[:, 0]
        track_side = torch.sign(cross)
        
        # Calculate heading difference
        heading_diff = yaw - heading
        heading_diff = torch.atan2(
            torch.sin(heading_diff),
            torch.cos(heading_diff)
        )
        self.heading_diff = heading_diff
        self.heading_diff_history.add(heading_diff)

        # Caclaute distance to track centerline
        n = - torch.sin(heading) * (car_pos[:, 0] - closest_pos[:, 0]) + torch.cos(heading) * (car_pos[:, 1] - closest_pos[:, 1])

        # Cacaluate velocity in Fernet frame
        v_x = self.state[:, 3]
        v_y = self.state[:, 4]
        self.v_f = ((v_x * torch.cos(heading_diff) - v_y * torch.sin(heading_diff)) 
               / (1 - n * local_curvature))
        
        # Clacaute dynamical and kinematic slip angles
        self.beta_dyn = torch.atan2(v_y, v_x)
        self.beta_kin = torch.atan(
            torch.tan(self.state[:, 8]) * self.vehicle_params[:, 3] / 
            (self.vehicle_params[:, 2] + self.vehicle_params[:, 3])
        )
        
        # Check if off track
        track_width = self.track_width[batch_idx, closest_idx]
        self.off_track = torch.abs(closest_dists) > (track_width / 2 + self.sim_constants.OFF_TRACK_DISTANCE)
        self.off_track_small = torch.abs(closest_dists) > (track_width / 2)  # offtrack without OFF_TRACK_DISTANCE
        self.dist_to_centerline = torch.abs(closest_dists)
        self.dist_to_track = torch.maximum(self.dist_to_centerline - track_width / 2, torch.zeros(1))
        dist_to_edege = track_width / 2 - self.dist_to_centerline
        self.current_track_width = track_width
        self.steps_no_off_track = torch.where(self.off_track, torch.zeros(1), self.steps_no_off_track + 1)
        
        # Signed distance to track
        self.closest_dists = closest_dists * track_side
        self.closest_dists_history.add(self.closest_dists)
        
        # Update progress metrics
        self.all_progress += self.progress
        self.all_out_of_track += self.off_track.int()

        # Create sin cos progres obs
        progres_norm = now_s / self.track_lengths
        progres_sin = torch.sin(progres_norm * 2 * 3.1415)
        progres_cos = torch.cos(progres_norm * 2 * 3.1415)

        obs_dict = {
            "vx": self.vx_history.get_history(),
            "vy": self.vy_history.get_history(),
            "r": self.r_history.get_history(),
            "omega_wheels": self.omega_history.get_history(),
            "omega_wheels_ref": self.omega_ref_history.get_history(),
            "delta_history": self.delta_history.get_history(),
            "delta_ref_history": self.delta_ref_history.get_history(),
            "omega_dot_history": self.omega_dot_history.get_history(),
            "progres_sin": progres_sin,
            "progres_cos": progres_cos,
            "heading_diff": self.heading_diff_history.get_history(),
            "closest_dist": self.closest_dists_history.get_history(),
            "curvatures": self.track_curvature,
            "widths": self.track_width,
            "dist_to_edge": dist_to_edege,
            "friction": self.friction,
        }

        observation = torch.zeros(
            (self.num_envs, self.obs_config.obs_dim), 
        )
        obs_idx = 0

        # TODO clean up obs creation 

        for obs in self.obs_config.observation_list:
            if obs.iterable:
                # Create indices tensor on GPU
                indices = (torch.arange(obs.length).unsqueeze(0) + 
                        closest_idx.unsqueeze(1))
                
                # Wrap around using modulo
                indices = indices % self.track_size.unsqueeze(1)
                
                # Create batch indices
                batch_idx = torch.arange(self.num_envs).unsqueeze(1)
                
                # Index the observation dictionary
                obss = obs_dict[obs.name][batch_idx, indices][:, ::obs.sample_rate]
                
                # Update observation tensor
                observation[:, obs_idx:obs_idx + obs.dim] = obss

                
                # normalize_observations if needed
                if self.normalize_observations:
                    observation[:, obs_idx:obs_idx + obs.dim] = (
                        observation[:, obs_idx:obs_idx + obs.dim] / obs.max_val
                    )
                    
                obs_idx += obs.dim
            else:
                if obs.dim > 1:
                    observation[:, obs_idx : obs_idx + obs.dim] = obs_dict[obs.name][:, 0 : obs.dim]
                    if self.normalize_observations:
                        observation[:, obs_idx : obs_idx + obs.dim] = (
                            observation[:, obs_idx : obs_idx + obs.dim]
                        ) / (obs.max_val)
                    # if self.randomize and obs.rand is not None:
                    #     sigma = torch.tensor(obs.rand)
                    #     observation[:, obs_idx : obs_idx + obs.dim] = (
                    #         observation[:, obs_idx : obs_idx + obs.dim]
                    #     ) * torch.normal(1, sigma, size=(self.num_envs, obs.dim), device=self.device)
                    obs_idx += obs.dim
                else:
                    new_obs = obs_dict[obs.name]
                    if len(new_obs.shape) > 1:
                        observation[:, obs_idx] = obs_dict[obs.name][:, 0]
                    else:
                        observation[:, obs_idx] = obs_dict[obs.name]

                    if self.normalize_observations:
                        observation[:, obs_idx] = (observation[:, obs_idx]) / (
                            obs.max_val
                        )
                    # if self.randomize and obs.rand is not None:
                    #     sigma = torch.tensor(obs.rand)
                    #     observation[:, obs_idx] = (
                    #         observation[:, obs_idx]
                    #     ) * torch.normal(1, sigma, size=(self.num_envs,), device=self.device)
                    obs_idx += 1

        return observation

    def _reset_off_track(self):
        x, y, yaw, last_s, start_idx = self._generate_start()

        # set start pose only for envs that are off track
        off_track = self.off_track.bool()
        self.state[:, 0] = torch.where(off_track, x, self.state[:, 0])
        self.state[:, 1] = torch.where(off_track, y, self.state[:, 1])
        self.state[:, 2] = torch.where(off_track, yaw, self.state[:, 2])
        self.last_s = torch.where(self.off_track, last_s, self.last_s)

        self.state[:, 3] = torch.where(off_track, self.min_speed, self.state[:, 3])
        self.state[:, 4] = torch.where(off_track, 0.0, self.state[:, 4])
        self.state[:, 5] = torch.where(off_track, 0.0, self.state[:, 5])
        self.state[:, 6] = torch.where(off_track, self.min_speed+self.eps, self.state[:, 6])
        self.state[:, 7] = torch.where(off_track, self.min_speed+self.eps, self.state[:, 7])

        self.closest_idx = torch.where(off_track, start_idx, self.closest_idx)

        self._reset_history(off_track)

    def _reset_history(self, env_indices):
        self.vx_history.reset(env_indices)
        self.vy_history.reset(env_indices)
        self.r_history.reset(env_indices)
        self.omega_history.reset(env_indices)
        self.omega_ref_history.reset(env_indices)
        self.delta_history.reset(env_indices)
        self.delta_ref_history.reset(env_indices)
        self.omega_dot_history.reset(env_indices)
        self.heading_diff_history.reset(env_indices)
        self.closest_dists_history.reset(env_indices)


    def reset(self):
        """"
        Resets the simulator to the initial state and returns the initial observation.
        """
        info = {}
        observation = self._calculate_observation()

        if self.iteration_count > 0:
            average_speed = torch.mean(self.all_progress) / (self.iteration_count * self.dt)
            average_off_track = torch.mean(self.all_out_of_track.float()) / (self.iteration_count * self.dt)
            average_reward = torch.mean(self.all_reward) / self.iteration_count
        else:
            average_speed = 0
            average_off_track = 0
            average_reward = 0

        self.iteration_count = 0

        self._initialize_history()
        self._initialize_state()
        if self.randomize:
            self._randomize_params()

        info["all"] = Info(average_speed, average_off_track, average_reward)

        return self.state, info, observation

    def _generate_start(self, random: bool = True):
        if random:
            rand_ = torch.rand(self.num_envs)
            start_idx = (rand_ * self.track_size).int()
        else:
            start_idx = torch.zeros(self.num_envs, dtype=torch.int64)

        start_x = self.track_x[torch.arange(self.num_envs), start_idx]
        start_y = self.track_y[torch.arange(self.num_envs), start_idx]
        start_yaw = self.track_heading[torch.arange(self.num_envs), start_idx]

        last_s = self.track_s[torch.arange(self.num_envs), start_idx]

        return start_x, start_y, start_yaw, last_s, start_idx

    def default_controls(self, horizon: int = 1):
        s = self.last_s
        times = torch.arange(horizon + 1)[1:] * self.dt
        vx = self.state[:, 3]
        next_s = s[:, None] + times[None, :] * vx[:, None]
        s_dists = (self.track_s[:, None] - next_s[:, :, None]).abs()
        tracks_idxs = torch.argmin(s_dists, dim=2)
        curvatures = self.track_curvature[torch.arange(self.num_envs), tracks_idxs]
        L = self.vehicle_params[:, 2] + self.vehicle_params[:, 3]
        steering_angles = torch.atan(curvatures * L)
        currents = torch.zeros((self.num_envs, horizon), dtype=torch.float32)
        controls = torch.stack([steering_angles, currents], dim=-1)
        return controls

    def _integrate(self, f, xu, dt):
        if self.integration_method == "rk4":
            return self._rk4_step(f, xu, dt)
        elif self.integration_method == "euler":
            return self._euler_step(f, xu, dt)
        else:
            raise NotImplementedError

    def _rk4_step(self, f, xu, dt):
        t = torch.zeros(1, dtype=torch.float32)
        k1 = f(t, xu, self.vehicle_params, self.tire_front_params, self.tire_rear_params)
        k2 = f(t, xu + dt / 2 * k1, self.vehicle_params, self.tire_front_params, self.tire_rear_params)
        k3 = f(t, xu + dt / 2 * k2, self.vehicle_params, self.tire_front_params, self.tire_rear_params)
        k4 = f(t, xu + dt * k3, self.vehicle_params, self.tire_front_params, self.tire_rear_params)
        return dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4) + xu

    def _euler_step(self, f, xu, dt):
        t = torch.zeros(1, dtype=torch.float32)
        return xu + dt * f(t, xu, self.vehicle_params, self.tire_front_params, self.tire_rear_params)

    def _load_params(self, vehicle_config, tire_config, rand_config):
        config_path = os.path.join(os.path.dirname(__file__), "config")
        with open(os.path.join(config_path, f"vehicle/{vehicle_config}.yaml")) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        vehicle_params = params["Vehicle"]
        tire_front_params = params["TireFront"]
        tire_rear_params = params["TireRear"]

        # Load vehicle parameters
        vehicle_parameters_object = VehicleParameters()
        self.vehicle_params = torch.tensor(
            [list(vehicle_params.values())] * self.num_envs, dtype=torch.float32
        )
        self.vehicle_params_orig = self.vehicle_params.clone()
        self.vehicle_params_names = list(vehicle_params.keys())

        # Load tires paramters
        self.tire_front_params = torch.tensor(
            [list(tire_front_params.values())] * self.num_envs, dtype=torch.float32
        )
        self.tire_rear_params = torch.tensor(
            [list(tire_rear_params.values())] * self.num_envs, dtype=torch.float32
        )
        # create tire param conts 
        self.tire_front_params_orig = self.tire_front_params.clone()
        self.tire_rear_params_orig = self.tire_rear_params.clone()

        if rand_config is not None:
            with open(os.path.join(config_path, f"randomization/{rand_config}.yaml")) as f:
                self.rand_params = yaml.load(f, Loader=yaml.FullLoader)

    def _randomize_params(self):
        print("Randomizing vehicle parameters")
        # Randomize vehicle parameters
        for param in self.rand_params.keys():
            if param in self.vehicle_params_names:
                self._randmize_param(param)
            elif param == "friction":
                self._randomize_friction()
            elif param == "tire":
                self._randomize_tire()
            else:
                logging.error(
                        f"Parameter '{param}' "
                        "that you want to randomize was not found in vehicle parameters"
                    )

    def _randmize_param(self, param):
        param_idx = self.vehicle_params_names.index(param)
        sigma = self.rand_params[param]
        rand = torch.normal(
            1, sigma, size=(self.num_envs,), device=self.device
        )

        # clip to avoid values close to zero
        if "tau" in param:
            # clip time constans more agresivly to avoid NaNs
            min_rand = torch.tensor(0.4)
        else:
            min_rand = torch.tensor(0.1)
        max_rand = torch.tensor(2.0)
        rand_clamp = torch.clamp(rand, min_rand, max_rand)

        self.vehicle_params[:, param_idx] = self.vehicle_params_orig[:, param_idx] * rand_clamp

        print(f"Randomizing parameter: {param} min param value: {torch.min(self.vehicle_params[:, param_idx])}")

        # if any valeu is negative log error
        if torch.any(self.vehicle_params[:, param_idx] < 0):
            logging.error(
                f"Randomized parameter '{param}' "
                f"resulted in negative value with rand_clamp: {rand_clamp}, min_rand: {min_rand}, max_rand: {max_rand}"
            )

    def _randomize_friction(self):
        friction_sigma = self.rand_params["friction"]
        base_friction = self.sim_constants.FRICTION

        for env_idx in range(self.num_envs):
            num_control_points = torch.randint(3, 7, (1,)) 
            track_size = self.track_size[env_idx]
            x_values = np.linspace(0, track_size-1, track_size)

            control_x = torch.tensor(torch.rand(num_control_points-2) * track_size)
            control_x = torch.sort(control_x)[0]
            extended_x = torch.cat([torch.tensor([0]), control_x, torch.tensor([track_size-1])])

            control_y = torch.normal(base_friction, friction_sigma, (num_control_points-1,))
            control_y = torch.clamp(control_y, 0.2, 2.0)  # Clamp to avoid negative values
            extended_y = torch.cat([control_y, control_y[0].unsqueeze(0)])
            
            # Interpolate between control points
            interpolated_values = np.interp(x_values, extended_x, extended_y)

            # Smooth the resulting profile
            smoothed_values = gaussian_filter1d(interpolated_values, 5)
            smoothed_values = torch.tensor(smoothed_values, device=self.device)

            # Set the friction values
            self.friction[env_idx, :track_size] = smoothed_values

    def get_friction(self):
        return self.friction[torch.arange(self.num_envs), self.closest_idx]
    
    def set_friction_curve(self, env_idx, x_points, y_points):
        """
        Set the friction curve for a specific environment.

        Args:
            env_idx (int): Index of the environment.
            x_points (list): List of x-coordinates for the friction curve noralize to 0, 1.
            y_points (list): List of y-coordinates for the friction curve.
        """
        x_points = torch.tensor(x_points, device=self.device)
        x_points = torch.clamp(x_points, 0, 1) * (self.track_size[env_idx] - 1)
        track_size = self.track_size[env_idx]
        x_values = np.linspace(0, track_size-1, track_size)

        print(f"Setting friction curve for env {env_idx} with x_points: {x_points} and y_points: {y_points}")

        # Interpolate between control points
        interpolated_values = np.interp(x_values, x_points, y_points)

        # Smooth the resulting profile
        smoothed_values = gaussian_filter1d(interpolated_values, 5)
        smoothed_values = torch.tensor(smoothed_values, device=self.device)

        # Set the friction values
        self.friction[env_idx, :track_size] = smoothed_values

        # plot 
        plt.plot(x_values, smoothed_values.cpu().numpy(), label=f"env {env_idx}")

    
    def get_friction_map(self):
        return self.friction.cpu().numpy()

    def _randomize_tire(self):
        tire_sigma = self.rand_params["tire"]
        tire_rand_front = torch.normal(
            1, tire_sigma, size=(self.num_envs, self.tire_front_params.shape[1]), device=self.device
        )
        tire_rand_rear = torch.normal(
            1, tire_sigma, size=(self.num_envs, self.tire_rear_params.shape[1]), device=self.device
        )

        self.tire_front_params = self.tire_front_params_orig * tire_rand_front
        self.tire_rear_params = self.tire_rear_params_orig * tire_rand_rear

    def _load_tracks(self, tracks: list):
        # TODO @Grzegorz check if my implementation of two way tracks is correct and if `self.two_way_tracks` is needed anywhere else
        print(f"Loading tracks: {tracks}")
        self.numbers_of_tracks = len(tracks)
        envs_per_track = self.num_envs // self.numbers_of_tracks
        envs_per_track_rest = self.num_envs % self.numbers_of_tracks

        # Longest track size is 75 m (750 points)
        max_track_size = self.sim_constants.MAX_TRACK_SIZE

        # Create friction tensor
        self.friction = torch.full((self.num_envs, max_track_size), self.sim_constants.FRICTION)
        
        # Pre-allocate tensors
        tensor_shape = (int(self.num_envs), max_track_size)
        pad_value = 1e7
        
        self.track_x = torch.full(tensor_shape, pad_value)
        self.track_y = torch.full(tensor_shape, pad_value)
        self.track_heading = torch.full(tensor_shape, pad_value)
        self.track_width = torch.full(tensor_shape, pad_value)
        self.track_curvature = torch.full(tensor_shape, pad_value)
        self.track_s = torch.full(tensor_shape, pad_value)
        
        # Store original track lengths and track size
        self.track_lengths = torch.zeros((self.num_envs), dtype=torch.float32)
        self.track_size = torch.zeros((self.num_envs), dtype=torch.int64)

        # craet dict with trax name and idx range name: int idx-start int idx-end
        self.tracks_idx = {}
        idx = 0

        if self.num_envs == 1:
            track_reader = TrackReader(tracks[0])
            self._append_track(track_reader, idx)
            self.tracks_idx[tracks[0]] = (idx, idx)
        else:
            for track in tracks:
                self.tracks_idx[track] = (idx, idx + envs_per_track - 1)

                n_envs = envs_per_track if not self.two_way_tracks else envs_per_track // 2
                track_reader = TrackReader(track)
                for _ in range(n_envs):
                    self._append_track(track_reader, idx)
                    idx += 1
                # append flipped envs
                if self.two_way_tracks:
                    track_reader_flip = TrackReader(track, flip=True)
                    for _ in range(envs_per_track - n_envs):
                        self._append_track(track_reader_flip, idx)
                        idx += 1

            if envs_per_track_rest > 0:
                track_reader = TrackReader(tracks[0])
                for _ in range(envs_per_track_rest):
                    self._append_track(track_reader, idx)
                    idx += 1

    def _append_track(self, track_reader: TrackReader, idx: int):
        path_s, x, y, width, curvature, heading = track_reader.preprocess_track(plot=False)

        self.track_lengths[idx] = torch.tensor(path_s[-1])
        self.track_size[idx] = torch.tensor(len(x))

        self.track_x[idx, : len(x)] = torch.tensor(x)
        self.track_y[idx, : len(y)] = torch.tensor(y)
        self.track_heading[idx, : len(heading)] = torch.tensor(heading)
        self.track_width[idx, : len(width)] = torch.tensor(width)
        self.track_curvature[idx, : len(curvature)] = torch.tensor(curvature)
        self.track_s[idx, : len(path_s)] = torch.tensor(path_s)

    def _initialize_state(self, start_at_zero: bool = False):
        self.state = torch.zeros((self.num_envs, self.state_dim), dtype=torch.float32)
        self.state[:, 3] = self.min_speed

        random = not start_at_zero
        x, y, yaw, last_s, start_idx = self._generate_start(random)
        self.state[:, 0] = x
        self.state[:, 1] = y
        self.last_s = last_s
        self.state[:, 2] = yaw

        self.closest_idx = start_idx

        self.steps_no_off_track = torch.zeros((self.num_envs), dtype=torch.int64)

    def _initialize_history(self):
        self.delta_ref_history = HistoryBuffer(self.sim_constants.HISTORY_SIZE, self.num_envs)
        self.omega_dot_history = HistoryBuffer(self.sim_constants.HISTORY_SIZE, self.num_envs)
        self.vx_history = HistoryBuffer(self.sim_constants.HISTORY_SIZE, self.num_envs)
        self.vy_history = HistoryBuffer(self.sim_constants.HISTORY_SIZE, self.num_envs)
        self.r_history = HistoryBuffer(self.sim_constants.HISTORY_SIZE, self.num_envs)
        self.omega_history = HistoryBuffer(self.sim_constants.HISTORY_SIZE, self.num_envs)
        self.omega_ref_history = HistoryBuffer(self.sim_constants.HISTORY_SIZE, self.num_envs)
        self.delta_history = HistoryBuffer(self.sim_constants.HISTORY_SIZE, self.num_envs)
        self.closest_dists_history = HistoryBuffer(self.sim_constants.HISTORY_SIZE, self.num_envs)
        self.heading_diff_history = HistoryBuffer(self.sim_constants.HISTORY_SIZE, self.num_envs)

        self.all_out_of_track = torch.zeros((self.num_envs), dtype=torch.int64)
        self.all_progress = torch.zeros((self.num_envs), dtype=torch.float32)
        self.all_reward = torch.zeros((self.num_envs), dtype=torch.float32)

    def get_state(self):
        """
        Returns:
            torch.Tensor: Current state of the simulator [batch_size, state_dim]
        """
        return self.state

    def set_s(self, s):
        """
        Args:
            s (torch.Tensor): New s coordinate of the simulator [batch_size]
        """
        self.last_s = torch.ones(self.num_envs) * s

    def get_s(self):
        """
        Returns:
            float: Current s coordinate of the simulator
        """
        return self.last_s.item()
    
    def get_clostest_dist(self):
        """
        Returns:
            float: Distance to the closest point on the track
        """
        return self.closest_dists.item()
    
    def get_heading_diff(self):
        """
        Returns:
            float: Heading difference between the vehicle and the track
        """
        return self.heading_diff.item()
    
    def get_x(self):
        return self.state[:, 0].item()
    
    def get_y(self):
        return self.state[:, 1].item()
    
    def get_yaw(self):
        return self.state[:, 2].item()
    
    def set_state(self, state):
        """
        Args:
            state (torch.Tensor): New state of the simulator [batch_size, state_dim]
        """
        self.state = state

    def set_friciton(self, friction):
        """
        Args:
            friction (torch.Tensor): New friction value [num_envs, track_size]
        """
        self.base_friction = torch.full((self.num_envs, self.track_size[0]), friction)

    def get_closes_idx(self):
        """
        Returns:
            torch.Tensor: Index of the closest point on the track [batch_size]
        """
        return self.closest_idx
    
    def get_track(self):
        """
        Returns:
            torch.Tensor: x coordinates of the track [track_size]
            torch.Tensor: y coordinates of the track [track_size]
            torch.Tensor: width of the track [track_size]
        """
        # if more than one track log error 
        if len(self.tracks_idx.keys()) > 1:
            logging.error("More than one track loaded, returning only first track")

        # return the track without padding
        size = self.track_size[0]
        return self.track_x[0, :size], self.track_y[0, :size], self.track_width[0, :size]
    
    def get_closest_idx(self):
        """
        Returns:
            torch.Tensor: Index of the closest point on the track [batch_size]
        """
        return self.closest_idx
    
    def get_vehicle_params(self):
        """
        Returns:
            torch.Tensor: Vehicle parameters [batch_size, num_vehicle_params]
        """
        return self.vehicle_params

    def close(self):
        logging.info("Closing simulator")