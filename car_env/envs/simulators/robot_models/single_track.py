import torch
from car_env.utils.state_wrapper import StateWrapper, ParamWrapper, TireWrapper
from car_env.envs.simulators.robot_models.pacejka_params import PacejkaParameters
from car_env.envs.simulators.robot_models.pacejka_tire_model import PacejkaTireModel
from car_env.envs.simulators.robot_models.single_track_params import VehicleParameters

class SingleTrackPacejkaModel(torch.nn.Module):

    def __init__(self) -> None:
        super(SingleTrackPacejkaModel, self).__init__()
        self.eps = 1e-6
        self.tire_model_parameters = PacejkaParameters()
        self.vehicle_parameters = VehicleParameters()

        self.tire_model = PacejkaTireModel()

    def forward(self, t, x, p_vehicle, p_tire_front, p_tire_rear):
        """
        t : float
        x: [batch_size, state_dim]
        p: [batch_size] : torch.nn.Module
        tire_model: [batch_size] : torch.nn.Module
        """
        p = self.vehicle_parameters(p_vehicle)
        wx = StateWrapper(x)

        wp_tire_f = self.tire_model_parameters(p_tire_front)
        wp_tire_r = self.tire_model_parameters(p_tire_rear)

        # Tire Forces
        tire_forces = self.tire_model(x, p, wp_tire_f, wp_tire_r) * wx.friction.unsqueeze(-1)

        Fy_f, Fy_r, Fx_f, Fx_r = torch.unbind(tire_forces, dim=-1)

        # Vehicle Dynamics

        F_drag = p.Cd0 * torch.sign(wx.v_x) +\
            p.Cd1 * wx.v_x +\
            p.Cd2 * wx.v_x * wx.v_x
        

        v_x_dot = 1.0 / p.m * (Fx_r + Fx_f * torch.cos(wx.delta) -
                               Fy_f * torch.sin(wx.delta) - F_drag + p.m * wx.v_y * wx.r)

        v_y_dot = 1.0 / p.m * (Fx_f * torch.sin(wx.delta) +
                               Fy_r + Fy_f * torch.cos(wx.delta) - p.m * wx.v_x * wx.r)

        r_dot = 1.0 / p.I_z * \
            ((Fx_f * torch.sin(wx.delta) + Fy_f *
             torch.cos(wx.delta)) * p.lf - Fy_r * p.lr)

        omega_wheels_dot = (wx.omega_wheels_ref - wx.omega_wheels) / p.tau_omega

        delta_dot = (wx.delta_ref - wx.delta) / p.tau_delta

        x_dot = (wx.v_x * torch.cos(wx.yaw) - wx.v_y * torch.sin(wx.yaw))
        y_dot = (wx.v_x * torch.sin(wx.yaw) + wx.v_y * torch.cos(wx.yaw))
        yaw_dot = wx.r

        if x.dim() == 1:
            return torch.tensor([x_dot, y_dot, yaw_dot, v_x_dot, v_y_dot, r_dot, omega_wheels_dot, wx.omega_wheels_ref_dot, delta_dot, torch.zeros_like(wx.friction), torch.zeros_like(wx.delta), torch.zeros_like(wx.omega_wheels_ref_dot)])
        else:
            return torch.stack([x_dot, y_dot, yaw_dot, v_x_dot, v_y_dot, r_dot, omega_wheels_dot, wx.omega_wheels_ref_dot, delta_dot, torch.zeros_like(wx.friction), torch.zeros_like(wx.delta), torch.zeros_like(wx.omega_wheels_ref_dot)], dim=1)


class SingleTrackPacejkaModelNoDelay(torch.nn.Module):

    def __init__(self) -> None:
        super(SingleTrackPacejkaModelNoDelay, self).__init__()
        self.eps = 1e-6
        self.tire_model_parameters = PacejkaParameters()
        self.vehicle_parameters = VehicleParameters()

        self.tire_model = PacejkaTireModel()

    def forward(self, t, x, p_vehicle, p_tire_front, p_tire_rear):
        """
        t : float
        x: [batch_size, state_dim]
        p: [batch_size] : torch.nn.Module
        tire_model: [batch_size] : torch.nn.Module
        """
        p = self.vehicle_parameters(p_vehicle)
        wx = StateWrapper(x)

        wp_tire_f = self.tire_model_parameters(p_tire_front)
        wp_tire_r = self.tire_model_parameters(p_tire_rear)

        # Tire Forces
        tire_forces = self.tire_model(x, p, wp_tire_f, wp_tire_r) * wx.friction.unsqueeze(-1)

        Fy_f, Fy_r, Fx_f, Fx_r = torch.unbind(tire_forces, dim=-1)

        # Vehicle Dynamics

        # To remove delay
        omega_wheels_dot = torch.zeros_like(wx.omega_wheels)
        delta_dot = torch.zeros_like(wx.delta)

        F_drag = p.Cd0 * torch.sign(wx.v_x) +\
            p.Cd1 * wx.v_x +\
            p.Cd2 * wx.v_x * wx.v_x
        

        v_x_dot = 1.0 / p.m * (Fx_r + Fx_f * torch.cos(wx.delta_ref) -
                               Fy_f * torch.sin(wx.delta_ref) - F_drag + p.m * wx.v_y * wx.r)

        v_y_dot = 1.0 / p.m * (Fx_f * torch.sin(wx.delta_ref) +
                               Fy_r + Fy_f * torch.cos(wx.delta_ref) - p.m * wx.v_x * wx.r)

        r_dot = 1.0 / p.I_z * \
            ((Fx_f * torch.sin(wx.delta_ref) + Fy_f *
             torch.cos(wx.delta_ref)) * p.lf - Fy_r * p.lr)



        x_dot = (wx.v_x * torch.cos(wx.yaw) - wx.v_y * torch.sin(wx.yaw))
        y_dot = (wx.v_x * torch.sin(wx.yaw) + wx.v_y * torch.cos(wx.yaw))
        yaw_dot = wx.r

        if x.dim() == 1:
            return torch.tensor([x_dot, y_dot, yaw_dot, v_x_dot, v_y_dot, r_dot, wx.omega_wheels_ref_dot, torch.zeros_like(wx.omega_wheels), torch.zeros_like(wx.delta), torch.zeros_like(wx.friction), torch.zeros_like(wx.delta), torch.zeros_like(wx.omega_wheels_ref_dot)])
        else:
            return torch.stack([x_dot, y_dot, yaw_dot, v_x_dot, v_y_dot, r_dot, wx.omega_wheels_ref_dot, torch.zeros_like(wx.omega_wheels), torch.zeros_like(wx.delta), torch.zeros_like(wx.friction), torch.zeros_like(wx.delta), torch.zeros_like(wx.omega_wheels_ref_dot)], dim=1)


class SingleTrackPacejkaModelVelocityControls(torch.nn.Module):

    def __init__(self) -> None:
        super(SingleTrackPacejkaModelVelocityControls, self).__init__()
        self.eps = 1e-6
        self.tire_model_parameters = PacejkaParameters()
        self.vehicle_parameters = VehicleParameters()

        self.tire_model = PacejkaTireModel()

    def forward(self, t, x, p_vehicle, p_tire_front, p_tire_rear):
        """
        t : float
        x: [batch_size, state_dim]
        p: [batch_size] : torch.nn.Module
        tire_model: [batch_size] : torch.nn.Module
        """
        p = self.vehicle_parameters(p_vehicle)
        wx = StateWrapper(x)

        wp_tire_f = self.tire_model_parameters(p_tire_front)
        wp_tire_r = self.tire_model_parameters(p_tire_rear)

        # Tire Forces
        tire_forces = self.tire_model(x, p, wp_tire_f, wp_tire_r) * wx.friction.unsqueeze(-1)

        Fy_f, Fy_r, Fx_f, Fx_r = torch.unbind(tire_forces, dim=-1)

        # Vehicle Dynamics

        F_drag = p.Cd0 * torch.sign(wx.v_x) +\
            p.Cd1 * wx.v_x +\
            p.Cd2 * wx.v_x * wx.v_x
        

        v_x_dot = 1.0 / p.m * (Fx_r + Fx_f * torch.cos(wx.delta) -
                               Fy_f * torch.sin(wx.delta) - F_drag + p.m * wx.v_y * wx.r)

        v_y_dot = 1.0 / p.m * (Fx_f * torch.sin(wx.delta) +
                               Fy_r + Fy_f * torch.cos(wx.delta) - p.m * wx.v_x * wx.r)

        r_dot = 1.0 / p.I_z * \
            ((Fx_f * torch.sin(wx.delta) + Fy_f *
             torch.cos(wx.delta)) * p.lf - Fy_r * p.lr)

        omega_wheels_dot = (wx.omega_wheels_ref - wx.omega_wheels) / p.tau_omega

        delta_dot = (wx.delta_ref - wx.delta) / p.tau_delta

        x_dot = (wx.v_x * torch.cos(wx.yaw) - wx.v_y * torch.sin(wx.yaw))
        y_dot = (wx.v_x * torch.sin(wx.yaw) + wx.v_y * torch.cos(wx.yaw))
        yaw_dot = wx.r

        if x.dim() == 1:
            return torch.tensor([x_dot, y_dot, yaw_dot, v_x_dot, v_y_dot, r_dot, omega_wheels_dot, torch.zeros_like(wx.omega_wheels_ref_dot), delta_dot, torch.zeros_like(wx.friction), torch.zeros_like(wx.delta), torch.zeros_like(wx.omega_wheels_ref_dot)])
        else:
            return torch.stack([x_dot, y_dot, yaw_dot, v_x_dot, v_y_dot, r_dot, omega_wheels_dot, torch.zeros_like(wx.omega_wheels_ref_dot), delta_dot, torch.zeros_like(wx.friction), torch.zeros_like(wx.delta), torch.zeros_like(wx.omega_wheels_ref_dot)], dim=1)
