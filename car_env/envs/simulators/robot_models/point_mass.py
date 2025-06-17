import torch
from car_env.utils.state_wrapper import StateWrapper, ParamWrapper, TireWrapper


class PointMass(torch.nn.Module):

    def __init__(self) -> None:
        super(PointMass, self).__init__()
        self.eps = 1e-6

    def forward(self, t, x, vehicle_parameters, tire_parameters) -> torch.Tensor:
        """
        t : float
        x: [batch_size, state_dim]
        p: [batch_size] : torch.nn.Module
        tire_model: [batch_size] : torch.nn.Module
        """
        p = ParamWrapper(vehicle_parameters)
        tire = TireWrapper(tire_parameters)
        wx = StateWrapper(x)

        # Tire Model

        sa_f = torch.atan((wx.v_y + p.lf * wx.r) / (wx.v_x + self.eps)) - wx.delta
        sa_r = torch.atan((wx.v_y - p.lr * wx.r) / (wx.v_x + self.eps))

        v_front = wx.v_x * torch.cos(wx.delta) + (wx.v_y + wx.r * p.lr) * torch.sin(wx.delta)
        sr_f = (wx.omega_wheels - v_front) / (v_front + self.eps)
        sr_r = (wx.omega_wheels - wx.v_x) / (wx.v_x + self.eps)

        Fn_f = p.m * p.g * p.lr / (p.lf + p.lr)
        Fn_r = p.m * p.g * p.lf / (p.lf + p.lr)

        Fy_f = - Fn_f * tire.D_f * torch.sin(tire.C_f * torch.atan(tire.B_f * sa_f)) * wx.friction
        Fy_r = - Fn_r * tire.D_r * torch.sin(tire.C_r * torch.atan(tire.B_r * sa_r)) * wx.friction

        # deviding by 0.6 noramlizes mu_tire TODO fix this
        Fx_f = p.m * p.g * tire.mu_tire * torch.sin(tire.C_long * torch.atan(tire.B_long * sr_f)) * wx.friction / 0.6
        Fx_r = p.m * p.g * tire.mu_tire * torch.sin(tire.C_long * torch.atan(tire.B_long * sr_r)) * wx.friction / 0.6

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

        omega_wheels_dot = p.R / p.I_e * (p.K_fi * wx.Iq - p.R * Fx_f - p.R * Fx_r
                                          - wx.omega_wheels * p.b1 - torch.sign(wx.omega_wheels) * p.b0)

        x_dot = (wx.v_x * torch.cos(wx.yaw) - wx.v_y * torch.sin(wx.yaw))
        y_dot = (wx.v_x * torch.sin(wx.yaw) + wx.v_y * torch.cos(wx.yaw))
        yaw_dot = wx.r

        return torch.tensor([x_dot, y_dot, yaw_dot, v_x_dot, v_y_dot, r_dot, omega_wheels_dot, torch.zeros_like(wx.friction), torch.zeros_like(wx.delta), torch.zeros_like(wx.Iq)])
