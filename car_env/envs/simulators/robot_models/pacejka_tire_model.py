import torch
import car_env.envs.simulators.robot_models.base_tire_model as base_tire_model   
from car_env.utils.state_wrapper import StateWrapper


class PacejkaTireModel(base_tire_model.BaseTireModel):
    def __init__(self) -> None:
        super().__init__()


    @staticmethod
    def tire_forces_model(slip_angle_rad, slip_ratio, wp):
        #[1] Bakker, E., Nyborg, L., and Pacejka, H.B.
        #   Tyre modelling for use in vehicle dynamics studies. United States: N. p., 1987. Web
        
        #[2] W. F. Milliken and D. L. Milliken,
        #   Race Car Vehicle Dynamics.Warrendale, PA, USA: SAE, 1995.
        
        # https://skill-lync.com/student-projects/Combined-slip-correction-using-Pacejka-tire-model-25918
        # https://www.researchgate.net/publication/344073372_Tire_Modeling_Using_Pacejka_Model
        # Unpack the state variables
        
        # alpha from radians to degrees
        slip_angle_deg = slip_angle_rad * 180.0 / torch.pi
        
        # Calculate normalized slip ratio and slip angle
        Sx_norm = slip_ratio / wp.Sx_p
        Alpha_norm = slip_angle_deg / wp.Alpha_p
        
        # Compute the resultant slip
        S_resultant = torch.sqrt(Sx_norm**2 + Alpha_norm**2)
        
        # Find the modified slip factors
        Sx_mod = S_resultant * wp.Sx_p
        Alpha_mod = S_resultant * wp.Alpha_p
        
        # Calculate the Lateral Force using Pacejka formula
        Alpha_final = Alpha_mod #+ wp.Shy
        Fy = ((Alpha_norm / S_resultant) * wp.Dy * torch.sin(wp.Cy * torch.atan((wp.By * Alpha_final) - 
            wp.Ey * -1.0 * (wp.By * Alpha_final - torch.atan(wp.By * Alpha_final))))) #+ wp.Svy
        
        # Calculate the Longitudinal Force using Pacejka formula
        Sx_final = Sx_mod #+ wp.Shx
        Fx = ((Sx_norm / S_resultant) * wp.Dx * torch.sin(wp.Cx * torch.atan((wp.Bx * Sx_final) - 
            wp.Ex * -1.0 * (wp.Bx * Sx_final - torch.atan(wp.Bx * Sx_final))))) #+ wp.Svx
        
        return Fx, - Fy
        
    
    def forward(self, x, wp_st, wp_tire_f, wp_tire_r):

        wx = StateWrapper(x)
        fric_xf, fric_yf = self.tire_forces_model(self.slip_angle_front_func(wx, wp_st),
                                                  self.slip_ratio_front_func(wx, wp_st), wp_tire_f)
        
        Fxf = self.Fz_front(wp_st) * fric_xf
        Fyf = self.Fz_front(wp_st) * fric_yf
        
        fric_xr, firc_yr = self.tire_forces_model(self.slip_angle_rear_func(wx, wp_st),
                                                  self.slip_ratio_func(wx, wp_st), wp_tire_r)
        
        Fxr = self.Fz_rear(wp_st) * fric_xr
        Fyr = self.Fz_rear(wp_st) * firc_yr
        
        return torch.stack([Fyf, Fyr, Fxf, Fxr], dim=-1)
        
        