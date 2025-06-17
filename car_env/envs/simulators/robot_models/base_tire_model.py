import torch
from car_env.utils.state_wrapper import StateWrapper


class BaseTireModel(torch.nn.Module):
    def __init__(self):
        super(BaseTireModel, self).__init__()

    @staticmethod
    def lf(wp):
        return wp.L - wp.lr

    @staticmethod
    def slip_angle_front_func(wx, wp):        
        return torch.atan((wx.v_y + wp.lf * wx.r) / (wx.v_x + wp.eps)) - wx.delta

    @staticmethod
    def slip_angle_rear_func(wx, wp):
        return torch.atan((wx.v_y - wp.lf * wx.r) / (wx.v_x + wp.eps))

    @staticmethod
    def slip_ratio_func(wx, wp):
        slip_ratio = (wx.omega_wheels - wx.v_x) / \
            (wx.v_x + wp.eps)
        return slip_ratio

    @staticmethod
    def slip_ratio_front_func(wx, wp):
        v_front = wx.v_x * \
            torch.cos(wx.delta) + (wx.v_y + wx.r *
                                   wp.lr) * torch.sin(wx.delta)
        slip_ratio = (wx.omega_wheels - v_front) / \
            (v_front + wp.eps)
        return slip_ratio
    

    @staticmethod
    def Fz_front(wp):
        return wp.m * wp.g * wp.lr / wp.L   
    
    @staticmethod
    def Fz_rear(wp):
        return wp.m * wp.g * wp.lf / wp.L
    
    

if __name__ == "__main__":
    import single_track_params
    
    # test all the functions
    x = torch.rand(1, 7)
    p = torch.rand(1, 200)
    
    tire = BaseTireModel()
    param_wraper_st = single_track_params.VehicleParameters()
    
    param_wraper_st = torch.compile(param_wraper_st, fullgraph=True)
    
    for i in range(100):
        p = torch.rand(1, 200)
        wp, _ = param_wraper_st(p)
        # print(wp)
    
    x = StateWrapper(x)
    print(tire.slip_angle_front_func(x, wp))
    print(tire.slip_angle_rear_func(x, wp))
    print(tire.slip_ratio_func(x, wp))
    print(tire.slip_ratio_front_func(x, wp))
    