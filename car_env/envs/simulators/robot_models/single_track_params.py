import torch
import collections

SingleTrackParamsList = ['m', 'g', 'I_z', 'L', 'lr', 'lf', 'Cd0', 'Cd1', 'Cd2',
                         'mu_static', 'I_e', 'K_fi', 'b0', 'b1', 'R', 'tau_omega', 'tau_delta', 'eps']
SingleTrackParams = collections.namedtuple('SingleTrackParams', SingleTrackParamsList,
                                           defaults=(None,) * len(SingleTrackParamsList))

class VehicleParameters(torch.nn.Module):
    def __init__(self) -> None:
        super(VehicleParameters, self).__init__()
        self.param_count = VehicleParameters.default_params_tensor().shape[-1]
        
        # Define constant tensors for detached parameters

        self.eps_ = torch.tensor([1e-6], requires_grad=False)
        
        self.register_buffer('eps', self.eps_)        
        

    def forward(self, p: torch.Tensor) -> SingleTrackParams:       
        named_tuple = SingleTrackParams(
            m=p[..., 0],
            g=p[..., 1],
            I_z=p[..., 2],
            L=p[..., 3],
            lr=p[..., 4],
            lf=p[..., 3] - p[..., 4],  # lf = L - lr
            Cd0=p[..., 5],
            Cd1=p[..., 6],
            Cd2=p[..., 7],
            mu_static=p[..., 8],
            I_e=p[..., 9],
            K_fi=p[..., 10],
            b0=p[..., 11],
            b1=p[..., 12],
            R=p[..., 13],
            tau_omega=p[..., 14],
            tau_delta=p[..., 15],
            eps=self.eps
        )
        return named_tuple
    

    @staticmethod
    def default_params_tensor(batch_size=1):
        return torch.tensor([
            0.46,  # I_z
            0.115,  # lr
            0.01,  # Cd0
            0.01,  # Cd2
            0.01,  # Cd1
            0.2,  # I_e
            0.90064745,  # K_fi
            0.304115174,  # b1
            0.50421894,  # b0
            0.05,  # R
        ]).unsqueeze(0).repeat(batch_size, 1)
        
    @staticmethod
    def get_params_names():
        return [
            "m",
            "g",
            "I_z",
            "L",
            "lr",
            "lf",
            "Cd0",
            "Cd1",
            "Cd2",
            "mu_static",
            "I_e",
            "K_fi",
            "b0",
            "b1",
            "R",
        ]
        
    
    