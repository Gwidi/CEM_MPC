import collections
import torch
import logging

# Define the Pacejka tire model parameters
PacejkaParamsList = ['Sx_p', 'Alpha_p', 'By', 'Cy', 'Dy', 'Ey', 'Bx', 'Cx', 'Dx', 'Ex']#'Svy', 'Svx', 'Shy', 'Shx']
PacejkaParams = collections.namedtuple('PacejkaParams', PacejkaParamsList, defaults=[] * len(PacejkaParamsList))

class PacejkaParameters(torch.nn.Module):
    def __init__(self) -> None:
        super(PacejkaParameters, self).__init__()
        self.param_count = len(PacejkaParamsList)
        logging.info(f'PacejkaParameters:__init__:self.param_count:{self.param_count}')

    def forward(self, p):
        # Extract tire parameters from input tensor
        # p_this_layer = p[..., :self.param_count]

        # Create a namedtuple of tire parameters
        named_tuple = PacejkaParams(
            Sx_p=p[..., 0],
            Alpha_p=p[..., 1],
            By=p[..., 2],
            Cy=p[..., 3],
            Dy=p[..., 4],
            Ey=p[..., 5],
            Bx=p[..., 6],
            Cx=p[..., 7],
            Dx=p[..., 8],
            Ex=p[..., 9],
            # # offsets
            # Svy=p_this_layer[..., 10],
            # Svx=p_this_layer[..., 11],
            # Shy=p_this_layer[..., 12],
            # Shx=p_this_layer[..., 13]
        )
        
        return named_tuple

    @staticmethod
    def default_params_tensor(batch_size=1):
        return torch.tensor([
            0.05,  # Sx_p: Lower peak slip ratio indicating higher sensitivity to slip
            2.0,   # Alpha_p: Lower peak slip angle indicating sharper lateral response
            0.35,  # By: Higher stiffness factor for lateral force
            1.4,   # Cy: Higher shape factor for a sharper peak
            1.0,   # Dy: Slightly higher peak value for lateral force
            1.2,  # Ey: Adjusted curvature factor for a more abrupt change in forces
            30.0,  # Bx: Higher stiffness factor for longitudinal force
            1.3,   # Cx: Higher shape factor for a more pronounced peak
            1.0,   # Dx: Slightly higher peak value for longitudinal force
            0.5,   # Ex: Adjusted curvature factor for sharper longitudinal force changes,
            
            # 0.0,  # Svy: Lateral force offset
            # 0.0,  # Svx: Longitudinal force offset
            # 0.0,  # Shy: Lateral force offset
            # 0.0   # Shx: Longitudinal force offset
        ]).unsqueeze(0).repeat(batch_size, 1)

    @staticmethod
    def get_params_names():
        return PacejkaParamsList