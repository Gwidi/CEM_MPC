import torch
from typing import Dict
from collections import namedtuple

STATE_DEF_LIST = [
    "x",
    "y",
    "yaw",
    "v_x",
    "v_y",
    "r",
    "omega_wheels",
    "omega_wheels_ref",
    "delta",
    "friction",
    "delta_ref",
    "omega_wheels_ref_dot",
]

STATE_DEF_LIST_SHORT = STATE_DEF_LIST[:10]

VEH_PARAM_LIST = [
    "m",
    "I_z",
    "lf",
    "lr",
    "R",
    "I_e",
    "K_fi",
    "b0",
    "b1",
    "Cd0",
    "Cd1",
    "Cd2",
    "g",
]

TIRE_PARAM_LIST = [
    "B_f",
    "C_f",
    "D_f",
    "B_r",
    "C_r",
    "D_r",
    "B_long",
    "C_long",
    "mu_tire",
]

WX = namedtuple("WX", STATE_DEF_LIST)

VEH_PARAMS = namedtuple("VEH_PARAMS", VEH_PARAM_LIST)

TIRE_PARAMS = namedtuple("TIRE_PARAMS", TIRE_PARAM_LIST)


@torch.jit.script
def StateWrapper(state):
    return WX(
        x=state[..., 0],
        y=state[..., 1],
        yaw=state[..., 2],
        v_x=state[..., 3],
        v_y=state[..., 4],
        r=state[..., 5],
        omega_wheels=state[..., 6],
        omega_wheels_ref=state[..., 7],
        delta=state[..., 8],
        friction=state[..., 9],
        delta_ref=state[..., 10],
        omega_wheels_ref_dot=state[..., 11],
    )


@torch.jit.script
def ParamWrapper(p):
    return VEH_PARAMS(
        m=p[..., 0],
        I_z=p[..., 1],
        lf=p[..., 2],
        lr=p[..., 3],
        R=p[..., 4],
        I_e=p[..., 5],
        K_fi=p[..., 6],
        b0=p[..., 7],
        b1=p[..., 8],
        Cd0=p[..., 9],
        Cd1=p[..., 10],
        Cd2=p[..., 11],
        g=p[..., 12],
    )


@torch.jit.script
def TireWrapper(p):
    return TIRE_PARAMS(
        B_f=p[..., 0],
        C_f=p[..., 1],
        D_f=p[..., 2],
        B_r=p[..., 3],
        C_r=p[..., 4],
        D_r=p[..., 5],
        B_long=p[..., 6],
        C_long=p[..., 7],
        mu_tire=p[..., 8],
    )
