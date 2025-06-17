# -*- coding: utf-8 -*-
"""
Created on Fri May 30 17:07:38 2025

@author: UseV
"""

import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import colorednoise as cn
mu=None
mus=[]
stds=[]
# xglob
# yglob

# --- Model Pojazdu ---

class BicycleModelCurvilinear:
    def __init__(self):
        self.m = 230         # masa [kg]
        self.Iz = 75    # bezwładność [kg m^2]
        self.lF = 0.9
        self.lR = 1.0
        self.L = self.lF + self.lR
        self.Cm = 0.3 * self.m
        self.Cr0 = 50
        self.Cr2 = 0.5

        self.Bf= 1.4928231239318848
        self.Cf= 2.1563267707824707
        self.Df= 0.015784228920936584
        self.Br= 10.573360443115234
        self.Cr= 1.0724537372589111
        self.Dr= 0.011050906181335


        self.ptv = 3.0
    def tire_forces_test(self,alpha):
        alpha=5*alpha
        F = self.Df * np.sin(self.Cf * np.arctan(self.Bf * alpha))
        return F
    def tire_forces(self, vx, vy, r, delta):
        alpha_f = np.arctan2(vy + self.lF * r, vx) - delta
        alpha_r = np.arctan2(vy - self.lR * r, vx)
        alpha_f=5*alpha_f
        alpha_r=5*alpha_r
        N=self.m*9.81/2
        FyF = self.Df * np.sin(self.Cf * np.arctan(self.Bf * alpha_f))*N
        FyR = self.Dr * np.sin(self.Cr * np.arctan(self.Br * alpha_r))*N
        return FyF, FyR

    def longitudinal_force(self, vx, T):
        return self.Cm * T - self.Cr0 - self.Cr2 * vx ** 2

    def tv_moment(self, vx, r, delta):
        rt = np.tan(delta) * vx / self.L
        return self.ptv * (rt - r)

    def dynamics(self, x, u, curvature):
        s, n, mu, vx, vy, r, delta, T = x
        delta_dot, T_dot = u
        FyF, FyR = self.tire_forces(vx, vy, r, delta)
        Fx = self.longitudinal_force(vx, T)
        Mtv = self.tv_moment(vx, r, delta)
        s_dot = (vx * np.cos(mu) - vy * np.sin(mu)) / (1 - curvature * n)
        n_dot = vx * np.sin(mu) + vy * np.cos(mu)
        mu_dot = r - curvature * s_dot
        vx_dot = (Fx - FyF * np.sin(delta) + self.m * vy * r) / self.m
        vy_dot = (FyR + FyF * np.cos(delta) - self.m * vx * r) / self.m
        r_dot = (FyF * self.lF * np.cos(delta) - FyR * self.lR + Mtv) / self.Iz

        # (v_x*np.sin(mu)+v_y*np.cos(mu))*dt
        # (v_x*np.cos(mu)+v_y*np.sin(mu))*dt
        return np.array([s_dot, n_dot, mu_dot, vx_dot, vy_dot, r_dot, delta_dot, T_dot])

    def rk4_step(self, x, u, curvature, dt):
        k1 = self.dynamics(x, u, curvature)
        k2 = self.dynamics(x + 0.5 * dt * k1, u, curvature)
        k3 = self.dynamics(x + 0.5 * dt * k2, u, curvature)
        k4 = self.dynamics(x + dt * k3, u, curvature)
        x_new = x + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
        # OGRANICZENIE T (napędu)
        # OGRANICZENIE delta (kąt skrętu), opcjonalnie:
        # x_new[6] = np.clip(x_new[6], -np.pi/4, np.pi/4)
        x_new[3] = np.clip(x_new[3], 5.0, np.inf)
        return x_new

# --- Generator Trasy ---
def generate_test_track(num_points=10000):
    s = np.linspace(0, 320, num_points)
    #curvature = 0.05 * np.sin(0.1 * s)
    curvature=np.ones(num_points)*0.02*0.0
    return s, curvature
def j_LTO(x, u, dt, R, model, curvature,q_beta=0.1,qn=10,qmu=50):
    """
    Evaluate the j_LTO function.

    Parameters:
    - x_k: np.array, the state vector at step k
    - u_k: np.array, the control vector at step k
    - delta_s: float, the step size Δs
    - s_dot_k: float, the speed at step k (dot{s}_k)
    - R: np.array, weight matrix for the control vector
    - B_func: function, computes B(x_k)

    Returns:
    - j_lto: float, the evaluated cost
    """

    term4=qn*x[1]**2
    term5=qmu*x[2]**2
    s_dot = (x[3] * np.cos(x[2]) - x[4] * np.sin(x[2])) / (1 - curvature * x[1])
    term1 = -dt * s_dot
    term2 = u.T @ R @ u
    beta_dyn = np.arctan(x[4] / x[3])
        
    beta_kin = np.arctan((x[6] * model.lR) / (model.lF + model.lR))
    
    term3 = q_beta * (beta_dyn - beta_kin) ** 2
    

    return term1 + term2 + term3 + term4 + term5
# --- Cross-Entropy MPC ---
def cem_control(model, x0, track_s, curvature, dt=0.05,
                samples=64, elite_frac=0.2,iterations=2, dTupper=1.0,dTlower=-1.0, ddeltaupper=1.0, ddeltalower=-1.0,horizon=20):
    beta = 1 # the exponent
    global mu
    global std
    if mu is None:
        mu=np.ones((horizon,2))*np.mean([[ddeltalower,ddeltaupper],[dTlower,dTupper]],axis=1)
    std=np.ones((horizon,2))*[abs(ddeltalower-ddeltaupper)/2,abs(dTlower-dTupper)/2]
    R=np.eye(2)*0
    for i in range(iterations):
        actions = cn.powerlaw_psd_gaussian(beta, (samples,horizon,2))
        actions=actions*std+mu
        
        actions[:,:,0]=np.clip(actions[:,:,0],ddeltalower,ddeltaupper)
        actions[:,:,1]=np.clip(actions[:,:,1],dTlower,dTupper)
        #actions[0,:,0]=actions[0,:,0]*0
        costs =[]
        for s in actions:
            cost=0 
            x=x0
            traj=[]
            
            for h in s:
                idx = np.argmin(np.abs(track_s - x[0]%max(track_s)))
                curv = curvature[idx]
                traj.append(model.rk4_step(x,h,curv,dt))
            
            for x,y in zip(traj,s):
                
                cost+=j_LTO(x,y,dt,R,model,curv)
            costs.append(cost)
        # for x in actions:
        #     idx = np.argmin(np.abs(track_s - x0[0]%max(track_s)))
        #     curv = curvature[idx]
        #     nextstates.append(model.rk4_step(x0,x,curv,dt))
        # for x,y in zip(nextstates,actions):
        #     costs.append(j_LTO(x,y,dt,np.eye(len(y)),model,curv))
        eliteuntruncated=[x for (_,x) in sorted(zip(costs, actions), key=lambda pair: pair[0])]
        elitetruncated=eliteuntruncated[0:round(elite_frac*len(eliteuntruncated))]
        
        mu=np.mean(elitetruncated,axis=0)
        mus.append(mu[0][0])
        std=np.std(elitetruncated,axis=0)
        stds.append(std[0][0])
    return elitetruncated[0]

# --- Środowisko ---
class RaceCarEnv(gym.Env):
    def __init__(self, model, track_s, curvature, dt=0.05, max_steps=200):
        super(RaceCarEnv, self).__init__()

        self.model = model
        self.track_s = track_s
        self.curvature = curvature
        self.dt = dt
        self.max_steps = max_steps

        high_obs = np.array([track_s[-1], 5.0, np.pi, 30.0, 10.0, 5.0, np.pi/4, 1.0])
        self.observation_space = spaces.Box(low=-high_obs, high=high_obs, dtype=np.float32)

        self.L = 1.8
        self.W = 1.0

        self._init_render()
        self.reset()

    def reset(self):
        self.state = np.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0])
        self.step_count = 0
        return self.state.copy()

    def step(self, action):
        for a in action:    
            delta_dot, T_dot = a
            u = np.array([delta_dot, T_dot])
    
            idx = np.argmin(np.abs(self.track_s - self.state[0]))
            curv = self.curvature[idx%len(curvature)]
            next_state = self.model.rk4_step(self.state, u, curv, self.dt)
            self.state = next_state
            env.render()

        return self.state.copy()

    def _init_render(self):
        self.fig, self.ax = plt.subplots()
        self.car_patch = Rectangle((0, 0), self.L, self.W, color='blue', alpha=0.8)
        self.ax.add_patch(self.car_patch)
        self.ax.set_aspect('equal')
        self.track_x, self.track_y = self._compute_reference_track()
        self.ax.set_xlim(-10+min(self.track_x), max(self.track_x)+10)
        self.ax.set_ylim(-10+min(self.track_y), max(self.track_y)+10)
        self.track_x, self.track_y = self._compute_reference_track()
        self.ax.plot(self.track_x, self.track_y, 'k--')

    def _compute_reference_track(self):
        ds = self.track_s[1] - self.track_s[0]
        theta = np.cumsum(self.curvature) * ds
        x = np.cumsum(np.cos(theta) * ds)
        y = np.cumsum(np.sin(theta) * ds)
        return x, y

    def render(self, mode='human'):
        s, n, mu = self.state[0], self.state[1], self.state[2]
        idx = np.argmin(np.abs(self.track_s - s))
        ref_x, ref_y = self.track_x[idx], self.track_y[idx]
        ref_theta = np.sum(self.curvature[:idx]) * (self.track_s[1] - self.track_s[0])
        x_car = ref_x - n * np.sin(ref_theta)
        y_car = ref_y + n * np.cos(ref_theta)
        angle = np.rad2deg(mu + ref_theta)
        self.car_patch.set_xy((x_car - self.L / 2, y_car - self.W / 2))
        self.car_patch.angle = angle
        self.step_count+=1
        self.ax.set_title(f"Step {self.step_count}")
        plt.pause(0.01)

    def close(self):
        plt.show(block=True)
        plt.close(self.fig)
    
def pure_straight_test(model, dt=0.05, steps=200, T_init=0.5, v0=1.0):
    """
    Test modelu na prostej drodze.
    Model dostaje tylko napęd, bez skrętu, bez toru (krzywizna=0).
    """
    # Stan początkowy: [s, n, mu, vx, vy, r, delta, T]
    x = np.array([0, 0, 0, v0, 0, 0, 0.4, T_init])
    traj = [x.copy()]
    u = np.array([0.0, 0.0])  # delta_dot = 0, T_dot = 0
    t=[0]
    for _ in range(steps):
        curvature = 0.0  # prosta
        x = model.rk4_step(x, u, curvature, dt)
        traj.append(x.copy())
        t.append(t[len(t)-1]+dt)
    traj = np.array(traj)
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,6))
    plt.subplot(2,2,1)
    plt.plot(t, traj[:,1])
    plt.xlabel("t [s]"); plt.ylabel("n [m]"); plt.title("Odchylenie od toru (n)")

    plt.subplot(2,2,2)
    plt.plot(t, traj[:,2])
    plt.xlabel("t [s]"); plt.ylabel("mu [rad]"); plt.title("Kąt względem toru (mu)")
    
    plt.subplot(2,2,3)
    plt.plot(t, traj[:,3], label="vx")
    plt.plot(t, traj[:,4], label="vy")
    plt.xlabel("t [s]"); plt.ylabel("Prędkości [m/s]"); plt.legend(); plt.title("vx, vy")

    plt.subplot(2,2,4)
    plt.plot(traj[:,0],traj[:,1])
    plt.xlabel("t [s]"); plt.ylabel("x"); plt.title("y")
    plt.plot(t, traj[:,6], label="delta")
    plt.xlabel("t [s]"); plt.ylabel("delta [rad]"); plt.title("Kąt skrętu (delta)")

    plt.tight_layout()
    plt.show()





model = BicycleModelCurvilinear()
track_s, curvature = generate_test_track()
env = RaceCarEnv(model, track_s, curvature)

#pure_straight_test(model, dt=0.05, steps=1000, T_init=0.5, v0=5.0)
# Fplot=[]
# a=16
# for x in np.linspace(0,a/180*np.pi,100000):
#     Fplot.append(model.tire_forces_test(x))
# plt.figure()
# plt.plot(np.linspace(0,a/180*np.pi,100000),Fplot)


obs = env.reset()
import cv2
keyp=ord('n')
for _ in range(200):
    keyp=cv2.waitKey()
    if keyp!=ord('q'):
        action = cem_control(model, obs, track_s, curvature)
        obs = env.step(action[0:2])
        print("T aktualne:", obs[7])
    else:
        break

    
env.close()
mu=np.array(mu)
std=np.array(std)
plt.plot(mu,'b')
plt.plot(mu-std,'r')
plt.plot(mu+std,'r')
