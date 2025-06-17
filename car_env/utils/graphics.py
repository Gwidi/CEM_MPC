import math
import pygame
import numpy as np
import torch
import time
import logging
from matplotlib import colormaps as cm
import matplotlib.pyplot as plt

from car_env.utils.state_wrapper import StateWrapper, ParamWrapper
from car_env.utils.utils import Track


def rect(x, y, angle, w, h, scale=10):
    x = x * scale
    y = y * scale
    w = w * scale
    h = h * scale

    return [
        translate(x, y, angle, -w / 2, h / 2),
        translate(x, y, angle, w / 2, h / 2),
        translate(x, y, angle, w / 2, -h / 2),
        translate(x, y, angle, -w / 2, -h / 2),
    ]


def translate(x, y, angle, px, py):
    x1 = x + px * math.cos(angle) - py * math.sin(angle)
    y1 = y + px * math.sin(angle) + py * math.cos(angle)
    return [int(x1), int(y1)]


def tensor_to_rdylgn(normalized_tensor):
    normalized_numpy = normalized_tensor.cpu().numpy()
    colormap = cm['RdYlGn']
    colored_np = colormap(normalized_numpy)[..., :3]
    return colored_np


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 100, 0)
RED = (255, 0, 0)
GREY = (200, 200, 200)


class SceneRenderer:
    def __init__(
        self,
        vehicle_params: torch.Tensor,
        track_x,
        track_y,
        track_width,
        scale: float = 80 / 1,  # px per meter
        dt: float = 0.02,
        friction_map=None,
    ):
        self.vehicle_params = vehicle_params
        self.track = Track(
            s=None,
            x=track_x,
            y=track_y,
            width=track_width,
            curvature=None,
            heading=None,
        )
        self.scale = scale
        self.clock = pygame.time.Clock()

        self.friction_map = friction_map[0]

        self.dt = dt
        # log now tim
        self.last_render = time.time()

        self.min_friction = 0.65
        self.max_friction = 0.95

        self.track_min_x = np.min(self.track.x).item()
        self.track_max_x = np.max(self.track.x).item()
        self.track_min_y = np.min(self.track.y).item()
        self.track_max_y = np.max(self.track.y).item()

        self.screen_width = int((self.track_max_x - (self.track_min_x) + 2) * self.scale)
        self.screen_height = int((self.track_max_y - (self.track_min_y) + 2) * self.scale)

        # add pading to the track
        self.track_min_x -= 1
        self.track_min_y -= 1

        self._track_surface = None
        self._rendering_started = False

        logging.info("SceneRenderer initalized")

    def start_render(self):
        self._rendering_started = True
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.scale = 80 / 1  # px per meter
        self._cache_track()
        logging.info("Rendering started")

    def render(self, idxs: list, closest_idx: int, state: torch.tensor,
               trajectories: torch.tensor = None, costs: torch.tensor = None):
        if not self._rendering_started:
            self.start_render()
        self.draw_track()
        self.draw_grid()
        if trajectories is not None:
            self.draw_trajectories(trajectories, costs)

        for idx in idxs:
            self.render_car(idx, closest_idx[idx], state[idx])

        now = time.time()
        diff = now - self.last_render
        offset = self.dt - diff
        wait = max(0, offset)
        if offset < 0: 
            logging.error(f"Rendering took too long: {diff}")
        time.sleep(wait)
        self.last_render = time.time()


        # flip the screen
        flipped_screen = pygame.transform.flip(self.screen, False, True)
        self.screen.blit(flipped_screen, (0, 0))

        self.draw_text()

        pygame.display.flip()
        self.clock.tick(0)

    def _cache_track(self):
        self._track_surface = pygame.Surface((self.screen_width, self.screen_height))
        self._track_surface.fill(WHITE)

        color_scaler = 255 / 1.25

        color_map = [GREY] * len(self.track.x)

        if self.friction_map is not None:
            self.min_friction = round(min(self.friction_map[:self.track.x.shape[0]]), 2)
            self.max_friction = round(max(self.friction_map[:self.track.x.shape[0]]), 2)
            range_friction = self.max_friction - self.min_friction
            for i in range(len(self.track.x)):
                friction = self.friction_map[i]
                color = int((friction - self.min_friction) / range_friction * color_scaler)
                color = max(0, min(color, 255))
                color_map[i] = (
                    color,
                    color,
                    color,
                )
                
        for i in range(len(self.track.x) - 1):
            x = self.track.x[i] - (self.track_min_x)
            y = self.track.y[i] - (self.track_min_y)
            w = self.track.width[i]

            pygame.draw.circle(
                self._track_surface,
                color_map[i],
                (int(x * self.scale), int(y * self.scale)),
                int(w * self.scale) / 2,  # radius is width / 2
                0,
            )

        for i in range(len(self.track.x) - 1):
            x = self.track.x[i] - (self.track_min_x)
            y = self.track.y[i] - (self.track_min_y)

            if i % 10 == 0:
                # write index of the point
                font = pygame.font.Font(None, 20)
                text = font.render(str(int(i / 10)), True, BLACK)
                self._track_surface.blit(
                    text, (int(x * self.scale), int(y * self.scale))
                )

        # add color legend at the bottom
        for i in range(256):
            color = int(i / 1.25)
            pygame.draw.rect(
                self._track_surface,
                (color, color, color),
                (i * 2, self.screen_height - 20, 2, 20),
            )

    def draw_track(self):
        if self._track_surface is None:
            self._cache_track()
        self.screen.blit(self._track_surface, (0, 0))

    def draw_grid(self):
        # draw grid 1m x 1m
        x_lines = np.arange(0, self.screen_width, 1 * self.scale)
        y_lines = np.arange(0, self.screen_height, 1 * self.scale)

        for x in x_lines:
            pygame.draw.line(self.screen, BLACK, (x, 0), (x, self.screen_height), 1)

        for y in y_lines:
            pygame.draw.line(self.screen, BLACK, (0, y), (self.screen_width, y), 1)

    def draw_trajectories(self, trajectories, costs=None):
        trajectories = trajectories.detach().numpy()
        colors = None
        if costs is not None:
            rewards = -costs
            rng = (-100., 100.)
            rewards = torch.clip(rewards, rng[0], rng[1])
            rewards_normalized = (rewards - rng[0]) / (rng[1] - rng[0])
            colors = 255 * tensor_to_rdylgn(rewards_normalized)
        for i in range(len(trajectories)):
            #pygame.draw.lines(self.screen, RED, False, trajectories[i], 2)
            traj = [((x[0].item() - self.track_min_x) * self.scale, (x[1].item() - self.track_min_y) * self.scale)
                    for x in trajectories[i]]
            pygame.draw.lines(self.screen, colors[i] if colors is not None else RED, False, traj, 2)

    def render_car(self, idx, closest_idx, state):
        rendered_state = StateWrapper(state)
        x = rendered_state.x - (self.track_min_x)
        y = rendered_state.y - (self.track_min_y)
        yaw = rendered_state.yaw
        delta = rendered_state.delta

        rendered_params = ParamWrapper(self.vehicle_params)
        Lf = rendered_params.lr[idx]
        Lr = rendered_params.lf[idx]
        length = Lf + Lr
        w = length / 8

        body = rect(x, y, yaw, length, w, self.scale)
        front_wheel = rect(
            x + Lf * math.cos(yaw),
            y + Lf * math.sin(yaw),
            yaw + delta,
            length / 4,
            w,
            self.scale,
        )
        rear_wheel = rect(
            x - Lr * math.cos(yaw),
            y - Lr * math.sin(yaw),
            yaw,
            length / 4,
            w,
            self.scale,
        )
        center = rect(x, y, yaw, w, w, self.scale)

        pygame.draw.polygon(self.screen, BLACK, body)
        pygame.draw.polygon(self.screen, GREEN, front_wheel)
        pygame.draw.polygon(self.screen, RED, rear_wheel)
        pygame.draw.polygon(self.screen, BLUE, center)

        x_point = self.track.x[closest_idx] - (self.track_min_x)
        y_point = self.track.y[closest_idx] - (self.track_min_y)
        pygame.draw.circle(
            self.screen,
            RED,
            (int(x_point * self.scale), int(y_point * self.scale)),
            5,
            0,
        )

        pygame.draw.line(
            self.screen,
            BLACK,
            (int(x * self.scale), int(y * self.scale)),
            (int(x_point * self.scale), int(y_point * self.scale)),
            2,
        )

        # render every 5th track point from closest_idx
        for i in range(0, 70, 10):
            ponint_idx = (closest_idx + i) % len(self.track.x)
            x_point = self.track.x[ponint_idx] - (self.track_min_x)
            y_point = self.track.y[ponint_idx] - (self.track_min_y)
            pygame.draw.circle(
                self.screen,
                BLUE,
                (int(x_point * self.scale), int(y_point * self.scale)),
                2,
                0,
            )

    def draw_text(self):
        y_offset = 30

        font = pygame.font.Font(None, 20)
        text = font.render(str(self.min_friction), True, BLACK)
        self.screen.blit(text, (5, y_offset))

        text = font.render(str(self.max_friction), True, BLACK)
        self.screen.blit(text, (self.screen_width-25, y_offset))

        text = font.render(str(round((self.max_friction + self.min_friction) / 2, 2)), True, BLACK)
        self.screen.blit(text, (self.screen_width // 2 - 15, y_offset))

    def set_frcition_map(self, friction_map):
        self.friction_map = friction_map[0]
        # plot with matplotlib
        plt.plot(self.friction_map)
        plt.show()

        self._cache_track()

    def close(self):
        pygame.quit()
