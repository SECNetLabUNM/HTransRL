"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
from typing import Optional, Union

import gymnasium as gym
import numpy as np
from air_corridor.d2.geometry.geom import directionalPartialAnnulus
from air_corridor.tools.uti_consts import *
from air_corridor.tools.util import is_line_line_intersect
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400
WORLD_WIDTH = (20 + 2) * 2 * 2
SCALE = SCREEN_WIDTH / WORLD_WIDTH


class AnnulusMoveV1(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    v1 use the same radian for rl and plot.
    simplified env can benefit rl, but much difficult for coding
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 3,
    }

    def __init__(self,
                 render_mode: Optional[str] = None,
                 task_complexity: str = 's',
                 discrete_action_options: int = 4,
                 state_mode: str = 'mode1',
                 reward_mode: str = 'mode1'):

        self.state_mode = state_mode
        self.reward_mode = reward_mode
        self.task_complexity = task_complexity

        self.end_rad_vec = None

        self.rad_begin_to_end = None
        self.rad_options = 6

        self.env_end_rad = None
        self.env_begin_rad = None
        self.beg_point = None
        self.train_progress = None

        self.rad_diff_all_steps = 0
        self.major_radius = 6
        self.minor_radius = 2

        self.maxspeed = 0.6
        self.stepstaken = 0
        self.stepsmax = None
        # first one for direction, second for speed ratio to maximum
        self.outside_counter = 0

        # if self.continuous:
        #     self.action_space=spaces.Box(
        #         np.array([-1, 0]).astype(np.float32),
        #         np.array([+1, +1]).astype(np.float32),
        #     )  # direction mapping to -pi,pi
        #         # acceleartion utilization
        # else:
        self.discrete_action_options = discrete_action_options
        self.action_space = spaces.Discrete(self.discrete_action_options)

        self.observation_space = spaces.Box(low=-20, high=20, shape=(5,), dtype=np.float32)

        # Angle at which to fail the episode

        self.position = None
        self.begin_rad = None
        self.end_rad = None
        self.end_point = None
        self.counter_clockwise = None
        self.end_diff = None

        self.corridor = directionalPartialAnnulus()
        self.render_mode = render_mode

        self.screen = None
        self.clock = None
        self.isopen = True

        self.UAV_size = 10.0
        self.offset_x = SCREEN_WIDTH / 2
        self.offset_y = SCREEN_HEIGHT / 2

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        # assert self.state is not None, K"Call reset before using step method."

        # first action goes right at rad 0
        action_rad = 2 * np.pi * action / self.discrete_action_options
        x_diff = math.cos(action_rad + self.rad_diff_all_steps + self.begin_rad) * self.maxspeed
        y_diff = math.sin(action_rad + self.rad_diff_all_steps + self.begin_rad) * self.maxspeed
        position_diff_one_step = np.array([x_diff, y_diff])

        # enter new position

        next_position = self.position + position_diff_one_step
        rad_diff_one_step = self.corridor.line_to_proj_rad(self.position, next_position)
        self.stepstaken += 1
        terminated = False

        self.rad_diff_all_steps += rad_diff_one_step

        progress_percent = self.rad_diff_all_steps / self.rad_begin_to_end

        if self.corridor.is_inside(next_position):
            if self.reward_mode == 'mode1':
                reward = rad_diff_one_step * self.major_radius
            elif self.reward_mode == 'mode2':
                reward = 0
        else:
            if is_line_line_intersect(next_position,
                                      self.position,
                                      self.end_point - self.end_rad_vec,
                                      self.end_point + self.end_rad_vec):
                reward = REWARD_REACH
                terminated = True
            else:
                reward = PENALTY_BREACH
                self.outside_counter += 1

        if self.outside_counter >= 2:
            terminated = True
        else:
            reward += PENALTY_TIME
        truncated = True if self.stepstaken == self.stepsmax else False

        self.position = next_position

        # visualize
        if self.render_mode == "human":
            self.render()

        if self.state_mode == 'mode1':
            state = self.position.tolist() + [self.major_radius, self.begin_rad, self.end_rad]
        elif self.state_mode == 'mode2':
            current_r = np.linalg.norm(self.position)
            state = [current_r - (self.major_radius - self.minor_radius),
                     self.major_radius + self.minor_radius - current_r, self.major_radius,
                     self.end_rad - self.rad_diff_all_step]

        return np.array(state, dtype=np.float32), reward, terminated, truncated, \
            {'stepstaken': self.stepstaken,
             'position': self.position,
             'progress': progress_percent}

    def reset(
            self,
            train_progress=0,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):

        self.train_progress = train_progress
        "in early stage, the range of rad is smaller"

        '''
        simplify environment:
        1) radian range between begin and end is [1,2,3 * np.pi/2]
        2) rl training only consider self.counter_clockwise=1, mirroring over y axis
        3) rl training always begins with 0, by rotating environment radian to 0
        '''
        self.counter_clockwise = 1 if np.random.randint(2) == 0 else -1
        self.begin_rad = 2 * np.pi * (np.random.randint(self.rad_options) / self.rad_options) - np.pi

        self.rad_begin_to_end = (np.pi * 1.5) * np.random.random()
        self.end_rad = self.begin_rad + self.counter_clockwise * self.rad_begin_to_end

        self.major_radius = np.random.random() * 14 + 6

        if self.counter_clockwise == 1:
            self.draw_bgein = int(math.degrees(self.begin_rad))
            self.draw_end = int(math.degrees(self.end_rad))
        elif self.counter_clockwise == -1:
            self.draw_bgein = int(math.degrees(self.end_rad))
            self.draw_end = int(math.degrees(self.begin_rad))

        self.rad_diff_all_steps = 0

        self.stepstaken = 0
        self.stepsmax = int(math.ceil(self.major_radius * 1.5 * np.pi / self.maxspeed * 2 / 100) * 100)
        self.outside_counter = 0

        self.position = np.array([np.cos(self.begin_rad) * self.major_radius,
                                  np.sin(self.begin_rad) * self.major_radius])

        # used for finishing determination

        self.begin_point = np.array([np.cos(self.begin_rad) * self.major_radius,
                                     np.sin(self.begin_rad) * self.major_radius])
        self.begin_rad_vec = np.array([np.cos(self.begin_rad) * self.minor_radius,
                                       np.sin(self.begin_rad) * self.minor_radius])

        self.end_point = np.array([np.cos(self.end_rad) * self.major_radius,
                                   np.sin(self.end_rad) * self.major_radius])
        self.end_rad_vec = np.array([np.cos(self.end_rad) * self.minor_radius,
                                     np.sin(self.end_rad) * self.minor_radius])

        self.corridor = directionalPartialAnnulus(np.array([0, 0]),
                                                  self.major_radius,
                                                  self.minor_radius,
                                                  self.begin_rad,
                                                  self.end_rad)
        super().reset(seed=seed)
        # state = [self.major_radius * np.cos(self.begin_rad),
        #          self.major_radius * np.sin(self.begin_rad)] + self.corridor.report_state()

        if self.state_mode == 'mode1':
            state = self.position.tolist() + [self.major_radius, self.begin_rad, self.end_rad]
        elif self.state_mode == 'mode2':
            current_r = np.linalg.norm(self.position)
            state = [current_r - (self.major_radius - self.minor_radius),
                     self.major_radius + self.minor_radius - current_r, self.major_radius,
                     self.end_rad - self.rad_diff_all_step]
        if self.render_mode == "human":
            self.render()
        return np.array(state, dtype=np.float32), {}

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (SCREEN_WIDTH, SCREEN_HEIGHT)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.surf.fill((255, 255, 255))

        x = self.position[0]
        y = self.position[1]

        gfxdraw.filled_circle(self.surf,
                              int(self.offset_x + x * SCALE),
                              int(self.offset_y + y * SCALE),
                              int(self.UAV_size / 2),
                              (129, 132, 203),
                              )

        gray = (10, 10, 10)

        gfxdraw.arc(self.surf,
                    int(self.offset_x),
                    int(self.offset_y),
                    int((self.major_radius + self.minor_radius) * SCALE),
                    self.draw_bgein,
                    self.draw_end,
                    gray)

        gfxdraw.arc(self.surf,
                    int(self.offset_x),
                    int(self.offset_y),
                    int((self.major_radius - self.minor_radius) * SCALE),
                    self.draw_bgein,
                    self.draw_end,
                    gray)

        # draw begin line
        gfxdraw.line(self.surf,
                     int(self.offset_x + (self.begin_point[0] + self.begin_rad_vec[0]) * SCALE),
                     int(self.offset_y + (self.begin_point[1] + self.begin_rad_vec[1]) * SCALE),
                     int(self.offset_x + (self.begin_point[0] - self.begin_rad_vec[0]) * SCALE),
                     int(self.offset_y + (self.begin_point[1] - self.begin_rad_vec[1]) * SCALE),
                     (0, 0, 255))

        # draw destination line
        gfxdraw.line(self.surf,
                     int(self.offset_x + (self.end_point[0] + self.end_rad_vec[0]) * SCALE),
                     int(self.offset_y + (self.end_point[1] + self.end_rad_vec[1]) * SCALE),
                     int(self.offset_x + (self.end_point[0] - self.end_rad_vec[0]) * SCALE),
                     int(self.offset_y + (self.end_point[1] - self.end_rad_vec[1]) * SCALE),
                     (255, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
