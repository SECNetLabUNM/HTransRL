"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
from typing import Optional, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled

from air_corridor.d2.geometry.geom import directionalPartialAnnulus
from air_corridor.tools.util import is_line_line_intersect

from air_corridor.tools.uti_consts import *


class AnnulusMove(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    ## Description

    This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
    ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
    The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
     in the left and right direction on the cart.

    ## Action Space

    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
     of the fixed force the cart is pushed with.

    - 0: Push cart to the left
    - 1: Push cart to the right

    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
     the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

    ## Observation Space

    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |

    **Note:** While the ranges above denote the possible values for observation space of each element,
        it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
       if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
       if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

    ## Rewards

    Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
    including the termination step, is allotted. The threshold for rewards is 475 for v1.

    ## Starting State

    All observations are assigned a uniformly random value in `(-0.05, 0.05)`

    ## Episode End

    The episode ends if any one of the following occurs:

    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500 (200 for v0)

    ## Arguments

    ```python
    import gymnasium as gym
    gym.make('CartPole-v1')
    ```

    On reset, the `options` parameter allows the user to change the bounds used to determine
    the new random state.
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
        # self.major_radius = np.random.random() * 15 + 5  # [5,20]
        # self.minor_radius = np.random.random() * self.major_radius / 2
        self.state_mode = state_mode
        self.reward_mode = reward_mode
        self.task_complexity = task_complexity
        self.draw_end_rad_vec = None
        self.draw_end_point = None
        self.end_rad_vec = None
        self.draw_begin_point = None
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

        self.observation_space = spaces.Box(low=-20, high=20, shape=(4,), dtype=np.float32)

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
        x_diff = math.cos(action_rad + self.rad_diff_all_steps) * self.maxspeed
        y_diff = math.sin(action_rad + self.rad_diff_all_steps) * self.maxspeed
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
            state = self.position.tolist() + [self.major_radius, self.end_rad]
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
        self.env_begin_rad = 2 * np.pi * (np.random.randint(self.rad_options) / self.rad_options) - np.pi

        # if self.task_complexity == 's':
        #     self.rad_begin_to_end = (np.pi / 2)
        # elif self.task_complexity == 'c':
        #     self.rad_begin_to_end = (np.pi / 2) * (np.random.randint(3) + 1)

        self.rad_begin_to_end = np.pi * (np.random.random()+0.5)
        self.env_end_rad = self.env_begin_rad + self.counter_clockwise * self.rad_begin_to_end
        self.begin_rad = 0
        self.end_rad = self.begin_rad + self.rad_begin_to_end
        self.major_radius = np.random.random() * 14 + 6

        # if self.counter_clockwise == 1:
        #     self.begin_rad = self.env_begin_rad
        #     self.end_rad = self.env_end_rad
        # elif self.counter_clockwise == -1:
        #     if self.env_begin_rad >= 0:
        #         self.begin_rad = np.pi - self.env_begin_rad
        #     else:
        #         self.begin_rad = -np.pi - self.env_begin_rad
        #     self.end_rad = self.begin_rad + np.pi / 2

        self.rad_diff_all_steps = 0

        self.stepstaken = 0
        self.stepsmax = int(math.ceil(self.major_radius * 1.5 * np.pi / self.maxspeed * 2 / 100) * 100)
        self.outside_counter = 0

        self.position = np.array([np.cos(self.begin_rad) * self.major_radius,
                                  np.sin(self.begin_rad) * self.major_radius])

        # used for finishing determination
        self.end_point = np.array([np.cos(self.end_rad) * self.major_radius,
                                   np.sin(self.end_rad) * self.major_radius])
        self.end_rad_vec = np.array([np.cos(self.end_rad) * self.minor_radius,
                                     np.sin(self.end_rad) * self.minor_radius])

        # used for drawing
        self.draw_ratate_angle = self.env_begin_rad - self.begin_rad
        self.draw_begin_point = np.array([np.cos(self.env_begin_rad) * self.major_radius,
                                          np.sin(self.env_begin_rad) * self.major_radius])
        self.draw_begin_rad_vec = np.array([np.cos(self.env_begin_rad) * self.minor_radius,
                                            np.sin(self.env_begin_rad) * self.minor_radius])
        self.draw_end_point = np.array([np.cos(self.env_end_rad) * self.major_radius,
                                        np.sin(self.env_end_rad) * self.major_radius])
        self.draw_end_rad_vec = np.array([np.cos(self.env_end_rad) * self.minor_radius,
                                          np.sin(self.env_end_rad) * self.minor_radius])

        # print(self.env_begin_rad, self.env_end_rad, self.counter_clockwise)
        self.corridor = directionalPartialAnnulus(np.array([0, 0]),
                                                  self.major_radius,
                                                  self.minor_radius,
                                                  self.begin_rad,
                                                  self.end_rad)
        super().reset(seed=seed)
        # state = [self.major_radius * np.cos(self.begin_rad),
        #          self.major_radius * np.sin(self.begin_rad)] + self.corridor.report_state()

        if self.state_mode == 'mode1':
            state = self.position.tolist() + [self.major_radius, self.end_rad]
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
        self.surf.fill(WHITE)

        x = self.position[0]
        y = self.position[1]
        draw_angle = self.draw_ratate_angle + (-1 + self.counter_clockwise) * self.rad_diff_all_steps
        draw_x = x * math.cos(draw_angle) - y * math.sin(draw_angle)
        draw_y = x * math.sin(draw_angle) + y * math.cos(draw_angle)
        gfxdraw.filled_circle(
            self.surf,
            int(self.offset_x + draw_x * SCALE),
            int(self.offset_y + draw_y * SCALE),
            int(self.UAV_size / 2),
            PURPLE,
        )

        if self.counter_clockwise > 0:
            draw_beg_degree = int(math.degrees(self.env_begin_rad))
            draw_end_degree = int(math.degrees(self.env_end_rad))
        else:
            draw_beg_degree = int(math.degrees(self.env_end_rad))
            draw_end_degree = int(math.degrees(self.env_begin_rad))

        gfxdraw.arc(self.surf,
                    int(self.offset_x),
                    int(self.offset_y),
                    int((self.major_radius + self.minor_radius) * SCALE),
                    draw_beg_degree,
                    draw_end_degree,
                    GRAY)
        gfxdraw.arc(self.surf,
                    int(self.offset_x),
                    int(self.offset_y),
                    int((self.major_radius - self.minor_radius) * SCALE),
                    draw_beg_degree,
                    draw_end_degree,
                    GRAY)

        # draw begin line
        gfxdraw.line(self.surf,
                     int(self.offset_x + (self.draw_begin_point[0] + self.draw_begin_rad_vec[0]) * SCALE),
                     int(self.offset_y + (self.draw_begin_point[1] + self.draw_begin_rad_vec[1]) * SCALE),
                     int(self.offset_x + (self.draw_begin_point[0] - self.draw_begin_rad_vec[0]) * SCALE),
                     int(self.offset_y + (self.draw_begin_point[1] - self.draw_begin_rad_vec[1]) * SCALE),
                     BLUE)

        # draw destination line
        gfxdraw.line(self.surf,
                     int(self.offset_x + (self.draw_end_point[0] + self.draw_end_rad_vec[0]) * SCALE),
                     int(self.offset_y + (self.draw_end_point[1] + self.draw_end_rad_vec[1]) * SCALE),
                     int(self.offset_x + (self.draw_end_point[0] - self.draw_end_rad_vec[0]) * SCALE),
                     int(self.offset_y + (self.draw_end_point[1] - self.draw_end_rad_vec[1]) * SCALE),
                     RED)

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
