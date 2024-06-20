import functools
import random

import gymnasium as gym
import pygame
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

from air_corridor.d2.corridor.corridor import RectangleCorridor, directionalPartialAnnulusCorridor
from air_corridor.d2.geometry.FlyingObject import UAV
from air_corridor.tools.uti_consts import *
from air_corridor.tools.util import distribute_evenly_on_line, collide_time_position


# @statistics
# def unpack(state):
#     s1={}
#     s2={}
#     s1_lst=[]
#     s2_lst=[]
#     for agent,s in state:
#         s1 = {agent: s['self'] for agent, s in state}
#         s1_lst.append(s['self'])
#         s2 = {agent: s['other'] for agent, s in state}
#         s2_lst.append(s['other'])


def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "rps_v2"}

    def __init__(self,
                 render_mode=None,
                 ):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        self.corridors = None

        self.render_mode = render_mode

        self.isopen = True

        self.distance_map = None

        self.velocity_align = None

        self.video_recording = False

        # self.num=num

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        # 4: self poistion + velocity
        # 5: corridor info
        # 5* existence+ neighbor poistion + velocity

        # return spaces.Dict(
        #     {'self': spaces.Box(low=-100, high=100, shape=(6 + 5,), dtype=np.float32),
        #      'other': spaces.Box(low=-100, high=100, shape=(7, (self.num_agents - 1)), dtype=np.float32)})
        return spaces.Dict(
            {'self': spaces.Box(low=-100, high=100, shape=(12 + 18,), dtype=np.float32),
             'other': spaces.Box(low=-100, high=100, shape=(14, (self.num_agents - 1)), dtype=np.float32)})

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def render(self, writer=None):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """

        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if not hasattr(self, 'screen') or self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (SCREEN_WIDTH, SCREEN_HEIGHT)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        if not hasattr(self, 'clock') or self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.surf.fill(WHITE)

        # if len(self.agents) == 2:
        #     string = "Current state: Agent1: {} , Agent2: {}".format(
        #         MOVES[self.state[self.agents[0]]], MOVES[self.state[self.agents[1]]]
        #     )
        # else:
        #     string = "Game over"
        # print(string)
        for _, one_corridor in self.corridors.items():
            one_corridor.render_self(self.surf)
        for agent in self.agents:
            agent.render_self(self.surf)

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            # self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def boid_evaluation_n_collision_detection(self, collide_distance=0.4, vigilant_distance=2):
        count = len(self.agents)
        n_second_ahead_collision = {agent: float('inf') for agent in self.agents}
        # self.distance_map = np.ones([count, count]) * float('inf')  # np.ones([count, count]) / 0.
        # self.velocity_map = np.ones([count, count]) * float('inf')

        for i, agent_A in enumerate(self.agents):

            for j in range(i + 1, count):
                agent_B = self.agents[j]
                t_A, t_B = collide_time_position(p_A=agent_A.position,
                                                 v_A=agent_A.velocity,
                                                 p_B=agent_B.position,
                                                 v_B=agent_B.velocity,
                                                 collide_distance=collide_distance)
                if t_A > 0:
                    n_second_ahead_collision[agent_A] = min(t_A, n_second_ahead_collision[agent_A])
                    n_second_ahead_collision[agent_B] = min(t_A, n_second_ahead_collision[agent_B])
                elif t_A <= 0 and 0 <= t_B:
                    self.agents[i].status = 'collided'
                    self.agents[j].status = 'collided'

            # concern degree, larger is better
            n_second_ahead_collision[agent_A] = max(
                agent_A.velocity_max / agent_A.acceleration_max - n_second_ahead_collision[agent_A], 3)

        return n_second_ahead_collision

    def ZCBF_reward(self, collision_distance=0.4):
        n = len(self.agents)
        ZCBF_measure = {agent: 0 for agent in self.agents}
        for i, a_i in enumerate(self.agents):
            for j in range(i + 1, n):
                a_j = self.agents[j]
                delta_p_ij = a_i.position - a_j.position
                norm_p = np.linalg.norm(a_i.position - a_j.position)
                if norm_p < collision_distance:
                    a_i.status = 'collided'
                    a_j.status = 'collided'
                else:
                    delta_v_ij = a_i.velocity - a_j.velocity
                    alpha_i = a_i.acceleration_max
                    alpha_j = a_j.acceleration_max
                    Ds = collision_distance
                    r_ij = - np.dot(delta_p_ij, delta_v_ij) / norm_p / np.sqrt(2 * (alpha_i + alpha_j) * (norm_p - Ds))
                    ZCBF_measure[a_i] = max(ZCBF_measure[a_i], r_ij)
                    ZCBF_measure[a_j] = max(ZCBF_measure[a_j], r_ij)
        return ZCBF_measure

    def access_neighbor_info(self):
        info = []
        for agent_i in self.agents:
            single_info = []
            for agent_j in self.agents:
                if agent_i is agent_j:
                    continue
                else:
                    single_info.append(list(self.agent_j.position - agent_i.position) + list(self.agent_j.velocity))
            info.append(single_info)
        return info

    def reset(self, seed=None, options=None, num_agents=3):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `env_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """

        # 3 difficult parameters
        difficultness = 1 if options is None else min(pow(options['difficultness'], 1 / 3), 1)

        if random.random() > 1:
            self.corridors = {'A': RectangleCorridor(anchor_point=np.array([0, 0]),
                                                     direction_rad=(np.random.random() - 0.5) * np.pi,
                                                     length=(random.random() + 1) * 10,
                                                     width=4,
                                                     name='A',
                                                     connections=[])}
        else:
            # begin_rad = (np.pi * 2 * random.random() - np.pi) * difficultness
            # counter_clockwise = 1 if np.random.randint(2) == 0 else -1
            # end_rad = begin_rad + counter_clockwise * np.pi * (0.5 + np.random.random() / 2 * difficultness)
            # major_radius = np.random.random() * 5 * difficultness + 5

            begin_rad = (np.pi * 2 * random.random() - np.pi) * difficultness
            counter_clockwise = 1 if np.random.randint(2) == 0 else -1
            end_rad = begin_rad + counter_clockwise * np.pi * (0.5 + np.random.random() / 2 * difficultness)
            major_radius = np.random.random() * 10 * difficultness + 5

            # begin_rad = (np.pi * 2 / 16 * np.random.randint(16) - np.pi) * difficultness
            # counter_clockwise = 1 if np.random.randint(2) == 0 else -1
            # end_rad = begin_rad + counter_clockwise * np.pi * (0.5 + np.random.randint(9) / 8 * difficultness)
            # major_radius = np.random.randint(5, 16)

            # print(begin_rad, counter_clockwise, end_rad)
            self.corridors = {'A': directionalPartialAnnulusCorridor(name='A',
                                                                     anchor_point=np.array([0, 0]),
                                                                     major_radius=major_radius,
                                                                     minor_radius=2,
                                                                     begin_rad=begin_rad,
                                                                     end_rad=end_rad,
                                                                     connections=[])}
        cor_graph = RectangleCorridor.convert2graph(self.corridors)

        # setup uavs
        distribute_percent = distribute_evenly_on_line(line_length=4, min_distance=1, num_points=num_agents)
        UAV.flying_list = []
        self.agents = [UAV(init_corridor='A',
                           des_corridor='A',
                           velocity_max=1,
                           name=None,
                           corridors=self.corridors,
                           graph=cor_graph,
                           location_assigned=location_assigned)
                       for location_assigned in distribute_percent]

        [agent.reset() for agent in self.agents]

        self.env_moves = 0

        observations = {agent: agent.report() for agent in self.agents}

        # infos = {agent: {} for agent in self.agents}
        # observe_neighbor = self.access_neighbor_info()

        self.state = observations
        if self.render_mode == "human":
            self.render()

        infos = None
        # dones = [False for _ in self.agents]

        return observations, infos

    def step(self, action_dic, dt=0.2, consider_boid=False):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # set up reward lst for all agents, each agent take a step
        # agent.take() return reward about the action's range
        # rewards = []
        # for agent, action in zip(self.agents, action_lst):
        #     if not agent.terminated:
        #
        #         rewards.append(agent.take(action, dt))
        #     else:
        #         rewards.append(0)
        rewards = {agent: agent.take(action, dt) for agent, action in action_dic.items()}

        # collision detection
        # boid_conern = self.boid_evaluation_n_collision_detection()
        boid_conern = self.ZCBF_reward()

        for agent, _ in rewards.items():
            if not agent.terminated:
                if agent.status == 'collided':
                    reward_from_corridor = PENALTY_COLLISION
                    penalty_boid = 0
                else:
                    if consider_boid:
                        penalty_boid = (1 - boid_conern[agent]) * REWARD_BOID
                    else:
                        penalty_boid = 0
                    reward_from_corridor = agent.corridors[agent.enroute['current']].evaluate_action(agent)
                rewards[agent] += reward_from_corridor + penalty_boid
                agent.instant_reward = rewards[agent]

        for agent in self.agents:
            if not agent.terminated:
                agent.update_position()
                agent.update_cumulative_reward()

        self.env_moves += 1
        env_truncation = self.env_moves >= NUM_ITERS
        truncations = {agent: env_truncation for agent in self.agents}

        # terminations = {agent: agent.status in UAV.events for agent in self.agents}
        terminations = {}
        for agent in self.agents:
            agent.terminated = agent.status in UAV.events
            terminations[agent] = agent.terminated

        observations = {agent: agent.report() for agent in self.agents}

        self.state = observations

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        # infos = {agent: {'status': agent.status, 'reward': agent.cumulative_reward, 'steps': agent.steps} for agent in
        #          self.agents}

        # if env_truncation:
        #     self.agents = []
        if self.render_mode == "human":
            self.render()

        # remove terminated agents
        # self.agents = [agent for agent in self.agents if not agent.terminated]

        return observations, rewards, terminations, truncations, None
