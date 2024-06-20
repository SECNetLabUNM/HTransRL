import functools

import gymnasium as gym
import pygame
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

from air_corridor.d3.corridor.corridor import CylinderCorridor, DirectionalPartialTorusCorridor
from air_corridor.d3.geometry.FlyingObject import UAV, NCFO
from air_corridor.tools.util import *


def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
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
                 reduce_space=True):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        self.state = None
        self.env_moves = None
        self.corridors = None
        self.render_mode = render_mode
        self.isopen = True
        self.distance_map = None

        self.liability = False
        self.collision_free = False
        self.dt = 1

        self.collisiion_distance = 0.4
        self.all_flying = []

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return spaces.Dict(
            {'self': spaces.Box(low=-100, high=100, shape=(16 + 10,), dtype=np.float32),
             'other': spaces.Box(low=-100, high=100, shape=(22, (self.num_agents - 1)), dtype=np.float32)})

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

    def render(self):
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
        try:
            if self.screen is not None:
                import pygame
                pygame.display.quit()
                pygame.quit()
                self.isopen = False
        except:
            pass

    def update_distance_map(self):
        count = len(self.all_flying)
        self.distance_map = np.ones([count, count]) / 1e-5

        for i in range(count):
            if self.all_flying[i].terminated:
                continue
            for j in range(i + 1, count):
                if self.all_flying[j].terminated:
                    continue
                dis = self.all_flying[i].get_distance_to(self.all_flying[j])
                self.distance_map[i, j] = dis

    def collision_detection(self):
        indices = np.where(self.distance_map < self.collisiion_distance)
        # collide_set = set(reduce((lambda x, y: x + y), [list(i) for i in index]))
        # for i in collide_set:
        #     assert not self.all_flying[i].terminated, 'calculate for terminated flying objects'
        #     if not self.all_flying[i].invincible:
        #         self.all_flying[i].status = 'collided'
        # collide_set = set(reduce((lambda x, y: x + y), [list(i) for i in index]))
        # for i in collide_set:
        #     assert not self.all_flying[i].terminated, 'calculate for terminated flying objects'
        #     if not self.all_flying[i].invincible:
        #         self.all_flying[i].status = 'collided'
        for idx in zip(*indices):
            # for i, j in idx:
            i, j = idx
            # if not self.all_flying[i].invincible:
            if isinstance(self.all_flying[i], UAV):
                if isinstance(self.all_flying[j], UAV):
                    self.all_flying[i].status = 'collided'
                    self.all_flying[j].status = 'collided'
                elif isinstance(self.all_flying[j], NCFO):
                    self.all_flying[i].status = 'collided_UAV'
                    if not self.all_flying[j].invincible:
                        self.all_flying[j].status = 'collided_NC'
            else:
                self.all_flying[j].status = 'collided_UAV'
                if not self.all_flying[i].invincible:
                    self.all_flying[i].status = 'collided_NCFO'

    def random_combination(self, ratio, num):
        seq = []
        for i in range(num):
            if random.random() < ratio:
                seq.append('t')
            else:
                seq.append('c')
        return tuple(seq)

    def generate_structure(self, difficulty=1, seq=None, minor_radius=2.0, test=False):
        '''
        :param connect_plane_anchor: in base,
        :param connect_plane_orientation: in base,
        :param rotation_matrix: base to remote,
        :param anchor_point: base to remote,
        :return:
        1e-3
        65 = ord('A')
        '''

        # distribute obstacles into corridors
        num_corridors = len(seq)
        # assert self.num_obstacles <= num_corridors
        # if not test:
        #     self.num_obstacles = num_corridors
        obstacle_corridor_index = sorted(
            np.random.choice(num_corridors, min(self.num_obstacles, num_corridors), replace=False))
        obstacles = []

        for i in range(num_corridors):
            non_last_flag = num_corridors > i + 1
            name = chr(65 + i)
            minor_radius = minor_radius
            if i == 0:
                intial_anchor = np.random.rand(3) * 2
                initial_orientation_rad = [random.random() * np.pi, (random.random() - 0.5) * 2 * np.pi]  # theta,phi
                if seq[i] == 'c':
                    cor = CylinderCorridor(anchor_point=intial_anchor,
                                           orientation_rad=initial_orientation_rad,
                                           length=random_(difficulty, epsilon=self.epsilon, segment=True) * 15 + 5,
                                           width=minor_radius * 2,
                                           name=name,
                                           connections=[])
                    if non_last_flag:
                        connect_plane_orientation = cor.orientation_vec
                        connect_plane_anchor = cor.endCirclePlane.anchor_point - CORRIDOR_OVERLAP * connect_plane_orientation
                elif seq[i] == 't':
                    begin_rad = np.pi * (2 * random.random() - 1)
                    if test and difficulty >= 1:
                        end_rad = begin_rad + np.pi / 2
                    else:
                        end_rad = begin_rad + np.pi / 2 * random_(difficulty, epsilon=self.epsilon)
                    major_radius = 5 * (random.random() + 1)
                    cor = DirectionalPartialTorusCorridor(name=name,
                                                          anchor_point=intial_anchor,
                                                          orientation_rad=initial_orientation_rad,
                                                          major_radius=major_radius,
                                                          minor_radius=minor_radius,
                                                          begin_rad=begin_rad,
                                                          end_rad=end_rad,
                                                          connections=[])
                    if non_last_flag:
                        connect_plane_orientation = cor.endCirclePlane.orientation_vec
                        connect_plane_anchor = cor.endCirclePlane.anchor_point - CORRIDOR_OVERLAP * connect_plane_orientation
                if non_last_flag:
                    cor.connections = ['B']
                    rotate_to_end_plane = cor.endCirclePlane.rotate_to_remote
                    # rotate_to_end_plane1 = cor.endCirclePlane.rotate_to_base
            else:
                if seq[i] == 'c':
                    length = random_(difficulty, epsilon=self.epsilon, segment=True) * 18 + 2
                    cor = CylinderCorridor(anchor_point=connect_plane_anchor + connect_plane_orientation * length / 2,
                                           orientation_vec=connect_plane_orientation,
                                           length=length,
                                           width=minor_radius * 2,
                                           name=name,
                                           connections=[])
                    if non_last_flag:
                        connect_plane_orientation = cor.orientation_vec
                        connect_plane_anchor = cor.endCirclePlane.anchor_point - CORRIDOR_OVERLAP * connect_plane_orientation
                elif seq[i] == 't':
                    # keep the mojor radius the same if the last one is a torus
                    if seq[i - 1] == 't':
                        major_radius = self.corridors[chr(65 + i - 1)].major_radius
                    else:
                        major_radius = 5 * (random.random() + 1)

                    if i >= 2 and seq[i - 1] == 'c' and seq[i - 2] == 't':
                        assert isinstance(self.corridors[chr(65 + i - 1)], CylinderCorridor)
                        assert isinstance(self.corridors[chr(65 + i - 2)], DirectionalPartialTorusCorridor)
                        last_cylinder_ori = self.corridors[chr(65 + i - 1)].orientation_vec
                        last_torus_ori = self.corridors[chr(65 + i - 2)].orientation_vec
                        unit_vec_connect_point_to_new_obj_anchor = np.cross(last_cylinder_ori, last_torus_ori)
                    else:
                        # pick a random randian for the next anchor point based on last corridor ending plane
                        connect_plane_x = rotate_to_end_plane(X_UNIT)
                        connect_plane_y = rotate_to_end_plane(Y_UNIT)
                        random_rad = (random.random() * 2 - 1) * np.pi
                        unit_vec_connect_point_to_new_obj_anchor = (connect_plane_y * math.sin(random_rad) +
                                                                    connect_plane_x * math.cos(random_rad))

                    new_obj_anchor = connect_plane_anchor + unit_vec_connect_point_to_new_obj_anchor * major_radius
                    orientation_vec = np.cross(-unit_vec_connect_point_to_new_obj_anchor, connect_plane_orientation)

                    new_obj_to_base_matrix = vec2vec_rotation(orientation_vec, Z_UNIT)
                    vec_on_base = np.dot(new_obj_to_base_matrix, -unit_vec_connect_point_to_new_obj_anchor)

                    begin_rad = np.arctan2(vec_on_base[1], vec_on_base[0])
                    # if test and difficulty >= 1:
                    #     end_rad = begin_rad + np.pi / 2
                    # else:
                    #     end_rad = begin_rad + np.pi / 2 * random_(difficulty, epsilon=self.epsilon)
                    end_rad = begin_rad + np.pi / 2 * random_(difficulty, epsilon=self.epsilon)
                    cor = DirectionalPartialTorusCorridor(name=name,
                                                          anchor_point=new_obj_anchor,
                                                          orientation_vec=orientation_vec,
                                                          major_radius=major_radius,
                                                          minor_radius=minor_radius,
                                                          begin_rad=begin_rad,
                                                          end_rad=end_rad,
                                                          connections=[])
                    if non_last_flag:
                        connect_plane_orientation = cor.endCirclePlane.orientation_vec
                        connect_plane_anchor = cor.endCirclePlane.anchor_point - CORRIDOR_OVERLAP * connect_plane_orientation
                if non_last_flag:
                    rotate_to_end_plane = cor.endCirclePlane.rotate_to_remote
                    # rotate_to_end_plane1 = cor.endCirclePlane.rotate_to_base
                    cor.connections = [chr(65 + i + 1)]
            for j in obstacle_corridor_index:
                if j > i:
                    break
                elif j == i:
                    point = cor.sample_a_point_within()
                    # if the corridor is too small, return None
                    if point is not None:
                        obstacles.append(point)
            self.corridors[name] = cor

        # align the center of corridors to [0,0,0]
        anchors = [corridor.anchor_point for key, corridor in self.corridors.items()]
        alignment_offset = np.average(anchors, axis=0)
        for key in self.corridors.keys():
            self.corridors[key].anchor_alignment(off_set=alignment_offset)
        obstacles = [obs - alignment_offset for obs in obstacles]
        # print(obstacles)

        boundry_max = np.max(anchors, axis=0) - alignment_offset + 10
        boundry_min = np.min(anchors, axis=0) - alignment_offset - 10

        # correct cylinder rotation based on the next torus to simplified state space
        # align the following torus orientation to [1,0,0]
        if self.rotate_for_cylinder:
            keys = list(self.corridors.keys())
            for key_1, key_2 in zip(keys[:-1], keys[1:]):
                cylinder_1, torus_2 = self.corridors[key_1], self.corridors[key_2]
                if isinstance(cylinder_1, CylinderCorridor) and isinstance(torus_2, DirectionalPartialTorusCorridor):
                    cylinder_1.determine_rotation_with_next_torus(torus_2.orientation_vec)
        return obstacles, boundry_max, boundry_min

    def reset(self,
              seed=None,
              options=None,
              num_agents=3,
              reduce_space=True,
              level=10,
              ratio=1,
              liability=True,
              collision_free=False,
              beta_adaptor_coefficient=1.0,
              num_corridor_in_state=1,
              dt=1.0,
              corridor_index_awareness=False,
              velocity_max=1.5,
              acceleration_max=0.3,
              uniform_state=False,
              dynamic_minor_radius=False,
              epsilon=0.1,
              num_obstacles=0,
              num_ncfo=0,
              visibility=4.5,
              capacity=None,
              rotate_for_cylinder=True,
              test=False,
              state_choice=2,
              cbf=False,
              rest_awareness=False,
              with_corridor_index=True,
              turbulence_variance=0):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `env_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        if reduce_space:
            self.rotate_for_cylinder = rotate_for_cylinder
        else:
            self.rotate_for_cylinder = False
        self.rest_awareness = rest_awareness
        self.with_corridor_index = with_corridor_index
        self.cbf = cbf
        self.epsilon = epsilon
        self.dt = dt
        self.liability = liability
        self.collision_free = collision_free

        self.num_obstacles = num_obstacles
        self.ncfos = []
        self.state_choice = state_choice

        # setup corridors
        difficulty = 1 if options is None else options['difficulty']

        self.corridors = {}
        # the following 4 parameters used for generating training env only
        CylinderCorridor.reduce_space = reduce_space
        DirectionalPartialTorusCorridor.reduce_space = reduce_space

        if level == 0:
            begin_rad = -np.pi
            end_rad = begin_rad + np.pi / 2
            major_radius = 10
            # orientation_rad = [0, 0]  #  theta,phi
        elif level == 1:
            begin_rad = np.pi * (2 * random.random() - 1)
            if difficulty <= 1:
                end_rad = begin_rad + np.pi / 2 * (difficulty + random.uniform(-0.1, 0.1))
            else:
                end_rad = begin_rad + np.pi / 2 * random.uniform(0.9, difficulty + 0.1)
            major_radius = 10
            orientation_rad = [random.random() * np.pi, (random.random() - 0.5) * 2 * np.pi]  # theta,
            if random.random() > ratio:
                self.corridors['A'] = CylinderCorridor(anchor_point=np.array([0, 0, 0]),
                                                       orientation_rad=orientation_rad,
                                                       length=random.random() * difficulty * 15 + 5,
                                                       width=4,
                                                       name='A',
                                                       connections=[])
            else:
                self.corridors['A'] = DirectionalPartialTorusCorridor(name='A',
                                                                      anchor_point=np.array([0, 0, 0]),
                                                                      orientation_rad=orientation_rad,
                                                                      major_radius=major_radius,
                                                                      minor_radius=2,
                                                                      begin_rad=begin_rad,
                                                                      end_rad=end_rad,
                                                                      connections=[],
                                                                      )
            self.segments = 1
        elif level == 2:
            # fixed ending degree and fixed radius, but gradually increase fixed ending degree
            seq = ('c')
            obstacles, boundary_max, boundary_min = self.generate_structure(difficulty, seq, test=test)
        elif level == 3:
            # fixed ending degree and fixed radius, but gradually increase fixed ending degree
            seq = ('t')
            obstacles, boundary_max, boundary_min = self.generate_structure(difficulty, seq, test=test)
        elif level == 4:
            # fixed ending degree and fixed radius, but gradually increase fixed ending degree
            seq = self.random_combination(ratio, num=1)
            obstacles, boundary_max, boundary_min = self.generate_structure(difficulty, seq, test=test)
        elif level == 6:
            seq = ('c', 't')
            obstacles, boundary_max, boundary_min = self.generate_structure(difficulty, seq=seq, test=test)
        elif level == 7:
            seq = ('t', 't')
            obstacles, boundary_max, boundary_min = self.generate_structure(difficulty, seq=seq, test=test)
        elif level == 8:
            seq = ('c', 't', 't')
            obstacles, boundary_max, boundary_min = self.generate_structure(difficulty, seq=seq, test=test)
        elif level == 9:
            seq = ('t', 't', 't')
            obstacles, boundary_max, boundary_min = self.generate_structure(difficulty, seq=seq, test=test)
        elif level == 10:
            seq = random.choice([('t'), ('c')])
            obstacles, boundary_max, boundary_min = self.generate_structure(difficulty, seq=seq, test=test)
        elif level == 11:
            seq = random.choice([('t', 't'), ('t', 'c'), ('c', 't')])
            obstacles, boundary_max, boundary_min = self.generate_structure(difficulty, seq=seq, test=test)
        elif level == 12:
            seq = random.choice([('t', 't', 'c'), ('c', 't', 't'), ('t', 'c', 't')])
            obstacles, boundary_max, boundary_min = self.generate_structure(difficulty, seq=seq, test=test)
        elif level == 13:
            if dynamic_minor_radius:
                minor_radius = np.random.uniform(1.8, 2.2)
            else:
                minor_radius = 2
            seq = random.choices([('t', 't', 'c'), ('c', 't', 't'), ('t', 'c', 't'), ('c', 't', 'c')],
                                 weights=[1.0, 1.0, 0.8, 0.8])[0]
            obstacles, boundary_max, boundary_min = self.generate_structure(difficulty, seq=seq, test=test,
                                                                            minor_radius=minor_radius)
        elif level == 14:
            seq = ('c', 't', 't', 'c')
            obstacles, boundary_max, boundary_min = self.generate_structure(difficulty, seq=seq, test=test)
        elif level == 15:
            seq = random.choice([('t', 't', 'c', 't'),
                                 ('t', 'c', 't', 't'),
                                 ('t', 'c', 't', 'c'),
                                 ('c', 't', 't', 'c'),
                                 ('c', 't', 'c', 't')])
            obstacles, boundary_max, boundary_min = self.generate_structure(difficulty, seq=seq, test=test)
        elif level == 16:
            seq = random.choice([('t', 't', 'c', 't', 't'),
                                 ('t', 't', 'c', 't', 'c'),
                                 ('t', 'c', 't', 't', 'c'),
                                 ('t', 'c', 't', 'c', 't'),
                                 ('c', 't', 't', 'c', 't'),
                                 ('c', 't', 'c', 't', 't'),
                                 ('c', 't', 'c', 't', 'c')])
            obstacles, boundary_max, boundary_min = self.generate_structure(difficulty, seq=seq, test=test)
        elif level == 17:
            seq = random.choice([('t', 't', 'c', 't', 't', 'c'),
                                 ('t', 't', 'c', 't', 'c', 't'),
                                 ('t', 'c', 't', 't', 'c', 't'),
                                 ('t', 'c', 't', 'c', 't', 't'),
                                 ('t', 'c', 't', 'c', 't', 'c'),
                                 ('c', 't', 't', 'c', 't', 't'),
                                 ('c', 't', 't', 'c', 't', 'c'),
                                 ('c', 't', 'c', 't', 't', 'c'),
                                 ('c', 't', 'c', 't', 'c', 't')])
            obstacles, boundary_max, boundary_min = self.generate_structure(difficulty, seq=seq, test=test)
        elif level == 18:
            seq = random.choice([('t', 't', 'c', 't'),
                                 ('t', 'c', 't', 't'),
                                 ('c', 't', 't', 'c')])
            obstacles, boundary_max, boundary_min = self.generate_structure(difficulty, seq=seq, test=test)
        elif level == 19:
            seq = random.choice([('t', 't', 'c', 't', 't'),
                                 ('t', 'c', 't', 't', 'c'),
                                 ('c', 't', 't', 'c', 't')])
            obstacles, boundary_max, boundary_min = self.generate_structure(difficulty, seq=seq, test=test)
        elif level == 20:
            seq = random.choice([('c', 't', 't', 'c', 't', 't', 'c'), ])
            obstacles, boundary_max, boundary_min = self.generate_structure(difficulty, seq=seq, test=test)
        elif level == 21:
            seq = random.choice([('c', 't', 't', 'c', 't', 't', 'c', 't', 't', 'c'), ])
            obstacles, boundary_max, boundary_min = self.generate_structure(difficulty, seq=seq, test=test)
        elif level == 22:
            seq = random.choice([('c', 't', 't', 'c', 't', 't', 'c', 't', 't', 'c', 't', 't', 'c'), ])
            obstacles, boundary_max, boundary_min = self.generate_structure(difficulty, seq=seq, test=test)
        elif level == 31:
            seq = random.choice([('t', 't'), ('c', 't'), ('t', 'c'), ('t', 't', 'c'), ('c', 't', 't'), ('t', 'c', 't'),
                                 ('c', 't', 't', 'c'), ('t', 'c', 't', 'c'), ('c', 't', 'c', 't'), ('t', 't', 'c', 't'),
                                 ('t', 'c', 't', 't')])
            obstacles, boundary_max, boundary_min = self.generate_structure(difficulty, seq=seq, test=test)
        elif level == 32:
            seq = random.choice([('c', 't', 't', 'c', 't', 't', 'c'), ('t', 't', 'c'), ('c', 't', 't'), ('t', 'c', 't'),
                                 ('c', 't', 't', 'c'), ('t', 'c', 't', 'c'), ('c', 't', 'c', 't'), ('t', 't', 'c', 't'),
                                 ('t', 'c', 't', 't')])
            obstacles, boundary_max, boundary_min = self.generate_structure(difficulty, seq=seq, test=test)
        elif level == 33:
            seq = random.choice([('c'), ('t'), ('c', 't'), ('t', 't'), ('t', 't', 'c'), ('c', 't', 't'),
                                 ('c', 't', 't', 'c'), ('t', 'c', 't', 't')])
            obstacles, boundary_max, boundary_min = self.generate_structure(difficulty, seq=seq, test=test)
        elif level == 34:
            seq = random.choice([('c', 't', 't', 'c', 't', 't', 'c'), ('c', 't', 't', 'c'), ('t')])
            obstacles, boundary_max, boundary_min = self.generate_structure(difficulty, seq=seq, test=test)

        # if not test:
        #     assert len(seq) >= sum(corridor_index_awareness)
        corridor_graph = self.corridors['A'].convert2graph(self.corridors)

        '''
        self.all_falying: all flying objects
        UAV.flying_list: all UAVs
        self.agents: all live UAVs
        '''

        # setup uavs
        plane_offsets = distribute_evenly_within_circle(radius=2, min_distance=1, num_points=num_agents)
        UAV.flying_list = []
        self.agents = [UAV(init_corridor='A',
                           des_corridor=chr(64 + len(self.corridors)),
                           name=i,
                           plane_offset_assigned=plane_offset,
                           velocity_max=velocity_max,
                           acceleration_max=acceleration_max) for i, plane_offset in enumerate(plane_offsets)]
        UAV.corridors = self.corridors
        UAV.reduce_space = reduce_space
        UAV.corridor_graph = corridor_graph
        UAV.beta_adaptor_coefficient = beta_adaptor_coefficient
        UAV.num_corridor_in_state = num_corridor_in_state
        UAV.capacity = max(capacity,
                           num_agents + self.num_obstacles + num_ncfo) if capacity else num_agents + self.num_obstacles + num_ncfo
        UAV.corridor_index_awareness = corridor_index_awareness
        UAV.turbulence_variance=turbulence_variance

        # index capability with 4 bits
        # index up to 2: [1,0,0,1]; up to 3: [1,1,0,1]; up to 4: [1,1,1,1].

        # UAV.corridor_state_length = 28  # * num_corridor_in_state
        UAV.uniform_state = uniform_state
        UAV.visibility = visibility

        # setup NCFOs
        NCFO.flying_list = []
        NCFO.corridors = self.corridors
        NCFO.boundary_max = boundary_max
        NCFO.boundary_min = boundary_min
        for obs_position in obstacles:
            self.ncfos.append(NCFO(position=obs_position, static=True))
        for other in range(num_ncfo):
            obj = NCFO()
            obj.reset()
            self.ncfos.append(obj)

        # flying lists concatenate
        self.all_flying = UAV.flying_list + NCFO.flying_list

        # reset all flying objects
        [agent.reset() for agent in self.all_flying]
        self.env_moves = 0

        if self.state_choice == 0: UAV.summary_cluster_state(self.all_flying)
        observations = {agent: agent.report(self.state_choice, all_flying=self.all_flying, cbf=self.cbf,
                                            rest_awareness=self.rest_awareness,
                                            with_corridor_index=self.with_corridor_index) for agent in
                        self.agents}

        # observations = {agent: agent.report() for agent in self.agents}
        # UAV.temporal_cluster_state = {}

        self.state = observations
        if self.render_mode == "human":
            self.render()
        infos = {'corridor_seq': seq}
        return observations, infos

    def step(self, action_dic):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """

        # all flying objects promptly move 1 step
        rewards = {agent: agent.take(action, self.dt) for agent, action in action_dic.items()}
        [ncfo.way_point_algo(self.dt) for ncfo in self.ncfos]

        # print([ncfo.next_position for ncfo in self.ncfos])

        # collision detection
        if not self.collision_free:
            self.update_distance_map()
            self.collision_detection()

        disaster = False
        for agent, _ in rewards.items():
            if not agent.terminated:
                if agent.status == 'collided':
                    reward_from_corridor = PENALTY_COLLISION
                else:
                    reward_from_corridor = agent.corridors[agent.enroute['current']].evaluate_action(agent)
                if agent.status != 'Normal':
                    disaster = True
                rewards[agent] += reward_from_corridor
                agent.instant_reward = rewards[agent]

        for agent in self.all_flying:
            if not agent.terminated:
                if isinstance(agent, UAV):
                    if not agent.terminated and self.liability and disaster:
                        rewards[agent] = rewards[agent] + LIABILITY_PENALITY
                    agent.update_position()
                    agent.update_accumulated_reward()
                elif isinstance(agent, NCFO):
                    agent.update_position()
                    # print(agent.position)

        '''
        observation need to be processed before terminations
        only observe non terminated flying objects, 
        terminations updates termination status
        '''
        if self.state_choice == 0: UAV.summary_cluster_state(self.all_flying)
        observations = {agent: agent.report(self.state_choice, all_flying=self.all_flying, cbf=self.cbf,
                                            rest_awareness=self.rest_awareness) for agent in
                        self.agents}

        self.env_moves += 1
        env_truncation = self.env_moves >= NUM_ITERS
        truncations = {agent: env_truncation for agent in self.agents}

        # terminations = {agent: agent.status in UAV.events for agent in self.agents}
        terminations = {}
        for agent in self.agents:
            cond = agent.status != 'Normal'
            if cond:
                agent.terminated = cond
                terminations[agent] = agent.terminated
                if agent.status == 'won':
                    agent.trajectory_ave_speed = np.average(agent.speeds)
                    agent.travel_time=len(agent.speeds)
            else:
                terminations[agent] = False

        for ncfo in NCFO.flying_list:
            if ncfo.status != 'Normal':
                ncfo.terminated = True

        self.state = observations

        if self.render_mode == "human":
            self.render()
        infos = {agent: None for agent in self.agents}

        return observations, rewards, terminations, truncations, infos
