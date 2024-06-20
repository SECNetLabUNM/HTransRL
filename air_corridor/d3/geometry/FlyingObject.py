import functools

from pygame import gfxdraw

# from air_corridor.tools._descriptor import Position, PositiveNumber
from air_corridor.tools.util import *

# from utils.memory import DebugTracking

'''
training and testing are very different
train: UAVs are only trained for a single corridor, 
        either DirectionalPartialAnnulusCorridor or RectangleCorridor
test: UAV only use well-trained model for testing in a multi-corridor environment, 
        with all positions reformed with relative positions to the centric of the current corridor.
So, accumulated reward in only calculated within one corridor, not across corridors.
'''


class FlyingObject:
    flying_list = []
    # position = Position(3)
    corridors = None
    '''
    corridors={'A':Corridor{'name','anchor'},
               'B':Corridor{'name','anchor'}}
    '''
    safe_distance = 1
    events = ['won', 'collided', 'breached', 'half', 'breached1', 'breached2', 'collided1', 'collided2']
    GAMMA = 0.99
    capacity = 6
    beta_adaptor_coefficient = 1.0
    num_corridor_in_state = 1

    # Flag indicating if the current environment is in the final corridor.
    # Essential for training in multi-corridor environments and
    # applicable to scenarios with a single-segment corridor in the state.
    corridor_index_awareness = False
    uniform_state = False
    corridor_state_length = 32
    full_state_length = 102
    reduce_space = True
    turbulence_variance = 0

    def __init__(self,
                 position=np.array([0, 0, 0]),
                 position_delta=np.array([0, 0, 0]),
                 next_position=np.array([0, 0, 0]),
                 name=None,
                 velocity=np.array([0, 0, 0]),
                 next_velocity=np.array([0, 0, 0]),
                 discrete=False,
                 invincible=False
                 ):
        self.discrete = discrete
        self.name = name
        self.terminated = False
        self.truncated = False
        self.position = position
        self.position_delta = position_delta
        self.next_position = next_position
        self.velocity = velocity
        self.next_velocity = next_velocity
        self.position_delta = None
        self.status = 'Normal'
        self.invincible = invincible
        self.flying_list.append(self)

    def apply_acceleration(self, acc, dt):
        self.next_velocity, self.position_delta, reward_illegal_acc = apply_acceleration(self.velocity,
                                                                                         self.velocity_max,
                                                                                         acc,
                                                                                         dt)
        self.next_position = self.position + self.position_delta
        if UAV.turbulence_variance > 0:
            position_offset = spherical_to_cartesian(r=random.gauss(mu=0, sigma=UAV.turbulence_variance),
                                                     theta=random.random() * np.pi,
                                                     phi=(random.random() - 0.5) * 2 * np.pi)
            self.next_position += position_offset
            if np.linalg.norm(self.next_position) > 500:
                input('abnormal')
            apply_acceleration(self.velocity, self.velocity_max, acc, dt)
        return reward_illegal_acc

    def get_distance_to(self, other_flying_object):
        distance = distance_point_point(self.position, other_flying_object.position)
        return distance

    def render_self(self):
        """ render itself """
        pass

    @classmethod
    def action_adapter(cls, action):
        '''
        r, theta, phi  = action
        r     = [0, 1] -> [0,1]
        theta = [0, 1] -> [0, np.pi]
        phi   = [0, 1] -> [-np.pi, np.pi]*1.1, with beta base of 1, the selection concentrate on [2pi,0] is truncated.
        :param action:
        :return:
        '''
        return [action[0], action[1] * np.pi, (action[0] - 0.5) * 2 * np.pi * cls.beta_adaptor_coefficient]

        # return [(action[0] - 0.5) * 2 * np.pi, action[1] * np.pi, action[2]]

    @classmethod
    def summary_cluster_state(cls, instances):
        cls.temporal_cluster_state = {}
        positions = []
        for fly_object in instances:
            if not fly_object.terminated:
                positions.append(fly_object.position)
        cluster_center = np.average(positions, axis=0)

        for fly_object in instances:
            # print(f" {fly_object}, {fly_object.terminated}")
            if not fly_object.terminated:
                cls.temporal_cluster_state[fly_object] = fly_object.report_flying(cluster_center)

        # input for transformer need to be consistent, padding only
        for i in range(len(cls.temporal_cluster_state), cls.capacity):
            cls.temporal_cluster_state[id(i)] = [0] * cls.full_state_length
        # print(1)


class UAV(FlyingObject):
    '''unmanned aerial vehicle'''
    corridor_graph = None

    visibility = 0
    temporal_cluster_state = {}

    def __init__(self,
                 init_corridor,
                 des_corridor=None,
                 discrete=False,
                 name=None,
                 # velocity_max=0.6,
                 # acceleration_max=0.6,
                 velocity_max=1.5,
                 acceleration_max=0.3,
                 plane_offset_assigned=None):

        super().__init__(name=name, discrete=discrete)
        # if self.corridor_graph is None:
        #     print("Error: Have not graph the corridors.")
        #     sys.exit()

        self.plane_offset_assigned = plane_offset_assigned

        if discrete:
            self.discrete_action_space = 8

        self.velocity_max = velocity_max
        self.acceleration_max = acceleration_max
        self.init_corridor = init_corridor
        self.des_corridor = des_corridor
        self.enroute = None
        self.instant_reward = 0
        self.outside_counter = None
        self.accumulated_reward = 0
        self.reward = 0
        self.flyingType = [1, 0, 0, 0]
        self.trajectory = []
        self.speeds = []
        self.trajectory_ave_speed = -1  # only apply to the successful arrivals
        self.travel_time=-1

    def update_position(self):
        self.position = self.next_position
        self.velocity = self.next_velocity
        self.speeds.append(np.linalg.norm(self.velocity))
        # print(self.position)

    def decompose_target(self):

        assert (self.enroute['init'] in self.corridor_graph.keys() and
                self.enroute['des'] in self.corridor_graph.keys()), \
            "Error, the initial or the last corridor is not specified."
        path = bfs_find_path(self.corridor_graph, self.enroute['init'], self.enroute['des'])
        if path is None:
            self.enroute['path'] = None
            self.terminated = True
        else:
            self.enroute['path'] = path
            if len(path) > 1:
                self.enroute['next'] = path[1]
        self.enroute['segments'] = len(self.enroute['path'])

    #
    def take(self, action, dt):
        '''
        in take action on the base with reduced space, while output the "actual" values
        '''
        action = self.action_adapter(action)
        # r, theta, phi = action
        r = action[0]
        heading_vector_on_base = polar_to_unit_normal(action[1:])
        if self.reduce_space:
            #  action is generated on based shape with direction of [0,0,1]
            heading_vector = self.corridors[self.enroute['current']].rotate_to_remote(heading_vector_on_base)
        else:
            heading_vector = heading_vector_on_base
        acc = self.acceleration_max * r * heading_vector
        reward_illegal_acc = self.apply_acceleration(acc, dt)
        # here penalize with illegal actions in two parts,
        # 1) action range beyond pre-determined range
        # 2) action within range but enforce uav goes beyond velocity max
        # print(f"acc: {np.round(acc,3)},last vel: {np.round(self.velocity,3)}, "
        #       f"next vel:{np.round(self.next_velocity,3)}, position_delta:{np.round(self.position_delta,3)}")
        return 0  # reward_illegal_acc

    def reset(self):
        self.terminated = False
        self.truncated = False
        self.enroute = {'init': self.init_corridor,
                        'des': self.des_corridor,
                        'current': self.init_corridor,
                        'next': None,
                        'path': None,
                        'segments': 0, }
        self.decompose_target()

        self.position = UAV.corridors[self.enroute['current']].release_uav(self.plane_offset_assigned)
        self.next_position = None
        self.velocity = np.array([0, 0, 0])
        self.next_velocity = None
        self.outside_counter = 0
        self.status = 'Normal'
        UAV.temporal_cluster_state = {}

    def update_accumulated_reward(self):
        self.accumulated_reward = self.accumulated_reward * UAV.GAMMA + self.instant_reward

    def _report_self(self, with_corridor=True, rest_awareness=False):
        # 4+3*4=16
        ref = self.corridors[self.enroute['current']]
        # if self.reduce_space:

        first = [self.velocity_max, self.acceleration_max, 0.4,
                 ref.distance_object_to_point(self.position),
                 np.linalg.norm(self.velocity)] + \
                list(ref.project_to_base(self.position)) + \
                list(ref.rotate_to_base(self.velocity))
        # if torus
        if ref.shapeType == [0, 0, 0, 1]:
            second = list(ref.convert_2_polar(self.position, self.reduce_space)) + \
                     list(ref.convert_vec_2_polar(self.position, self.velocity, self.reduce_space))
        # if cylinder
        elif ref.shapeType == [0, 0, 1, 0]:
            second = [0] * 6
            # indicate whether being in the last corridor

        if any(np.isnan(first + second)):
            print('nan in self')
            input("Press Enter to continue...")
        agent_status = first + second
        if with_corridor:
            corridor_status = self._report_corridor(self.position, reference=ref)
            final = agent_status + corridor_status
        else:
            final = agent_status
        # if rest_awareness:
        #     final += [len(self.enroute['path']) - self.enroute['path'].index(self.enroute['current'])]
        return final, ref

    def _report_other(self, reference, all_flying=None, with_corridor=True, cbf=False):
        other_uavs_status = []
        available = [0, 1]
        for agent in all_flying:
            if agent is self or agent.terminated:
                continue
            relative_position = agent.position - self.position
            distance_self_other = np.linalg.norm(relative_position)
            # standard is 4.5 times
            if distance_self_other > UAV.visibility * self.velocity_max:
                continue

            # 3+3*6=22
            # 2[flying type]+4+3*6=2
            relative_velocity = agent.velocity - self.velocity
            # cur_cor = self.corridors[agent.enroute['current']]
            cur_cor = reference
            # if self.reduce_space:
            if cbf:
                cbf = 2 * (self.acceleration_max + agent.acceleration_max) * (distance_self_other - 0.4) - (
                        np.dot(relative_position, relative_velocity) / (distance_self_other + 1e-5)) ** 2
            else:
                cbf = 0

            first = ([agent.velocity_max, agent.acceleration_max, 0.4,
                      distance_self_other, np.linalg.norm(relative_velocity), cbf] +
                     # list(agent.velocity) +
                     # list(relative_position) +
                     # list(relative_velocity) +
                     list(cur_cor.project_to_base(agent.position)) +
                     list(cur_cor.rotate_to_base(agent.velocity)) +
                     list(cur_cor.rotate_to_base(relative_position)) +
                     list(cur_cor.rotate_to_base(relative_velocity)))
            if cur_cor.shapeType == [0, 0, 0, 1]:
                second = list(cur_cor.convert_2_polar(agent.position, self.reduce_space)) + \
                         list(cur_cor.convert_vec_2_polar(agent.position, agent.velocity, self.reduce_space))
            elif cur_cor.shapeType == [0, 0, 1, 0]:
                second = [0] * 6

            agent_status = available + agent.flyingType + first + second
            if with_corridor:
                if isinstance(agent, UAV):
                    corridor_status = agent._report_corridor(self.position, reference=reference)
                elif isinstance(agent, NCFO):
                    corridor_status = [0] * 64
                other_uavs_status.append(agent_status + corridor_status)
            else:
                corridor_status = []
                pad = 32 - len(agent_status)
                assert pad >= 0, 'other uav expression larger than 32'
                other_uavs_status.append(agent_status + [0] * pad)

        available = [1, 0]
        while len(other_uavs_status) < self.capacity - 1:
            # base_elements = 23 if UAV.corridor_index_awareness else 22
            # other_uavs_status.append([0] * (base_elements + 17 * self.num_corridor_in_state))
            if with_corridor:
                other_uavs_status.append(
                    available + [0] * (24 + UAV.corridor_state_length * self.num_corridor_in_state))
            else:
                other_uavs_status.append(available + [0] * 30)
        return other_uavs_status

    def report_flying(self, cluster_center):
        position_in_cluster = self.position - cluster_center
        # 3+2+3
        global_info = list(position_in_cluster) + self.flyingType + list(self.velocity) + [
            np.linalg.norm(self.velocity), self.velocity_max, self.acceleration_max, ]

        # state for UAVs
        # 4+3*4=16
        cur = self.corridors[self.enroute['current']]
        # if self.reduce_space:
        # 4+3*2
        local_cartesian_info = ([cur.distance_object_to_point(self.position)] +
                                list(cur.project_to_base(self.position)) +
                                list(cur.rotate_to_base(self.velocity)))
        rotation_matrix = (functools.reduce(lambda x, y: x + y, [list(i) for i in cur.rotation_matrix_to_base]) +
                           functools.reduce(lambda x, y: x + y, [list(i) for i in cur.rotation_matrix_to_remote]))

        # if torus, 3*2, for cylinder, padding with 0
        if cur.shapeType == [0, 0, 0, 1]:
            local_spherical_info = list(cur.convert_2_polar(self.position, self.reduce_space)) + \
                                   list(cur.convert_vec_2_polar(self.position, self.velocity, self.reduce_space))
        elif cur.shapeType == [0, 0, 1, 0]:
            local_spherical_info = [0] * 6

        # 42=(3+2+3+1)+18+(3+3*2)+(3*2)
        agent_status = global_info + local_cartesian_info + rotation_matrix + local_spherical_info
        if any(np.isnan(agent_status)):
            print('nan in self')
            input("Press Enter to continue...")
        corridor_status = self._report_corridor(cluster_center)

        # 88 = 42 + 23 * 2
        return agent_status + corridor_status

    def _report_corridor(self, cluster_center=np.array([0, 0, 0]), reference=None):
        # 16 elements
        cur = self.corridors[self.enroute['current']]
        corridor_status = []
        cur_index = self.enroute['path'].index(self.enroute['current'])
        res_path = self.enroute['path'][cur_index:]
        for i, cor_name in enumerate(res_path):
            if i + 1 > UAV.num_corridor_in_state:
                break
            corridor_index_state = [0, 0, 0, 0]
            if res_path[-1] == cor_name:
                corridor_index_state[3] = 1
            elif self.enroute['path'][0] == cor_name:
                corridor_index_state[0] = 1
            else:
                if UAV.corridor_index_awareness == [1, 1, 0, 1]:
                    corridor_index_state[1] = 1  # rest are all 2
                elif UAV.corridor_index_awareness == [1, 1, 1, 1]:
                    # rest following 2-3
                    if cor_name == res_path[-2]:
                        corridor_index_state[2] = 1
                    elif cur.name == cor_name and cor_name in res_path[:-2]:
                        corridor_index_state[1] = 1
                    elif cur.name != cor_name and cor_name in res_path[1:-1]:
                        corridor_index_state[2] = 1
            assert sum(
                corridor_index_state) == 1, f"UAV.corridor_index_awareness: {corridor_index_state}, {cur.name}, {cor_name}, {res_path}"

            position_in_cluster = list(self.position - cluster_center)
            single_c_status = self.corridors[cor_name].report(base=reference)
            # 4+3+16
            single_c_status = corridor_index_state + position_in_cluster + single_c_status

            corridor_status += single_c_status
        # padding if being the last one
        corridor_status += [0] * (UAV.corridor_state_length * (max(0, UAV.num_corridor_in_state - len(res_path))))
        return corridor_status

    def report_corridor_only(self, reference, with_corridor_index):
        # 16 elements
        cur = self.corridors[self.enroute['current']]
        corridor_status = []
        cur_index = self.enroute['path'].index(self.enroute['current'])
        res_path = self.enroute['path'][cur_index:]

        # extra_bits = 6 if with_corridor_index else 2
        for i in range(-1, 3):
            index = i + cur_index
            corridor_index_state = [0, 0, 0, 0]
            corridor_index_state[i + 1] = 1
            assert sum(
                corridor_index_state) == 1, f"UAV.corridor_index_awareness: {corridor_index_state}, {cur.name}, {cor_name}, {res_path}"
            if index < 0 or index >= self.enroute['segments']:
                available = [1, 0]
                single_c_status = [0] * (32 - 6)
            else:
                available = [0, 1]
                cor_name = self.enroute['path'][index]
                single_c_status = self.corridors[cor_name].report(base=reference)
                pad = 32 - len(single_c_status) - 6
                # print(pad, len(single_c_status), extra_bits)
                assert pad >= 0, 'corridor expression more than 32'
                # print(pad,len(single_c_status),extra_bits)
                single_c_status += [0] * pad
            # 4+3+16
            if with_corridor_index:
                single_c_status = available + corridor_index_state + single_c_status
            else:
                single_c_status = available + [0, 0, 0, 0] + single_c_status

            corridor_status.append(single_c_status)
        # padding if being the last one
        return corridor_status

    def report(self,
               trial=2,
               all_flying=None,
               cbf=False,
               rest_awareness=False,
               with_corridor_index=True):
        '''
        corridor_status: 16*n, single is 16
        self= 16+16*n
        other_uav: 22+16*n
        :param padding:
        :param reduce_space:
        :return:
        '''
        if trial == 0:
            try:
                return {'self': UAV.temporal_cluster_state[self], 'other': list(UAV.temporal_cluster_state.values())}
            except:
                print(self)
                print(self.terminated)
                print(UAV.temporal_cluster_state.keys())
        elif trial == 1:
            uav_status, base_corridor = self._report_self()
            other_uavs_status = self._report_other(all_flying=all_flying, reference=base_corridor)
            # print(f" corridor, {len(corridor_status)}")
            # print(f" uav_status, {len(uav_status)}")
            return {'self': uav_status, 'other': other_uavs_status}

        elif trial == 2:
            uav_status, base_corridor = self._report_self(with_corridor=False, rest_awareness=rest_awareness)
            other_uavs_status = self._report_other(all_flying=all_flying, reference=base_corridor, with_corridor=False,
                                                   cbf=cbf)
            corridor_info = self.report_corridor_only(reference=base_corridor, with_corridor_index=with_corridor_index)
            # print(f" corridor, {len(corridor_status)}")
            # print(f" uav_status, {len(uav_status)}")
            return {'self': uav_status, 'other': other_uavs_status + corridor_info}

    def render_self(self, surf):
        if self.status == 'won':
            gfxdraw.filled_circle(
                surf,
                int(OFFSET_x + self.position[0] * SCALE),
                int(OFFSET_y + self.position[1] * SCALE),
                FLYOBJECT_SIZE - 1,
                GREEN,
            )
        elif self.terminated:
            gfxdraw.filled_circle(
                surf,
                int(OFFSET_x + self.position[0] * SCALE),
                int(OFFSET_y + self.position[1] * SCALE),
                FLYOBJECT_SIZE - 1,
                RED,
            )
        else:
            gfxdraw.filled_circle(
                surf,
                int(OFFSET_x + self.position[0] * SCALE),
                int(OFFSET_y + self.position[1] * SCALE),
                FLYOBJECT_SIZE,
                PURPLE,
            )


class NCFO(FlyingObject):
    """
    non-cooperative flying objects
    """
    boundary_max = None
    boundary_min = None
    inflate = 3

    def __init__(self,
                 position=np.array([0, 0, 0]),
                 name=None,
                 velocity=np.array([0, 0, 0]),
                 static=False,
                 invincible=True):
        super().__init__(position, invincible)
        self.static = static
        self.acceleration_max = 0

        if static:
            self.velocity = np.array([0, 0, 0])
            self.position = position
            self.center = position
        self.flyingType = [0, 1, 0, 0]
        self.velocity_max = np.linalg.norm(self.velocity)

    def reset(self):
        # assert not self.static
        if not self.static:
            self.position = self.pick_a_point_outside_corridor()
            self.destination = self.pick_a_point_outside_corridor()
            way_point = self.destination - self.position
            self.velocity = np.random.uniform(1, 1.5) * way_point / np.linalg.norm(way_point)
            self.velocity_max = np.linalg.norm(self.velocity)

    def pick_a_point_outside_corridor(self):
        i = 0
        while 1:
            i += 1
            assert i < 1000, 'dead loop for picking new point'
            point = np.random.uniform(NCFO.boundary_min, NCFO.boundary_max)
            for name, cor in NCFO.corridors.items():
                b, _ = cor.is_inside(point, inflate=self.inflate)
                if b:
                    continue
            return point

    def way_point_algo(self, dt):
        if not self.static:
            self.next_position = self.position + self.velocity * dt
            # print(f"self {self.position}; next {self.next_position}")
            if np.dot(self.next_position - self.destination, self.position - self.destination) < 0:
                self.position = self.destination
                self.destination = self.pick_a_point_outside_corridor()
                way_point = self.destination - self.position
                self.velocity = np.random.uniform(0.5, 2) * way_point / np.linalg.norm(way_point)
        else:
            self.next_position = self.center + 0.01 * (np.random.uniform(size=3) - 0.5)

    def update_position(self):
        self.position = self.next_position

    def report_flying(self, cluster_center):
        position_in_cluster = self.position - cluster_center
        # 3+2+3+1
        flying_info = list(position_in_cluster) + self.flyingType + list(self.velocity) + [
            np.linalg.norm(self.velocity)]
        return flying_info + [0] * (self.full_state_length - 9)
