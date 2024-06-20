from pygame import gfxdraw

from air_corridor.tools._descriptor import Position, PositiveNumber
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
    position = Position(2)

    '''
    corridors={'A':Corridor{'name','anchor'},
               'B':Corridor{'name','anchor'}}
    '''
    safe_distance = 1
    capacity = 3
    events = ['won', 'collided', 'breached']
    GAMMA = 0.99

    def __int__(self,
                name,
                position=np.array([0, 0]),
                position_delta=np.array([0, 0]),
                next_position=np.array([0, 0]),
                velocity=np.array([0, 0]),
                next_velocity=np.array([0, 0]),
                discrete=False):
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
        self.acceleration = None

        self.status = None

    def apply_acceleration(self, dt):
        self.next_velocity, self.position_delta, reward_illegal_acc = apply_acceleration(self.velocity,
                                                                                         self.velocity_max,
                                                                                         self.acceleration,
                                                                                         dt)

        self.next_position = self.position + self.position_delta

        return reward_illegal_acc

    def get_distance_to(self, other_flying_object):
        distance = distance_point_point(self.position, other_flying_object.position)
        return distance

    def render_self(self):
        """ render itself """
        pass

    def action_adapter(self, action):
        return [2 * (action[0] - 0.5) * np.pi, action[1]]


class UAV(FlyingObject):
    '''unmanned aerial vehicle'''
    corridor_graph = None
    corridors = None

    def __init__(self,
                 init_corridor,
                 des_corridor=None,
                 corridors=None,
                 discrete=False,
                 name=None,
                 # velocity_max=0.6,
                 # acceleration_max=0.6,
                 # velocity_max=0.9,
                 # acceleration_max=0.3,
                 velocity_max=1.5,
                 acceleration_max=0.3,
                 graph=None,
                 location_assigned=None):

        # if self.corridor_graph is None:
        #     print("Error: Have not graph the corridors.")
        #     sys.exit()
        super().__int__(name, discrete=discrete)

        self.location_assigned = location_assigned

        if discrete:
            self.discrete_action_space = 8

        self.corridors = corridors
        self.corridor_graph = graph

        self.velocity_max = velocity_max
        self.acceleration_max = acceleration_max

        self.init_corridor = init_corridor
        self.des_corridor = des_corridor
        self.enroute = None

        self.instant_reward = 0
        self.cumulative_reward = 0

        self.neighbors = []

        self.steps = 0
        self.outside_counter = None
        self.flying_list.append(self)

        self.accumulated_reward = 0

        self.trajectory = []
        self.reward = 0
        self.extra_report = True

    def update_position(self):
        self.position = self.next_position
        self.velocity = self.next_velocity
        # print(self.position)

    def decompose_target(self):

        assert (self.enroute['init'] in self.corridor_graph.keys() and self.enroute[
            'des'] in self.corridor_graph.keys()), \
            "Error, the initial or the last corridor is not specified."
        path = bfs_find_path(self.corridor_graph, self.enroute['init'], self.enroute['des'])
        if path is None:
            self.enroute['path'] = None
            self.terminated = True
        else:
            self.enroute['path'] = path

    # def step(self, action):
    #     self.take(action)
    #     self.

    def take(self, action, dt):
        # action has [direction, acc]
        action = self.action_adapter(action)
        # action_rad = action[0]
        # heading_vector = np.array([np.cos(action_rad), np.sin(action_rad)])
        heading_vector = polar_to_unit_normal([action[0]])
        acc = self.acceleration_max * action[1] * heading_vector
        # self.update_position(acc)
        self.acceleration = acc
        reward_illegal_acc = self.apply_acceleration(dt)
        self.steps += 1

        # here penalize with illegal actions in two parts,
        # 1) action range beyond pre-determined range
        # 2) action within range but enforce uav goes beyond velocity max
        # print(f"acc: {np.round(acc,3)},last vel: {np.round(self.velocity,3)}, "
        #       f"next vel:{np.round(self.next_velocity,3)}, position_delta:{np.round(self.position_delta,3)}")
        return reward_illegal_acc

    def reset(self):
        self.terminated = False
        self.truncated = False

        self.enroute = {'init': self.init_corridor,
                        'des': self.des_corridor,
                        'current': self.init_corridor}
        self.decompose_target()

        self.position = self.corridors[self.enroute['current']].release_uav(self.location_assigned)
        self.next_position = None
        self.velocity = np.array([0, 0])
        self.next_velocity = None
        self.outside_counter = 0
        self.status = None

    def update_cumulative_reward(self):
        self.cumulative_reward = self.cumulative_reward * UAV.GAMMA + self.instant_reward
        if self.cumulative_reward > 200:
            print(1)

    def _report_self(self):
        # 2+2+2 =6
        position = list(self.corridors[self.enroute['current']].point_relative_center_position(self.position))
        velocity = list(self.velocity)
        max_values = [self.velocity_max, self.acceleration_max]
        minimum_state = position + velocity + max_values
        if self.extra_report:
            # 6+ 2+2+2=12

            polar_position = list(self.corridors[self.enroute['current']].convert_2_polar(self.position))
            polar_velocity = list(
                self.corridors[self.enroute['current']].convert_vec_2_polar(self.position, self.velocity))
            distance_to_inner_outer_circles = list(
                self.corridors[self.enroute['current']].distance_to_inner_outer_circles(self.position))

            return minimum_state + polar_position + polar_velocity + distance_to_inner_outer_circles
        else:
            # 2+2+2
            return minimum_state

    def report(self, padding=True):
        corridor_status = self.corridors[self.enroute['current']].report()
        uav_status = self._report_self()
        other_uavs_status = []

        for one_agent in self.flying_list:

            if one_agent is not self:
                # 1+ 2+2+2=7
                terminal_state = [not one_agent.terminated]
                position = list(self.corridors[self.enroute['current']].point_relative_center_position(
                    one_agent.position))
                velocity = list(one_agent.velocity)
                max_values = [one_agent.velocity_max, one_agent.acceleration_max]
                minimum_state = terminal_state + position + velocity + max_values
                if self.extra_report:
                    # (7)+ 2+1+2  + 2+2=16

                    relative_position = list(self.corridors[self.enroute['current']].point_relative_center_position(
                        self.position) - self.corridors[
                                                 self.enroute['current']].point_relative_center_position(
                        one_agent.position))
                    distance = [np.linalg.norm(
                        self.corridors[self.enroute['current']].point_relative_center_position(
                            self.position) - self.corridors[
                            self.enroute['current']].point_relative_center_position(
                            one_agent.position))]
                    relative_velocity = list(one_agent.velocity - self.velocity)
                    polar_position = list(self.corridors[self.enroute['current']].convert_2_polar(one_agent.position))
                    polar_velocity = list(self.corridors[self.enroute['current']].convert_vec_2_polar(
                        one_agent.position,
                        one_agent.velocity))
                    one_agent_status = minimum_state + \
                                       relative_position + distance + relative_velocity + polar_position + polar_velocity
                else:
                    # 1+ 2+2+2
                    one_agent_status = minimum_state
                other_uavs_status.append(one_agent_status)

        while padding and len(other_uavs_status) < FlyingObject.capacity - 1:
            if self.extra_report:
                other_uavs_status.append([0] * 16)
            else:
                other_uavs_status.append([0] * 7)
        return {'self': uav_status + corridor_status, 'other': other_uavs_status}
        # return uav_status + corridor_status + other_uavs_status

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
    boundary = [None] * 3
    velocity = PositiveNumber()

    def __int__(self, position, velocity):
        super().__int__(position)
        self.velocity = velocity
        self.flying_object_list.append(self)

    def setup_boundary(self, boundary):
        # setup around [0,0,0]
        self.boundary = boundary / 2

    def is_boundary_breach(self, tentative_next_position):
        return True if any(tentative_next_position > self.boundary) or any(
            tentative_next_position < -self.boundary) else False


class Baloon(NCFO):

    def __int__(self, position, speed, velocity):
        super().__int__(position, velocity)
        self.flying_object_list.append(self)

    def update_position(self):
        while True:
            tentative_next_position = self.position + self.direction * self.speed
            if not self.is_boundary_breach(tentative_next_position):
                self.position = tentative_next_position
                break
            v = np.random.randn(3)


class Flight(NCFO):
    pass
