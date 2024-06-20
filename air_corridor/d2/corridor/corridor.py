import functools

from pygame import gfxdraw

from air_corridor.d2.geometry.geom import *
from air_corridor.tools.uti_consts import *

'''
corridors={'A':{'object':Torus,'connections':{'B':LineSegment,'D':LineSegment }},
					'B':{},
					'C':{}}
corridors=[{'object':Torus,'connections':{'B':LineSegment,'D':LineSegment }},
		   { },
		   { }]
* corridors
    - object: 
        -- Torus_1
    - connections: 
        -- LineSegment_1
        -- LineSegment_2
					'''


class Corridor:
    # all_corridors = []
    graph = None

    def __init__(self, name, connections):
        # Initialize corridor properties
        self.name = name
        self.connections = connections

    @classmethod
    def convert2graph(cls, corridors):
        cls.graph = {}
        for name, one_corridor in corridors.items():
            cls.graph[name] = one_corridor.connections
        return cls.graph

    def evaluate_action(self, a_uav, instant_reward):

        # a_uav.steps_taken_current += 1
        # a_uav.steps_taken_all += 1

        if self.is_inside(a_uav.next_position):
            # reward = instant_reward + PENALTY_TIME
            # simplified reward
            move_forward = 1 if instant_reward > 0 else -1
            reward = move_forward * REWARD_POSITIVE_STEP + PENALTY_TIME
        else:
            cross_flag, last_corridor_residual_vec = (
                self.line_cross_des_plane_n_how_much(inside_point=a_uav.position,
                                                     outside_point=a_uav.next_position))
            if cross_flag:
                reward = REWARD_REACH
                a_uav.status = 'won'
                path = a_uav.enroute['path']
                if path[-1] == self.name:
                    pass
                else:
                    path_index = path.index(self.name)
                    a_uav.enroute['current'] = path[path_index + 1]

                # if self.name == a_uav.enroute['current']:
                #     a_uav.status = 'won'
                # else:
                #     path = a_uav.enroute['path']
                #     path_index = path.index(self.name)
                #     a_uav.enroute['current'] = path[path_index + 1]
            else:
                a_uav.outside_counter += 1

                reward = PENALTY_TIME + PENALTY_BREACH
                if a_uav.outside_counter > BREACH_TOLERANCE:
                    a_uav.status = 'breached'

            # a_uav.outside_counter += 1

        return reward


class RectangleCorridor(Corridor, DirectionalRectangle):
    def __init__(self,
                 anchor_point,
                 direction_rad,
                 length,
                 width,
                 name,
                 connections):
        Corridor.__init__(self, name, connections)
        self.direction_rad = direction_rad
        direction = np.array([math.cos(direction_rad), math.sin(direction_rad)])
        DirectionalRectangle.__init__(self, anchor_point, direction, length, width)
        self.shape_type = [1, 0]

    def evaluate_action(self, a_uav):
        projected_distance = np.dot(a_uav.next_position - a_uav.position, self.direction)
        reward = super().evaluate_action(a_uav, instant_reward=projected_distance)
        return reward

    def render_self(self, surf):
        gfxdraw.line(surf,
                     int(OFFSET_x + self.down_left[0] * SCALE),
                     int(OFFSET_y + self.down_left[1] * SCALE),
                     int(OFFSET_x + self.down_right[0] * SCALE),
                     int(OFFSET_y + self.down_right[1] * SCALE),
                     BLUE)
        # draw destination line
        gfxdraw.line(surf,
                     int(OFFSET_x + self.up_left[0] * SCALE),
                     int(OFFSET_y + self.up_left[1] * SCALE),
                     int(OFFSET_x + self.up_right[0] * SCALE),
                     int(OFFSET_y + self.up_right[1] * SCALE),
                     RED)
        gfxdraw.line(surf,
                     int(OFFSET_x + self.down_left[0] * SCALE),
                     int(OFFSET_y + self.down_left[1] * SCALE),
                     int(OFFSET_x + self.up_left[0] * SCALE),
                     int(OFFSET_y + self.up_left[1] * SCALE),
                     BLACK)
        gfxdraw.line(surf,
                     int(OFFSET_x + self.down_right[0] * SCALE),
                     int(OFFSET_y + self.down_right[1] * SCALE),
                     int(OFFSET_x + self.up_right[0] * SCALE),
                     int(OFFSET_y + self.up_right[1] * SCALE),
                     BLACK)

    @lru_cache(maxsize=8)
    def report(self):
        # 2+[1 +1 +1]
        corridor_status = self.shape_type + [self.length, np.atan2(self.direction[1], self.direction[0]), 0]
        return corridor_status

    def release_uav(self, location_assigned):
        while 1:
            rv = random.random() if location_assigned is None else location_assigned
            if 0 < rv < 1:
                break
        relative_position = self.down_right * rv + self.down_left * (1 - rv) + 0.2 * self.direction
        return relative_position + self.anchor_point


class directionalPartialAnnulusCorridor(Corridor, directionalPartialAnnulus):
    def __init__(self,
                 anchor_point: np.ndarray,
                 major_radius: float,
                 minor_radius: float,
                 begin_rad: float,
                 end_rad: float,
                 name=None,
                 connections=None):
        Corridor.__init__(self, name, connections)
        directionalPartialAnnulus.__init__(self, anchor_point, major_radius, minor_radius, begin_rad=begin_rad,
                                           end_rad=end_rad)

        assert -np.pi <= self.begin_rad <= np.pi, "Error, begin radian needs to be in [-pi,pi]"

        self.shape_type = [0, 1]

    def evaluate_action(self, a_uav):
        act_rad_diff_one_step = self.line_to_proj_rad(a_uav.position, a_uav.next_position)
        directional_rad_diff_one_step = act_rad_diff_one_step * self.counter_clockwise
        reward = super().evaluate_action(a_uav, instant_reward=directional_rad_diff_one_step * self.major_radius)
        return reward

    @lru_cache(maxsize=8)
    def report(self, points=3):
        # 2+ [1,1,1,1] +4*3=18
        corridor_status = self.shape_type + [self.major_radius, self.minor_radius, self.begin_rad, self.end_rad]

        inner = self.major_radius - self.minor_radius
        outer = self.major_radius + self.minor_radius
        point_state = []
        for theta in np.linspace(self.begin_rad, self.end_rad, num=points):
            point_state+=list(polar_to_cartesian(inner, theta)) + list(polar_to_cartesian(outer, theta))
        return corridor_status + point_state


    def release_uav(self, location_assigned):
        rv = random.random() if location_assigned is None else location_assigned
        relative_position = rv * self.begin_inner + (1 - rv) * self.begin_outter
        return relative_position + self.anchor_point + self.begin_dir * 0.2


    def render_self(self, surf):
        gfxdraw.arc(surf,
                    int(OFFSET_x),
                    int(OFFSET_y),
                    int((self.major_radius + self.minor_radius) * SCALE),
                    self.draw_bgein,
                    self.draw_end,
                    GRAY)
        gfxdraw.arc(surf,
                    int(OFFSET_x),
                    int(OFFSET_y),
                    int((self.major_radius - self.minor_radius) * SCALE),
                    self.draw_bgein,
                    self.draw_end,
                    GRAY)
        # draw begin line
        gfxdraw.line(surf,
                     int(OFFSET_x + self.begin_outter[0] * SCALE),
                     int(OFFSET_y + self.begin_outter[1] * SCALE),
                     int(OFFSET_x + self.begin_inner[0] * SCALE),
                     int(OFFSET_y + self.begin_inner[1] * SCALE),
                     BLUE)
        # draw destination line
        gfxdraw.line(surf,
                     int(OFFSET_x + self.end_outter[0] * SCALE),
                     int(OFFSET_y + self.end_outter[1] * SCALE),
                     int(OFFSET_x + self.end_inner[0] * SCALE),
                     int(OFFSET_y + self.end_inner[1] * SCALE),
                     RED)
