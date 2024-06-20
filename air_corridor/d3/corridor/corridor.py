from air_corridor.d3.geometry.geom3d import Cylinder, newTorus
from air_corridor.tools.util import *

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
    # consider_next_corridor = False
    reduce_space = True

    def __init__(self, name, connections, ):
        # Initialize corridor properties
        self.name = name
        self.connections = connections

    @classmethod
    def convert2graph(cls, corridors):
        cls.graph = {}
        for name, one_corridor in corridors.items():
            cls.graph[name] = one_corridor.connections
        return cls.graph

    def evaluate_action(self, a_uav, alignment=1, crossed=False):

        '''
        alignement [-1,1]
        corssed [False, True]
        '''
        reward = PENALTY_TIME

        # all specified corridors follow dual inheritance, is_inside is from geom
        # a=self.is_inside(a_uav.next_position)
        flag, ststus = self.is_inside(a_uav.next_position)
        if flag:
            # reward += aligned * REWARD_POSITIVE_STEP
            pass
        elif crossed:
            '''
            whether consider the next corridor, considering two corridors' reward as one episode
            reward only make sense during training, here is trained with considering two corridors.
            '''
            path = a_uav.enroute['path']

            if path[-1] == self.name:
                a_uav.status = 'won'
                reward += REWARD_REACH
            else:
                reward += REWARD_INTERMEDIA
                path_index = path.index(self.name)
                a_uav.enroute['current'] = path[path_index + 1]
            reward += alignment * REACH_ALIGNMENT

        else:
            # breach boundary
            # a_uav.outside_counter += 1
            # reward += PENALTY_BREACH
            # if a_uav.outside_counter > BREACH_TOLERANCE:
            #     a_uav.status = 'breached'
            reward += PENALTY_BREACH
            a_uav.status = ststus
        return reward


class CylinderCorridor(Corridor, Cylinder):
    def __init__(self,
                 anchor_point,
                 length,
                 width,
                 name,
                 connections,
                 orientation_rad=None,
                 orientation_vec=None, ):
        Corridor.__init__(self, name, connections)
        self.radius = width / 2
        Cylinder.__init__(self,
                          anchor_point=anchor_point,
                          orientation_vec=orientation_vec,
                          orientation_rad=orientation_rad,
                          length=length,
                          radius=self.radius)
        self.shapeType = [0, 0, 1, 0]

    def evaluate_action(self, a_uav):
        alignment = 0
        cross_flag = self.endCirclePlane.cross_circle_plane(line_start=a_uav.position,
                                                            line_end=a_uav.next_position)
        if cross_flag:
            alignment = align_measure(end=a_uav.next_position, start=a_uav.position, direction=self.orientation_vec)
        reward = super().evaluate_action(a_uav, alignment=alignment, crossed=cross_flag)

        return reward

    def render_self(self, ax):
        def cylinder(r, h, theta_res=100, z_res=100):
            theta = np.linspace(0, 2 * np.pi, theta_res)
            z = np.linspace(0, h, z_res)
            theta, z = np.meshgrid(theta, z)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            return x, y, z

        Xc, Yc, Zc = cylinder(self.radius, self.length)
        x_rot, y_rot, z_rot = [], [], []
        for a, b, c in zip(Xc, Yc, Zc):
            x_p, y_p, z_p = np.dot(self.rotation_matrix, np.array([a, b, c]))
            x_rot.append(x_p)
            y_rot.append(y_p)
            z_rot.append(z_p)
        ax.plot_surface(np.array(x_rot), np.array(y_rot), np.array(z_rot), edgecolor='royalblue', lw=0.1, rstride=20,
                        cstride=8, alpha=0.3)

    @lru_cache(maxsize=8)
    def report(self, base=None):
        # 7+2+4=13,
        # last two 0 are padding, keeping the format the same as
        common_part = super().report(base=base, reduce_space=self.reduce_space)

        corridor_status = common_part + self.shapeType + [self.length, self.radius] + [0] * 4

        if any(np.isnan(corridor_status)):
            print('nan in cylinder')
            input("Press Enter to continue...")
        return corridor_status

    def release_uav(self, plane_offset_assigned):
        plane_offset = self.x * plane_offset_assigned[0] + self.y * plane_offset_assigned[1]
        direction_offset = (0.2 - self.length / 2) * self.orientation_vec
        return self.anchor_point + plane_offset + direction_offset


class DirectionalPartialTorusCorridor(Corridor, newTorus):
    def __init__(self,
                 anchor_point: np.ndarray,
                 major_radius: float,
                 minor_radius: float,
                 begin_rad: float,
                 end_rad: float,
                 orientation_rad=None,
                 orientation_vec=None,
                 name=None,
                 connections=None,
                 ):
        Corridor.__init__(self, name, connections)
        newTorus.__init__(self,
                          anchor_point=anchor_point,
                          orientation_vec=orientation_vec,
                          orientation_rad=orientation_rad,
                          major_radius=major_radius,
                          minor_radius=minor_radius,
                          begin_rad=begin_rad,
                          end_rad=end_rad)
        assert -np.pi <= self.begin_rad <= np.pi, "Error, begin radian needs to be in [-pi,pi]"
        self.shapeType = [0, 0, 0, 1]

    def evaluate_action(self, a_uav):
        alignment = 0
        cross_flag = self.endCirclePlane.cross_circle_plane(line_start=a_uav.position,
                                                            line_end=a_uav.next_position)
        if cross_flag:
            positive_direction = self.determine_positive_direction(a_uav.position)
            alignment = align_measure(end=a_uav.next_position, start=a_uav.position, direction=positive_direction)
        reward = super().evaluate_action(a_uav, alignment=alignment, crossed=cross_flag)
        return reward

    @lru_cache(maxsize=8)
    def report(self, base=None):
        common_part = super().report(base=base, reduce_space=self.reduce_space)
        corridor_status = common_part + self.shapeType + [self.major_radius, self.minor_radius,
                                                          self.major_radius + self.minor_radius,
                                                          self.major_radius - self.minor_radius]
        # if self.reduce_space:
        #     radian_range = [np.pi / 2 - (self.end_rad - self.begin_rad), np.pi / 2]
        # else:
        #     radian_range = [self.begin_rad, self.end_rad]
        # radian_range = [np.pi / 2 - (self.end_rad - self.begin_rad), np.pi / 2, self.begin_rad, self.end_rad]
        radian_range = [np.pi / 2 - (self.end_rad - self.begin_rad), np.pi / 2]#, self.begin_rad, self.end_rad]
        if any(np.isnan(corridor_status)):
            print('nan in torus')
            input("Press Enter to continue...")

        return corridor_status + radian_range

    def release_uav(self, plane_offset_assigned):
        plane_offset = self.beginCirclePlane.x * plane_offset_assigned[0] + \
                       self.beginCirclePlane.y * plane_offset_assigned[1]
        direction_offset = 0.2 * self.beginCirclePlane.orientation_vec

        return self.beginCirclePlane.anchor_point + plane_offset + direction_offset

    def render_self(self, ax):
        def torus(R, r, R_res=100, r_res=100):
            u = np.linspace(0, 1.5 * np.pi, R_res)
            v = np.linspace(0, 2 * np.pi, r_res)
            u, v = np.meshgrid(u, v)
            x = (R + r * np.cos(v)) * np.cos(u)
            y = (R + r * np.cos(v)) * np.sin(u)
            z = r * np.sin(v)
            return x, y, z

        Xc, Yc, Zc = torus(self.major_radius, self.minor_radius)
        x_rot, y_rot, z_rot = [], [], []
        for a, b, c in zip(Xc, Yc, Zc):
            x_p, y_p, z_p = np.dot(self.rotation_matrix, np.array([a, b, c]))
            x_rot.append(x_p)
            y_rot.append(y_p)
            z_rot.append(z_p)
        ax.plot_surface(np.array(x_rot), np.array(y_rot), np.array(z_rot), edgecolor='royalblue', lw=0.1, rstride=20,
                        cstride=8, alpha=0.3)
