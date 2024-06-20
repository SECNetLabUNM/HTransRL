from air_corridor.tools._descriptor import Direction, Position, PositiveNumber
from air_corridor.tools._geometric import Geometric3D
from air_corridor.tools.util import *


# @abstractmethod
# def project_point(self, point):
# pass
class Point3D(Geometric3D):
    anchor_point = Position(3)
    orientation_vec = Direction(3)

    def __init__(self, anchor_point=np.array([0, 0, 0]), orientation_vec=None, orientation_rad=None):
        self.anchor_point = anchor_point
        # can be considered as projected z

        assert orientation_rad is not None or orientation_vec is not None
        if orientation_vec is None:
            # theta, phi = orientation_rad
            # self.orientation_rad = orientation_rad
            self.orientation_vec = polar_to_unit_normal(orientation_rad)
        else:
            self.orientation_vec = orientation_vec
        if orientation_rad is None:
            self.orientation_rad = cartesian_to_polar_or_spherical(self.orientation_vec)[1:]
        else:
            self.orientation_rad = orientation_rad

        # projected x from base x
        self.x = rotate(vec=X_UNIT, fromVec=Z_UNIT, toVec=self.orientation_vec)

        # projected y from base x
        self.y = np.cross(self.orientation_vec, self.x)
        self.rotation_matrix_to_remote = vec2vec_rotation(Z_UNIT, self.orientation_vec)
        self.rotation_matrix_to_base = vec2vec_rotation(self.orientation_vec, Z_UNIT)

    def rotate_to_base(self, vec):
        return np.dot(self.rotation_matrix_to_base, vec)

    def rotate_to_remote(self, vec):
        return np.dot(self.rotation_matrix_to_remote, vec)

    def project_to_base(self, point):
        vec = self.point_relative_center_position(point)
        return self.rotate_to_base(vec)

    def project_to_remote(self, point):
        # vec = self.point_relative_center_position(point)
        return self.anchor_point + self.rotate_to_remote(point)

    def point_relative_center_position(self, point):
        return point - self.anchor_point

    # @lru_cache(maxsize=2)
    def convert_2_polar(self, point, reduce_space=True):
        if reduce_space:
            point = self.project_to_base(point)
        else:
            point = self.point_relative_center_position(point)
        r, theta, phi = cartesian_to_polar_or_spherical(point)
        return r, theta, phi

    def convert_vec_2_polar(self, point, velocity, reduce_space=True):
        r1, theta1, phi1 = self.convert_2_polar(point, reduce_space)
        r2, theta2, phi2 = self.convert_2_polar(point + velocity, reduce_space)
        return r2 - r1, theta2 - theta1, phi2 - phi1,

    def cartesian_to_polar(self, point):
        """ 1) convert to relative position and 2) then convert to polar coordinate """
        relative_position = self.point_relative_center_position(point)
        return cartesian_to_polar_or_spherical(relative_position)

    def is_inside(self, point):
        return np.allclose(point, self.anchor_point)

    def report(self, base=None, reduce_space=True):
        # only fit reduce_space scenario
        '''
         list(position_diff_on_base) + [r1, t1, p1]
        position_diff_on_base was using the position diff without rotation in the last version
        '''
        if True:
            if reduce_space:
                if self != base:
                    ori_on_base = base.rotate_to_base(self.orientation_vec)
                    _, theta, phi = cartesian_to_polar_or_spherical(ori_on_base)
                    position_diff_on_base = base.project_to_base(self.anchor_point - base.anchor_point)
                    r1, t1, p1 = cartesian_to_polar_or_spherical(position_diff_on_base)
                    status = list(position_diff_on_base) + [r1, t1, p1] + \
                             list(ori_on_base) + [theta, phi]
                else:
                    status = [0, 0, 0, 0, 0, 0] + list(Z_UNIT) + [0, 0]
            else:
                _, theta, phi = cartesian_to_polar_or_spherical(self.orientation_vec)
                status = [0, 0, 0, 0, 0, 0] + list(self.orientation_vec) + [theta, phi]
        else:
            if self != base:
                ori_based_on_former = base.rotate_to_base(self.orientation_vec)
                _, theta, phi = cartesian_to_polar_or_spherical(ori_based_on_former)
                status = list(self.anchor_point - base.anchor_point) + \
                         list(ori_based_on_former) + \
                         [theta, phi] + list(self.orientation_vec) + list(self.orientation_rad)
            else:
                status = [0, 0, 0] + list(Z_UNIT) + [0, 0] + list(self.orientation_vec) + list(self.orientation_rad)
        return status

    def anchor_alignment(self, off_set):
        self.anchor_point = self.anchor_point - off_set


class Sphere(Point3D):
    radius = PositiveNumber()

    def __init__(self, anchor_point, orientation_vec, radius):
        super().__init__(anchor_point, orientation_vec)
        self.radius = radius

    def __repr__(self):
        return f"Sphere(center={self.anchor_point.tolist()}, " \
               f"radius={self.radius})"

    def distance_object_to_point(self, point):
        return super().point_relative_center_position(point) - self.radius

    def is_inside(self, point, inflate=0):
        return True if self.distance_object_to_point(point) < TRIVIAL_TOLERANCE + inflate else False


class Cylinder(Point3D):
    radius = PositiveNumber()
    length = PositiveNumber()

    def __init__(self, anchor_point, orientation_vec, orientation_rad, radius, length=1):
        super().__init__(anchor_point,
                         orientation_vec=orientation_vec,
                         orientation_rad=orientation_rad)
        self.radius = radius
        self.length = length

        self.endCirclePlane = (
            Circle(
                anchor_point=self.anchor_point + self.orientation_vec * self.length / 2,
                orientation_vec=self.orientation_vec,
                radius=radius
            )
        )

    def determine_rotation_with_next_torus(self, torus_ori=np.array([1, 0, 0])):
        '''
        this step is to simplify the state for cylinder, if there are at least 2 corridors in the state.
        align the following torus orientation to [1,0,0]
        '''
        torus_ori_base = self.rotate_to_base(torus_ori)
        if list(torus_ori_base) != [1, 0, 0]:
            rotate_torus_ori_2_std = vec2vec_rotation(torus_ori_base, np.array([1, 0, 0]))
            rotate_std_2_torus_ori = vec2vec_rotation(np.array([1, 0, 0]), torus_ori_base)

            self.rotation_matrix_to_base = np.dot(rotate_torus_ori_2_std, self.rotation_matrix_to_base)
            self.rotation_matrix_to_remote = np.dot(self.rotation_matrix_to_remote, rotate_std_2_torus_ori)

    #     self.rotate_cylinder_to_base = self.rotation_matrix_to_base
    #     self.rotate_cylinder_to_remote = self.rotation_matrix_to_remote
    #
    # def determine_rotation_with_next_torus(self, torus_ori=np.array([1, 0, 0])):
    #     '''
    #     this step is to simplify the state for cylinder, if there are at least 2 corridors in the state.
    #     align the following torus orientation to [1,0,0]
    #     '''
    #     if torus_ori != np.array([1, 0, 0]):
    #         rotate_torus_ori_2_std = vec2vec_rotation(torus_ori, np.array([1, 0, 0]))
    #         rotate_std_2_torus_ori = vec2vec_rotation(np.array([1, 0, 0]), torus_ori)
    #
    #         self.rotate_cylinder_to_base = np.dot(rotate_torus_ori_2_std, self.rotation_matrix_to_base)
    #         self.rotate_cylinder_to_remote = np.dot(self.rotation_matrix_to_remote, rotate_std_2_torus_ori)
    #
    # def rotate_to_base(self, vec):
    #     return np.dot(self.rotate_cylinder_to_base, vec)
    #
    # def rotate_to_remote(self, vec):
    #     '''
    #     project action back to global coordination, since action is only a vector. rotation only in enough.
    #     '''
    #     return np.dot(self.rotate_cylinder_to_remote, vec)
    #
    # def project_to_base(self, point):
    #     '''
    #     simplify state space
    #     '''
    #     vec = self.point_relative_center_position(point)
    #     return self.rotate_to_base(vec)

    def __repr__(self):
        return f"Cylinder(anchor_point={self.anchor_point.tolist()}, " \
               f"orientation_vec={self.orientation_vec.tolist()}, " \
               f"radius={self.radius}," \
               f"length={self.length})"

    def distance_object_to_point(self, point):
        # y: perpendicular to the line; x: parallel/projected to the line
        distance_y = distance_perpendicular_line_point(self.anchor_point, self.orientation_vec, point) - self.radius
        distance_x = np.abs(
            distance_signed_parallel_line_point(self.anchor_point, self.orientation_vec, point)) - self.length / 2
        return max(distance_x, distance_y)

    def is_inside(self, point, inflate=0):
        if self.distance_object_to_point(point) <= TRIVIAL_TOLERANCE + inflate:
            return (True, None)
        else:
            return (False, 'breached_c')

    def line_cross_des_plane_n_how_much(self, inside_point, outside_point):
        if self.is_inside(self.point_relative_center_position(outside_point)):
            raise Exception("outside point is not outside")
        return is_line_line_intersect(self.point_relative_center_position(inside_point),
                                      self.point_relative_center_position(outside_point),
                                      self.up_left,
                                      self.up_right)

    def sample_a_point_within(self):
        if self.length < 0.8:
            return None
        diff_r, z, rad_tube = uniform_sample_circle(self.radius)
        point = (self.anchor_point +
                 self.radius * (self.x * math.cos(rad_tube) + self.y * math.sin(rad_tube)) +
                 self.orientation_vec * random.uniform(-self.length / 2 + 0.8, self.length / 2))

        return point

    def anchor_alignment(self, off_set):
        super().anchor_alignment(off_set)
        self.endCirclePlane.anchor_alignment(off_set)


class Circle(Point3D):
    def __init__(self, anchor_point, orientation_vec, radius):
        super().__init__(anchor_point, orientation_vec)
        self.radius = radius
        if any(np.isnan(self.x)) or any(np.isnan(self.y)):
            print(1)
            super().__init__(anchor_point, orientation_vec)

    def cross_circle_plane(self, line_start, line_end):
        return is_line_circle_intersect(line_start=line_start,
                                        line_end=line_end,
                                        anchor=self.anchor_point,
                                        direction=self.orientation_vec,
                                        radius=self.radius)

    def distance_object_to_point(self, point):
        pass

    def anchor_alignment(self, off_set):
        super().anchor_alignment(off_set)


class newTorus(Point3D):
    major_radius = PositiveNumber()
    minor_radius = PositiveNumber()

    def __init__(self,
                 anchor_point,
                 orientation_vec,
                 orientation_rad,
                 major_radius,
                 minor_radius,
                 begin_rad,
                 end_rad):
        super().__init__(anchor_point,
                         orientation_vec=orientation_vec,
                         orientation_rad=orientation_rad)
        self.begin_rad = begin_rad
        self.end_rad = end_rad

        self.beginCirclePlane = (
            Circle(
                anchor_point=self.anchor_point + major_radius * (
                        self.x * math.cos(begin_rad) + self.y * math.sin(begin_rad)),
                orientation_vec=(-self.x * math.sin(begin_rad) + self.y * math.cos(begin_rad)),
                radius=minor_radius
            )
        )
        self.endCirclePlane = (
            Circle(
                anchor_point=self.anchor_point + major_radius * (
                        self.x * math.cos(end_rad) + self.y * math.sin(end_rad)),
                orientation_vec=(-self.x * math.sin(end_rad) + self.y * math.cos(end_rad)),
                radius=minor_radius
            )
        )

        self.major_radius = major_radius
        self.minor_radius = minor_radius

        # attatch begin radian to 0 rad
        # self.rotate_xy_begin_to_x = o3d.geometry.get_rotation_matrix_from_axis_angle(-self.begin_rad * Z_UNIT)
        # self.rotate_xy_x_to_begin = o3d.geometry.get_rotation_matrix_from_axis_angle(+self.begin_rad * Z_UNIT)

        # attatch end radian to pi/2 rad
        self.rotate_end_rad_to_y_in_xy = o3d.geometry.get_rotation_matrix_from_axis_angle(
            (+np.pi / 2 - self.end_rad) * Z_UNIT)
        self.rotate_y_to_enx_rad_in_xy = o3d.geometry.get_rotation_matrix_from_axis_angle(
            (-np.pi / 2 + self.end_rad) * Z_UNIT)

        self.rotate_torus_to_base = np.dot(self.rotate_end_rad_to_y_in_xy, self.rotation_matrix_to_base)
        self.rotate_torus_to_remote = np.dot(self.rotation_matrix_to_remote, self.rotate_y_to_enx_rad_in_xy)

    def rotate_to_base(self, vec):
        return np.dot(self.rotate_torus_to_base, vec)

    def rotate_to_remote(self, vec):
        '''
        project action back to global coordination, since action is only a vector. rotation only in enough.
        '''
        return np.dot(self.rotate_torus_to_remote, vec)

    def project_to_base(self, point):
        '''
        simplify state space
        '''
        vec = self.point_relative_center_position(point)
        return self.rotate_to_base(vec)

    def __repr__(self):
        return f"Torus(center={self.anchor_point.tolist()}, " \
               f"orientation_vec={self.orientation_vec.tolist()}, " \
               f"major_radius={self.major_radius}, " \
               f"minor_radius={self.minor_radius}, " \
               f"begin_degree={self.begin_rad}, " \
               f"end_degree={self.end_rad})"

    def determine_positive_direction(self, point):
        '''
        out put the positive direction based on current position
        :param point:
        :return:
        '''
        vec_to_point = self.point_relative_center_position(point)
        orientation_vec = np.cross(self.orientation_vec, vec_to_point)

        return orientation_vec / np.linalg.norm(orientation_vec)

    def distance_object_to_point(self, point, consider_angle=False):
        '''
        1. Project the Point onto the Plane of the Circle
        2. Find the Closest Point on the Full Circle
        3. Check if the Closest Point is within the Quarter Circle Segment
        '''
        # Project the point onto the plane of the circle

        vec_to_point = self.point_relative_center_position(point)
        projection_on_plane = proj_to_plane(vec_to_point, self.orientation_vec)
        unit_projection = projection_on_plane / np.linalg.norm(projection_on_plane)

        # Closest point on the full circle
        closest_on_circle = self.anchor_point + self.major_radius * unit_projection
        signed_distance = np.linalg.norm(point - closest_on_circle) - self.minor_radius

        if not consider_angle:
            return signed_distance

        angle = np.arctan2(np.dot(closest_on_circle - self.anchor_point, self.y),
                           np.dot(closest_on_circle - self.anchor_point, self.x))
        degree_inside = self.is_degree_in(angle)

        return signed_distance, degree_inside

    def is_degree_in(self, angle):
        '''
        always incurs a lot of bugs
        range for self.begin_rad is [-np.pi,np.pi]
        range for self.end_rad for [-np.pi, 2+np.pi]

        :param angle:
        :return:
        '''

        assert self.end_rad > self.begin_rad

        while angle < self.begin_rad:
            angle += np.pi * 2

        if self.begin_rad <= angle <= self.end_rad:
            return True
        else:
            return False

    def is_inside(self, point, inflate=0):
        signed_distance, degree_inside = self.distance_object_to_point(point, consider_angle=True)
        status = []
        # if degree_inside and signed_distance <= TRIVIAL_TOLERANCE:
        #     return True
        # else:
        #     return False

        if not degree_inside:
            status.append('rad')
        if not signed_distance <= TRIVIAL_TOLERANCE + inflate:
            status.append('wall')
        if status:
            return False, f"breached_{'_t_'.join(status)}"
        else:
            return True, None

    def sample_a_point_within(self):
        '''
        randomly pick a point within for static obstacle
        torus_rad is the radian based on torus
        diff_r,z are generated by uniformly sampling the tube
        '''
        # torus_rad = random.random() * (self.end_rad - self.begin_rad) + self.begin_rad
        diff = self.end_rad - self.begin_rad
        if diff < 0.05479 * math.pi:  # 10'
            return None
        torus_rad = random.uniform(self.begin_rad, self.end_rad)
        diff_r, z, rad_tube = uniform_sample_circle(self.minor_radius)
        actual_radius = self.major_radius + diff_r
        point = np.array([actual_radius * math.cos(torus_rad), actual_radius * math.sin(torus_rad), z])
        return np.dot(self.rotation_matrix_to_remote, point) + self.anchor_point

    def anchor_alignment(self, off_set):
        super().anchor_alignment(off_set)
        self.beginCirclePlane.anchor_alignment(off_set)
        self.endCirclePlane.anchor_alignment(off_set)
