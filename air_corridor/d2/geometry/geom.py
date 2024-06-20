from air_corridor.tools._descriptor import Direction, Position, PositiveNumber
from air_corridor.tools._geometric import Geometric2D
from air_corridor.tools.util import *


# in 2d only cylinder has the concept of direction.
# circular ring or partial circular ring donot have it.

class Point2D(Geometric2D):
    anchor_point = Position(2)

    def __init__(self, anchor_point=np.array([0, 0])):
        self.anchor_point = anchor_point

    def distance_center_to_point(self, point):
        return np.linalg.norm(point - self.anchor_point)

    def distance_object_to_point(self, point):
        return self.distance_center_to_point(point)

    def point_relative_center_position(self, point):
        return point - self.anchor_point

    def cartesian_to_polar(self, point):
        """ 1) convert to relative position and 2) then convert to polar coordinate """
        relative_position = self.point_relative_center_position(point)
        return cartesian_to_polar_or_spherical(relative_position)

    def is_inside(self, point):
        return np.allclose(point, self.anchor_point)

    def report_state(self):
        return self.anchor_point.tolist()

    def convert_2_polar(self, point):
        r = self.distance_center_to_point(point)
        theta = np.arctan2(point[1], point[0])
        return r, theta

    def convert_vec_2_polar(self, point, velocity):
        r1, theta1 = self.convert_2_polar(point)
        r2, theta2 = self.convert_2_polar(point + velocity)
        return r2 - r1, theta2 - theta1

    def counter_clockwise_angle_between_from_p1_to_p2(self, point1, point2):
        # provide counter-clock radian from point1 to point2
        relative_position1 = self.point_relative_center_position(point1)
        relative_position2 = self.point_relative_center_position(point2)
        # dot = np.dot(relative_position1, relative_position2)
        # det = np.linalg.det([relative_position1, relative_position2])
        # return np.arctan2(det, dot)

        return counter_clockwise_radian(relative_position1, relative_position2)

    def line_to_proj_rad(self, p1, p2):
        return project_line_2_rad(self.point_relative_center_position(p1), self.point_relative_center_position(p2))

    # @staticmethod
    # def line_to_proj_rad(p1, p2):
    #     v1 = p1[:2]
    #     v2 = p2[:2]
    #     dot = np.dot(v1, v2)
    #     det = np.linalg.det([v1, v2])
    #     return np.arctan2(det, dot)


# class LineSegment(Point2D):
#     direction = Direction(2)
#     width = PositiveNumber()
#     width = PositiveNumber()
#
#     def __init__(self, anchor_point=np.array([0, 0]), direction=np.array([1, 0]), width=1):
#         super().__init__(anchor_point)
#         self.direction = direction
#         self.width = width
#         angle = np.arctan2(self.direction[1], self.direction[0])
#         angle += np.pi / 2
#         self.orthogonal_direction = np.array([math.cos(angle), math.sin(angle)])
#
#     def is_cross(self, point_last, point_current):
#         line_end1 = self.anchor_point + self.width / 2 * self.orthogonal_direction
#         line_end2 = self.anchor_point - self.width / 2 * self.orthogonal_direction
#         return is_line_line_intersect(point_last, point_current, line_end1, line_end2)


class DirectionalRectangle(Point2D):
    direction = Direction(2)
    width = PositiveNumber()
    length = PositiveNumber()

    def __init__(self, anchor_point=np.array([0, 0]), direction=np.array([0, 1]), length=10, width=4):
        super().__init__(anchor_point)
        self.direction = direction
        self.length = length
        self.width = width
        self.shape_type = [1, 0]

        self.half_width = 2
        # length_vec = np.array([self.length * math.cos(self.direction), self.length * math.sin(self.direction)])
        # width_vec = np.array(
        #     [self.width * math.cos(self.direction - np.pi / 2), self.width * math.sin(self.direction - np.pi / 2)])
        length_vec = self.length / 2 * self.direction
        width_vec = self.half_width * np.dot(np.array([[0, -1], [1, 0]]), self.direction)

        self.up_right = +length_vec + width_vec
        self.up_left = +length_vec - width_vec
        self.down_right = -length_vec + width_vec
        self.down_left = -length_vec - width_vec

    def __repr__(self):
        return f"Rectangle(anchor_point={self.anchor_point.tolist()}, " \
               f"direction={self.direction.tolist()}, " \
               f"length={self.length}," \
               f"width={self.width})"

    def distance_object_to_point(self, point):
        # y: perpendicular to the line; x: parallel/projected to the line
        distance_y = distance_perpendicular_line_point(self.anchor_point, self.direction, point) - self.width / 2
        distance_x = np.abs(
            distance_signed_parallel_line_point(self.anchor_point, self.direction, point)) - self.length / 2
        return max(distance_x, distance_y)

    def is_inside(self, point):
        return True if self.distance_object_to_point(point) <= TRIVIAL_TOLERANCE else False

    def line_cross_des_plane_n_how_much(self, inside_point, outside_point):
        if self.is_inside(self.point_relative_center_position(outside_point)):
            raise Exception("outside point is not outside")
        return is_line_line_intersect(self.point_relative_center_position(inside_point),
                                      self.point_relative_center_position(outside_point),
                                      self.up_left,
                                      self.up_right)


# class Circle(Point):
#     radius = PositiveNumber()
#
#     def __init__(self, anchor_point=np.array([0, 0]), radius=1):
#         super().__init__(anchor_point)
#         self.radius = radius
#
#     def __repr__(self):
#         return f"Circle(anchor_point={self.anchor_point.tolist()}, " \
#                f"radius={self.radius})"
#
#     def distance_to_point(self, point):
#         return max(0, np.abs(super().distance_to_point(point) - self.radius))
#
#     def isinside(self, point):
#         return True if self.distance_to_point(point) == 0 else False
#
#     def projection_from_point(self, point):
#         relative_point_position = self.position_point_to_shape(point)
#         rad = vector2d_to_angle(relative_point_position)
#         if self.isinside(point) == True:
#             projected_point = point
#         else:
#             projected_point = polar_to_cartesian(self.radius, rad)
#         return projected_point, rad


class Annulus(Point2D):
    """
    full circular ring without degree constrain
    """

    major_radius = PositiveNumber()
    minor_radius = PositiveNumber()

    def __init__(self, anchor_point=np.array([0, 0]), major_radius=2, minor_radius=1):
        super().__init__(anchor_point)
        self.major_radius = major_radius
        self.minor_radius = minor_radius

    def __repr__(self):
        return f"Annulus(anchor_point={self.anchor_point.tolist()}, " \
               f"major_radius={self.major_radius}" \
               f"major_radius={self.minor_radius})"

    def status_object_to_point(self, point):
        # within inner circle
        # in annulus
        # outside outer circle
        ds = self.distance_to_inner_outer_circles(point)
        if ds[0] < - TRIVIAL_TOLERANCE:
            return "inner"
        elif ds[1] < TRIVIAL_TOLERANCE:
            return "in"
        else:
            return "outer"

    def distance_to_inner_outer_circles(self, point):
        distance = super().distance_center_to_point(point)
        return distance - (self.major_radius - self.minor_radius), distance - (self.major_radius + self.minor_radius)

    def is_inside(self, point):
        return True if self.status_object_to_point(point) == "in" else False


class directionalPartialAnnulus(Annulus):
    """
        partial circular ring with degree range.
        range for begin radian is [-pi,pi]
        range for end radian is [-3pi,3pi]
        range for whole_diff is [-2pi,2pi]
    """

    # begin_rad = PositiveNumber()
    # end_rad = PositiveNumber()

    def __init__(self,
                 anchor_point=np.array([0, 0]),
                 major_radius=2,
                 minor_radius=1,
                 begin_rad=0,
                 end_rad=np.pi):
        super().__init__(anchor_point, major_radius, minor_radius)
        self.begin_rad = begin_rad
        self.end_rad = end_rad
        self.counter_clockwise = 1 if end_rad > begin_rad else -1
        self.whole_diff = end_rad - begin_rad

        self.begin_outter = np.array([math.cos(begin_rad) * (major_radius + minor_radius),
                                      math.sin(begin_rad) * (major_radius + minor_radius)])
        self.begin_inner = np.array([math.cos(begin_rad) * (major_radius - minor_radius),
                                     math.sin(begin_rad) * (major_radius - minor_radius)])
        self.begin_dir = self.counter_clockwise * np.array([-math.sin(begin_rad), math.cos(begin_rad)])
        self.end_outter = np.array([math.cos(end_rad) * (major_radius + minor_radius),
                                    math.sin(end_rad) * (major_radius + minor_radius)])
        self.end_inner = np.array([math.cos(end_rad) * (major_radius - minor_radius),
                                   math.sin(end_rad) * (major_radius - minor_radius)])

        self.shapeType = [0, 1]

        if self.counter_clockwise == 1:
            self.draw_bgein = int(math.degrees(self.begin_rad))
            self.draw_end = int(math.degrees(self.end_rad))
        elif self.counter_clockwise == -1:
            self.draw_bgein = int(math.degrees(self.end_rad))
            self.draw_end = int(math.degrees(self.begin_rad))
        # self.draw_bgein = int(math.degrees(self.begin_rad))
        # self.draw_end = int(math.degrees(self.end_rad))

    def __repr__(self):
        return f"DirectionalPartialAnnulus(anchor_point={self.anchor_point.tolist()}, " \
               f"major_radius={self.major_radius}" \
               f"minor_radius={self.minor_radius}" \
               f"begin_rad={self.begin_rad}" \
               f"end_rad={self.end_rad})"

    def radian_progress(self, point_current, point_last):
        rad_current = vector_to_2d_rad(self.point_relative_center_position(point_current))
        diff_current = radian_difference(self.begin_rad, rad_current, self.counter_clockwise, output_range='[0, 2pi]')
        if diff_current >= self.whole_diff:
            rad_last = vector_to_2d_rad(self.point_relative_center_position(point_last))
            diff_last = radian_difference(self.begin_rad, rad_last, self.counter_clockwise,
                                          output_range='[0, 2pi]')

        return diff_current, diff_current / self.whole_diff

    #
    # def radian_progress(self, ):
    #     pass
    def _is_radian_in(self, radian):
        if self.counter_clockwise == 1:
            if self.end_rad >= radian >= self.begin_rad \
                    or self.end_rad >= radian + self.counter_clockwise * 2 * np.pi >= self.begin_rad:
                return True
            else:
                return False
        elif self.counter_clockwise == -1:
            if self.end_rad <= radian <= self.begin_rad \
                    or self.end_rad <= radian + self.counter_clockwise * 2 * np.pi <= self.begin_rad:
                return True
            else:
                return False

    def line_cross_des_plane_n_how_much(self, inside_point, outside_point):
        if self.is_inside(self.point_relative_center_position(outside_point)):
            raise Exception("outside point is not outside")
        return is_line_line_intersect(self.point_relative_center_position(inside_point),
                                      self.point_relative_center_position(outside_point),
                                      self.end_inner,
                                      self.end_outter)

    def is_inside(self, point, rad=0):
        rad_current = vector_to_2d_rad(self.point_relative_center_position(point))
        radius_in = self.status_object_to_point(point) == 'in'
        if self._is_radian_in(rad_current) and radius_in:
            return True
        return False


        # closest_on_torus_circle=np.array([self.major_radius*math.cos(torus_rad),self.major_radius*math.sin(torus_rad),0])

        # return True if self.status_object_to_point(point) == "in" and self.is_rad_in(point) else False

    # def report_state(self):
    #     return self.shape_type+[ self.major_radius, self.begin_rad, self.end_rad]
