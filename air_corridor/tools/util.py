import json
import math
import random
from collections import deque
from functools import lru_cache

import open3d as o3d
import scipy
import torch

from .uti_consts import *


def is_inside_circle(point, circle_radius):
    """Check if a point is inside a circle with given radius."""
    return point[0] ** 2 + point[1] ** 2 <= circle_radius ** 2


def rotate_point(point, angle):
    """Rotate a point by a given angle around the origin."""
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    return rotation_matrix.dot(point)


def generate_hexagon_grid(circle_radius, side_length):
    """Generate a grid of hexagon centers and vertices within a given circle."""
    hexagon_centers = []
    hexagon_vertex_list = []

    # 60 degrees in radians and vertical spacing between centers
    angle = np.deg2rad(60)
    vertical_spacing = side_length * np.sqrt(3)

    # Determine the range for the grid
    grid_range_x = int(circle_radius / side_length) + 1
    grid_range_y = int(circle_radius / (vertical_spacing / 2)) + 1

    # Generate potential hexagon centers and their vertices
    for i in range(-grid_range_y, grid_range_y + 1):
        for j in range(-grid_range_x, grid_range_x + 1):
            # Offset for even and odd rows
            offset = 0 if i % 2 == 0 else side_length * 1.5
            center_x = j * 3 * side_length + offset
            center_y = i * vertical_spacing / 2
            center = (center_x, center_y)

            # Add the center if it's inside the circle
            if is_inside_circle(center, circle_radius):
                hexagon_centers.append(center)

            # Calculate vertices and add those that are inside the circle
            hex_vertices = [
                (center[0] + np.cos(k * angle) * side_length, center[1] + np.sin(k * angle) * side_length)
                for k in range(6)
            ]
            for vertex in hex_vertices:
                if is_inside_circle(vertex, circle_radius) and vertex not in hexagon_vertex_list:
                    can_add_vertex = True
                    for exist in hexagon_vertex_list:
                        if np.linalg.norm(np.array(vertex) - np.array(exist)) < (side_length / 2):
                            can_add_vertex = False
                            break
                    if can_add_vertex:
                        hexagon_vertex_list.append(vertex)

    return hexagon_centers + hexagon_vertex_list


def generate_random_hexagon_grid(circle_radius, side_length):
    centers_vertices = generate_hexagon_grid(circle_radius, side_length)

    # Choose a random angle between 0 and 360 degrees (in radians)
    random_angle = np.deg2rad(random.uniform(0, 360))

    # Apply rotation to each center and vertex
    rotated_centers_vertices = [rotate_point(center, random_angle) for center in centers_vertices]

    return rotated_centers_vertices


def uniform_sample_circle(radius):
    theta = random.uniform(0, 2 * math.pi)
    u = random.uniform(0, 1)
    r = radius * math.sqrt(u)
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return x, y, math.atan2(y, x)


def index_offset(lst, target, offset):
    n = len(lst)
    i = lst.index(target)
    j = i + offset
    if j > n - 1:
        return None
    else:
        return lst[j]


def random_unit(dim):
    if dim == 3:
        vec = np.random.random(dim)
    if dim == 2:
        vec = np.array(list(np.random.random(2)) + [0])
    return vec / np.linalg.norm(vec)


# def save_init_params(**kwargs,name):
#     with open(f"{kwargs['dir']}/net_params.json", 'w') as f:
#         kwargs['writer'] = None
#         kwargs['logger'] = None
#         json.dump(kwargs, f, indent=4)

def save_init_params(name, **kwargs):
    with open(f"{kwargs['dir']}/{name}.json", 'w') as f:
        try:
            kwargs['writer'] = None
            kwargs['logger'] = None
        except:
            pass
        json.dump(kwargs, f, indent=4)


def load_init_params(name, dir):
    with open(f"{dir}/{name}.json", 'r') as f:
        params = json.load(f)
    return params


def get_variable_name(var):
    for name in globals():
        if id(var) == id(globals()[name]):
            return name


def nan_recoding(log, variable, variable_name):
    nan_mask = torch.isnan(variable)
    if nan_mask.sum() > 0:
        if variable.sum() > 0:
            log.info(f"{variable_name} has nan: {variable}")


def duplicate_and_shuffle(lst, n, seed=0):
    random.seed(seed)
    while len(lst) <= n:
        lst.extend(lst)

    random.shuffle(lst)

    return lst[:n]


def distribute_evenly_on_line(line_length, min_distance, num_points):
    # Calculate the number of segments based on minimum distance
    num_segments = int(np.floor(line_length // min_distance))
    if num_points > num_segments:
        raise ValueError("Can't fit the given number of points with the specified minimum distance.")

    # Choose `num_points` segments randomly from `num_segments`
    chosen_segments = random.sample(range(num_segments), num_points)

    # Sort segments for ascending point placement
    chosen_segments.sort()

    points = []
    for segment in chosen_segments:
        # Generate a random point within the segment
        point = random.uniform(segment * min_distance, (segment + 1) * min_distance)
        points.append(point)
    points = list(map(lambda x: x / line_length, points))
    return points


def distribute_evenly_within_circle(radius, min_distance, num_points, mode='hexagon'):
    # Calculate the number of segments based on minimum distance
    if mode == 'hexagon':
        points = generate_random_hexagon_grid(radius, 0.65)
        random.shuffle(points)
        return points[:num_points]
    elif mode == 'circle':
        place_radius = radius - 0.5
        rad_base = np.random.random() * np.pi * 2
        points = []
        rad_diff = 2 * np.pi / num_points
        for i in range(num_points):
            rad_point = rad_base + i * rad_diff
            points.append(place_radius * np.array([np.cos(rad_point), np.sin(rad_point)]))
        if num_points > 1 and np.linalg.norm(points[0] - points[1]) < min_distance:
            raise ValueError('cannot meet min_distance requirement')

    return points


def pad_nested_list(nested_list, list_len, num_lists):
    # Pad shorter inner lists with zeros
    for inner_list in nested_list:
        padding_len = list_len - len(inner_list)
        inner_list.extend([0] * padding_len)

    # Add additional zero-filled lists if needed
    additional_lists = num_lists - len(nested_list)
    if additional_lists > 0:
        zero_filled_list = [0] * list_len
        nested_list.extend([zero_filled_list.copy() for _ in range(additional_lists)])

    return nested_list


def project_line_2_rad(p1, p2):
    v1 = p1[:2]
    v2 = p2[:2]
    dot = np.dot(v1, v2)
    det = np.linalg.det([v1, v2])
    return np.arctan2(det, dot)


def apply_acceleration(init_velocity: np.ndarray,
                       velocity_max: float,
                       acc: np.ndarray,
                       dt=1):
    # verify if acc is practiable, high than self.velocity_max

    uncorrected_final_velocity = init_velocity + acc * dt
    norm_final = np.linalg.norm(uncorrected_final_velocity)
    if norm_final > velocity_max:
        final_velocity = uncorrected_final_velocity / norm_final
    else:
        final_velocity = uncorrected_final_velocity

    position_delta = (init_velocity + final_velocity) / 2 * dt
    return final_velocity, position_delta, 0


# def apply_acceleration(init_velocity: np.ndarray,
#                        velocity_max: float,
#                        ori_acc: np.ndarray,
#                        dt=1):
#     acc = ori_acc * dt
#     final_velocity = np.linalg.norm(init_velocity + acc)
#     # verify if acc is practiable, high than self.velocity_max
#
#     # # here may lead to final_velocity>1 and cuase problem for the next iteration
#     # if np.isclose(final_velocity, velocity_max):
#     #     final_velocity = velocity_max
#
#     if final_velocity > velocity_max:
#         if isinstance(init_velocity, types.FunctionType):
#             input('init_velocity is function')
#         norm_acc, _ = law_of_sines(velocity_max, np.linalg.norm(init_velocity),
#                                    np.pi - vec_vec_rad(init_velocity, ori_acc))
#         first_part_unit_time = norm_acc / np.linalg.norm(acc)
#         final_velocity = init_velocity + first_part_unit_time * acc
#         distance_part_1 = (init_velocity + final_velocity) / 2 * first_part_unit_time * dt
#         distance_part_2 = final_velocity * (1 - first_part_unit_time) * dt
#         position_delta = distance_part_1 + distance_part_2
#     else:
#         first_part_unit_time = 1
#         final_velocity = init_velocity + acc
#         position_delta = init_velocity * dt + 0.5 * ori_acc * dt ** 2  # (init_velocity + final_velocity) / 2 * dt
#     return final_velocity, position_delta, np.linalg.norm(ori_acc) * (first_part_unit_time - 1)


def polar_to_unit_normal(direction):
    """
    convert polar expression direction to normal
    :param direction:
    :return:
    """
    if len(direction) == 2:
        # Spherical coordinate system: (r), theta, phi
        theta, phi = direction
        return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    elif len(direction) == 1:
        # polar coordinate system, phi, (r)
        phi = direction[0]
        return np.array([np.cos(phi), np.sin(phi)])


def law_of_cosines(a, b, C):
    """
    lower letters for sides, upper letters for opposite angels
    a/sin(A)=b/sin(B)=c/sin(C)
    A,B,C in radians
    """
    return np.sqrt(a ** 2 + b ** 2 - 2 * a * b * np.cos(C))


# elements = (a, b, A, B)
# count_none = elements.count(None)
# if count_none <= 1:
#     if a is not None and b is not None:
#         if A is None:
#             a, b, A = b, a, B  # Swap variables
#         ratio = a / np.sin(A)
#         # Clamping the value inside the arcsine function to the range [-1, 1]
#         B = np.arcsin(np.clip(b * ratio, -1, 1))
#     elif a is None:
#         a, A, B = b, B, A  # Swap variables
#         ratio = a / np.sin(A)
#
#     # Handle the case where A or B are close to pi
#     if np.isclose(A, np.pi) or np.isclose(B, np.pi):
#         C = 0  # The other angle is 0 when one of the angles is pi
#     else:
#         C = np.pi - A - B  # Calculate the remaining angle
#
#     c = np.sin(C) / ratio
def law_of_sines(a, b, A=None, B=None):
    """
    lower letters for sides, upper letters for opposite angels
    a/sin(A)=b/sin(B)=c/sin(C)
    A,B,C in radians
    """
    elements = (a, b, A, B)
    count_none = elements.count(None)
    if count_none <= 1:
        if a is not None and b is not None:
            if A is None:
                a, b, A = b, a, B
            # ratio = a / np.sin(A)
            ratio = np.sin(A) / a
            if b * ratio >= 1 or b * ratio <= -1:
                print(b, ratio)

                print(1)
            B = np.arcsin(b * ratio)

        elif a is None:
            a, A, B = b, B, A
            ratio = np.sin(A) / a

        if np.isclose(A, np.pi) or np.isclose(B, np.pi):
            C = 0
            c = max(a, b) - min(a, b)
        else:
            C = np.pi - A - B
            c = np.sin(C) / ratio

        return c, C
    else:
        raise ValueError('a,b at least two elements are None')
        return None


def vec_vec_rad(vec_1, vec_2):
    """


    unit_vec_1 is setted as [0,0,1]
    vec_2 may need to be normalized
    """

    dot_product = np.dot(vec_1, vec_2)
    mag_1 = np.linalg.norm(vec_1)
    mag_2 = np.linalg.norm(vec_2)
    # print(dot_product,mag_1,mag_2)
    if np.isnan(mag_1) or np.isnan(mag_2):
        print(1)
    pro = np.clip(dot_product / mag_1 / mag_2, -1, 1)
    rad = np.arccos(pro)
    return rad


# @lru_cache(maxsize=8)
def rotate(vec, fromVec=None, toVec=None, matrix=None):
    if matrix is not None:
        rotation_matrix = matrix
    else:
        rotation_matrix = vec2vec_rotation(fromVec, toVec)
    return np.dot(rotation_matrix, vec)


##########################################################################################
def proj_to_plane(vec, normal):
    # remove the component in the direction of normal
    projected_vec_align_to_normal = np.dot(vec, normal) * normal
    return vec - projected_vec_align_to_normal


# @lru_cache(maxsize=8)
def transform2global(point, from_vec, corridinationOffset):
    '''
        trans corridor-centri cooridination to global coordination

        :param point:
        :param toVec:
        :param corridinationOffset:
        :return:
    '''
    deoffset_point = point - corridinationOffset
    base_point = rotate(deoffset_point, from_vec, np.array(0, 0, 1))
    return base_point


@lru_cache(maxsize=8)
def transform2relative(point, toVec, corridinationOffset):
    '''
    trans global cooridination to corridor-centric coordination

    :param point:
    :param toVec:
    :param corridinationOffset:
    :return:
    '''
    rotated_point = rotate(point, np.array([0, 0, 1]), toVec)
    offset_point = rotated_point + corridinationOffset
    return offset_point


def vec2vec_rotation(unit_vec_1, vec_2):
    norm_vec_2 = np.linalg.norm(vec_2)
    if np.abs(norm_vec_2 - 1) > TRIVIAL_TOLERANCE:
        unit_vec_2 = vec_2 / norm_vec_2
    else:
        unit_vec_2 = vec_2

    if np.abs(np.linalg.norm(unit_vec_1) - 1) > TRIVIAL_TOLERANCE:
        unit_vec_1 /= np.linalg.norm(unit_vec_1)
    dot_product = np.dot(unit_vec_1, unit_vec_2)
    if dot_product > 1 or dot_product < -1:
        print(dot_product)
    # Ensure dot product is within valid range [-1, 1]
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate the angle
    angle = np.arccos(dot_product)
    # angle = np.arccos(np.dot(unit_vec_1, unit_vec_2))
    if angle < TRIVIAL_TOLERANCE:
        return np.identity(3, dtype=np.float64)

    if angle > (np.pi - TRIVIAL_TOLERANCE):
        # WARNING this only works because all geometries are rotationaly invariant
        # minus identity is not a proper rotation matrix
        return -np.identity(3, dtype=np.float64)

    rot_vec = np.cross(unit_vec_1, unit_vec_2)
    rot_vec = rot_vec.astype(float)
    rot_vec /= np.linalg.norm(rot_vec)
    return o3d.geometry.get_rotation_matrix_from_axis_angle(angle * rot_vec)


def vector_equal(v1, v2):
    return v1.shape == v2.shape and np.allclose(v1, v2)


def distance_point_point(p1, p2):
    """Calculates the euclidian distance between two points or sets of points
    >>> distance_point_point(np.array([1, 0]), np.array([0, 1]))
    1.4142135623730951
    >>> distance_point_point(np.array([[1, 1], [0, 0]]), np.array([0, 1]))
    array([1., 1.])
    >>> distance_point_point(np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, -3]]))
    array([1., 3.])
    """
    return scipy.spatial.minkowski_distance(p1, p2)


def random_(difficulty, epsilon=0.1, segment=False):
    assert difficulty > 0
    if segment:

        return random.uniform(min(difficulty, 0.1), difficulty + epsilon)
    else:
        if difficulty < 1:
            random_value = max(difficulty, 0.1) + random.uniform(-epsilon, epsilon)
        else:
            random_value = random.uniform(1 - epsilon, difficulty + epsilon)
            # random_value = 1
    return random_value


def distance_circle_point(anchor_point, direction, radius, point):
    delta_p = point - anchor_point
    x1 = np.matmul(
        np.expand_dims(np.dot(delta_p, direction), axis=-1),
        np.atleast_2d(direction),
    )
    x2 = delta_p - x1
    return np.sqrt(
        np.linalg.norm(x1, axis=-1) ** 2
        + (np.linalg.norm(x2, axis=-1) - radius) ** 2
    )


def distance_plane_point(plane_point, plane_normal, point):
    """Calculates the signed distance from a plane to one or more points
    >>> distance_plane_point(np.array([0, 0, 1]), np.array([0, 0, 1]), np.array([2, 2, 2]))
    1
    >>> distance_plane_point(np.array([0, 0, 1]), np.array([0, 0, 1]), np.array([[2, 2, 2], [2, 2, 3]]))
    array([1, 2])
    """
    assert np.allclose(np.linalg.norm(plane_normal), 1.0)
    return np.dot(point - plane_point, plane_normal)


def distance_perpendicular_line_point(anchor_point, direction, point):
    """Calculates the distance from a line to a point
    >>> distance_perpendicular_line_point(np.array([0, 0, 1]), np.array([0, 0, 1]), np.array([1, 1, 2]))
    1.4142135623730951
    >>> distance_perpendicular_line_point(np.array([0, 0, 1]), np.array([0, 0, 1]), np.array([[1, 0, 1], [0, 2, 3]]))
    array([1., 2.])
    """
    # assert np.allclose(np.linalg.norm(direction), 1.0)
    direction = direction / np.linalg.norm(direction)
    delta_p = point - anchor_point
    return distance_point_point(
        delta_p,
        np.matmul(
            np.expand_dims(np.dot(delta_p, direction), axis=-1),
            np.atleast_2d(direction),
        ),
    )


def distance_signed_parallel_line_point(anchor_point, direction, point):
    """Calculates the SIGNED (+-) distance between anchor point to the projected point on the line,
        """
    anchor_point = np.array(anchor_point)
    direction = np.array(direction)
    point = np.array(point)

    if len(anchor_point) != len(direction) or len(anchor_point) != len(point):
        raise ValueError("Dimensions of anchor_point, direction, and point must be equal.")

    if len(anchor_point) not in [2, 3]:
        raise ValueError("This function only supports 2D and 3D cases.")

    # Normalize the direction vector
    direction_normalized = direction / np.linalg.norm(direction)

    # Calculate the vector from the anchor_point to the given point
    point_vector = point - anchor_point

    # Calculate the scalar projection of point_vector onto the direction_normalized vector
    scalar_projection = np.dot(point_vector, direction_normalized)

    # Calculate the signed distance between the anchor_point and the projected point
    signed_distance = scalar_projection

    return signed_distance


def are_two_lines_parallel(d1, d2):
    if np.allclose(d1, d2):
        return True
    return False


def closest_points_on_two_skew_lines(p1, d1, p2, d2):
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)
    if are_two_lines_parallel(d1, d2):
        return None, None, np.linalg.norm(np.cross(p2 - p1, d1)) / np.linalg.norm(d2)
    v3 = np.cross(d1, d2)
    v3 = v3 / np.linalg.norm(v3)

    mat = np.column_stack((d1, -d2, v3))
    t1, t2, t3 = np.linalg.solve(mat, p2 - p1)
    q1 = p1 + t1 * d1
    q2 = p2 + t2 * d2
    return q1, q2, np.abs(t3)


def onoff_ramp_points(p1, d1, p2, d2):
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)
    if are_two_lines_parallel(d1, d2):
        return None, None
    q1, q2, t3 = closest_points_on_two_skew_lines(p1, d1, p2, d2)
    middle = (q1 + q2) / 2

    r = t3 / 2
    # centric, begin, end
    c1 = middle - d1 * r
    b1 = q1 - d1 * t3 / 2
    e1 = middle

    c2 = middle + d2 * r
    b2 = middle
    e2 = q2 + d2 * r

    return [c1, b1, e1], [c2, b2, e2], r


def rotate_to_xy_plane(torus_points):
    # c: centric, a: begin point, b: end point
    c, b, e = torus_points

    # Find the normal vector of the plane
    normal_vec = np.cross(b - c, e - c)
    normal_vec = normal_vec / np.linalg.norm(normal_vec)

    # Find the rotation matrix
    rotation_matrix = vec2vec_rotation(normal_vec, [0, 0, 1])

    # Rotate b and c to the x-y plane
    b_new = np.matmul(rotation_matrix, b - c)
    e_new = np.matmul(rotation_matrix, e - c)

    return b_new, e_new


def distance_line_line(p1, d1, p2, d2):
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)

    if are_two_lines_parallel(d1, d2):
        return np.linalg.norm(np.cross(p2 - p1, d1)) / 1  # np.linalg.norm(d1)
    else:
        return distance_two_skew_lines(p1, d1, p2, d2)


def distance_two_skew_lines(p1, d1, p2, d2):
    v = np.cross(d1, d2)

    # Calculate the denominator of the line parameter equations
    denom = np.linalg.norm(v)

    w = p1 - p2

    # Calculate the line parameters
    a = np.dot(v, w)
    distance = np.linalg.norm(a) / denom

    return distance


def polar_to_cartesian(radius, angle):
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    return np.array([x, y])


def cartesian_to_polar_or_spherical(vector):
    if len(vector) == 2:
        # Handle 2D case
        radius = np.linalg.norm(vector)
        angle = np.arctan2(vector[1], vector[0])
        return radius, angle
    elif len(vector) == 3:
        # Handle 3D case
        r = np.linalg.norm(vector)
        phi = np.arctan2(vector[1], vector[0])
        theta = np.arccos(vector[2] / r)
        return r, theta, phi
    else:
        raise ValueError("Input vector must be 2D or 3D")


def is_line_line_intersect(a: np.ndarray,
                           b: np.ndarray,
                           c: np.ndarray,
                           d: np.ndarray):
    # is line ab cd, cross each other? and how much for the vector between point b and the cross point

    # Calculate vectors AB and CD
    # sequence does not matter
    AB = b - a
    CD = d - c

    # Calculate the cross product of AB and CD
    cross_product = np.cross(AB, CD)

    # If the cross product is zero, the lines are parallel
    if np.abs(cross_product) > 1e-5:
        # Check if the intersection point is within the bounds of both lines
        if np.cross(c - a, c - b) * np.cross(d - a, d - b) <= 0 and \
                np.cross(a - c, a - d) * np.cross(b - c, b - d) <= 0:
            #   is cross? and how much after crossing?
            return True, ((b - c) + (b - d)) / 2
        else:
            return False, None

    # If the cross product is negative, the lines do not intersect
    else:
        return False, None


def align_measure(start, end, direction):
    moving = end - start
    unit_moving = moving / np.linalg.norm(moving)
    return np.dot(unit_moving, direction)


def is_line_circle_intersect(line_start, line_end, anchor, direction, radius):
    '''

    :param line_start:
    :param line_end:
    :param anchor:
    :param direction:
    :param radius:
    :return: not only check intersection, but also how much the moving vector is aligned with the direction
    '''

    # Check if line intersects plane
    to_start = line_start - anchor
    to_end = line_end - anchor
    dot_start = np.dot(to_start, direction)
    dot_end = np.dot(to_end, direction)

    # Both points are on the same side of the plane
    if dot_start * dot_end > 0:  # Both points are on the same side of the plane
        return False
    else:
        # Calculate intersection point with the circle plane
        line_dir = line_end - line_start
        t = np.linalg.norm(dot_start) / np.linalg.norm(dot_end - dot_start)
        intersection_point = line_start + t * line_dir

        # Check if the intersection point is within the circle
        vec_to_intersection = intersection_point - anchor
        if np.linalg.norm(vec_to_intersection) <= radius:
            # moving_vec = line_end - line_start
            # unit_moving_vec = moving_vec / np.linalg.norm(moving_vec)
            return True
        else:
            return False


# def is_line_circle_intersect(a, b, anchor, direction, radius):
#     # Calculate the line segment direction vector
#     AB = b - a
#
#     # Normalize the plane direction vector
#     normal = direction / np.linalg.norm(direction)
#
#     # Calculate the dot product of the plane normal and AB
#     dot_product = np.dot(AB, normal)
#
#     # If the dot product is zero, the line is parallel to the plane and doesn't intersect
#     if np.abs(dot_product) < 1e-5:
#         return False
#
#     # Calculate the distance from point A to the plane along the line direction
#     distance = np.dot(anchor - a, normal) / dot_product
#
#     # If the intersection point is not within the bounds of the line segment, return False
#     if distance < 0 or distance > 1:
#         return False
#
#     # Calculate the intersection point
#     intersection = a + distance * AB
#
#     # Check if the intersection point is within the circle on the plane
#     circle_center_to_intersection = intersection - anchor
#     distance_to_circle_center = np.linalg.norm(circle_center_to_intersection)
#
#     return distance_to_circle_center <= radius


def vector_to_2d_rad(vector):
    return np.arctan2(vector[1], vector[0])


def vector_to_perpendicular(direction):
    direction = np.array(direction)

    # Handle 2D case
    if direction.shape[0] == 2:
        return np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)

    # Handle 3D case
    elif direction.shape[0] == 3:
        # Pick an arbitrary vector perpendicular to direction
        if np.abs(direction[0]) < np.abs(direction[1]):
            perp_vector = np.array([1, 0, 0])
        else:
            perp_vector = np.array([0, 1, 0])
        # Compute the cross product between direction and perp_vector
        perp_unit_vector = np.cross(direction, perp_vector)
        return perp_unit_vector / np.linalg.norm(perp_unit_vector)

    # Handle invalid input
    else:
        raise ValueError("Input direction vector must be 2D or 3D")


def radian_difference(angle1_rad, angle2_rad, counter_clockwise=True, output_range='[-pi, pi]'):
    if counter_clockwise:
        diff_rad = angle1_rad - angle2_rad
    else:
        diff_rad = angle2_rad - angle1_rad
    if output_range == '[-pi, pi]':
        # Normalize the angle difference to be between -pi and pi
        diff_rad = (diff_rad + math.pi) % (2 * math.pi) - math.pi
    elif output_range == '[0, 2pi]':
        # Normalize the angle difference to be between 0 and 2*pi
        diff_rad = diff_rad % (2 * math.pi)
    else:
        raise ValueError("Invalid output_range specified. Choose either '[-pi, pi]' or '[0, 2pi]'.")
    return diff_rad


def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])


def onoff_ramp_points(p1, d1, p2, d2):
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)
    if are_two_lines_parallel(d1, d2):
        return None, None
    q1, q2, t3 = closest_points_on_two_skew_lines(p1, d1, p2, d2)
    middle = (q1 + q2) / 2

    r = t3 / 2
    # centric, begin, end
    c1 = middle - d1 * r
    b1 = q1 - d1 * t3 / 2
    e1 = middle

    c2 = middle + d2 * r
    b2 = middle
    e2 = q2 + d2 * r

    return [c1, b1, e1], [c2, b2, e2], r


def closest_points_on_two_skew_lines(p1, d1, p2, d2):
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)
    if are_two_lines_parallel(d1, d2):
        return None, None, np.linalg.norm(np.cross(p2 - p1, d1)) / np.linalg.norm(d2)
    v3 = np.cross(d1, d2)
    v3 = v3 / np.linalg.norm(v3)

    mat = np.column_stack((d1, -d2, v3))
    t1, t2, t3 = np.linalg.solve(mat, p2 - p1)
    q1 = p1 + t1 * d1
    q2 = p2 + t2 * d2
    return q1, q2, np.abs(t3)


def unify_rad(rad):
    while 1:
        if rad > np.pi:
            rad -= 2 * np.pi
        elif rad < -np.pi:
            rad += 2 * np.pi
        else:
            return rad


def bfs_find_path(graph, init, des):
    """
    graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
    }

    print(bfs_find_path(graph, 'A', 'F'))
    """
    # deal with the case that init==des
    if init == des:
        return [init]
    queue = deque([(init, [init])])
    while queue:
        (vertex, path) = queue.popleft()
        for neighbor in graph[vertex]:
            if neighbor == des:
                return path + [neighbor]
            elif neighbor not in path:
                queue.append((neighbor, path + [neighbor]))


def counter_clockwise_radian(point1, point2):
    # point1 and point2 shall be relative positions to an anchor for measuring angle
    dot = np.dot(point1, point2)
    det = np.linalg.det([point1, point2])
    return np.arctan2(det, dot)


def intersect_time_position(p_A, v_A, p_B, v_B):
    # Calculate the time at which they intersect
    # Solve for t where p_A + t*v_A = p_B + t*v_B
    v_diff = v_A - v_B
    p_diff = p_B - p_A

    # Check if the velocities are parallel
    if np.cross(v_A, v_B) == 0:
        # If velocities are parallel, check if they are collinear (same line of motion)
        if np.cross(p_B - p_A, v_A) == 0:
            # Calculate the time when B reaches A's initial position, or vice versa
            t = np.dot(p_B - p_A, v_A) / np.dot(v_A, v_A)
            intersection_point = p_A if t > 0 else p_B
        else:
            t = None
            intersection_point = None
    else:
        # The velocities are not parallel, solve for t using least squares
        A = np.vstack([v_A, -v_B]).T
        b = p_B - p_A
        t, res, rank, s = np.linalg.lstsq(A, b, rcond=None)
        # intersection_point = p_A + t[0] * v_A
    if t is None or any(t < 0):
        return None
    else:
        return t


def collide_time_position(p_A, v_A, p_B, v_B, collide_distance):
    if np.linalg.norm(v_A - v_B) < TRIVIAL_TOLERANCE:
        return float('inf'), float('inf')

    # Define the threshold distance c
    c = collide_distance  # Example value for c

    # Calculate the coefficients of the quadratic equation
    # The equation takes the form of at^2 + bt + c < d^2

    # Coefficient a: squared norm of the relative velocity
    a = np.dot(v_A - v_B, v_A - v_B)

    # Coefficient b: 2 times the dot product of the relative position and relative velocity
    b = 2 * np.dot(p_A - p_B, v_A - v_B)

    # Coefficient c: squared norm of the relative position - c^2
    c_coeff = np.dot(p_A - p_B, p_A - p_B) - c ** 2

    # Calculate the discriminant of the quadratic equation
    discriminant = b ** 2 - 4 * a * c_coeff

    # Check if the discriminant is negative, which means no real solutions (they never get that close)
    if discriminant < 0:
        # t_values = None #"No solution: They never get within the specified distance."
        return float('inf'), float('inf')
    else:
        # Calculate the two solutions of the quadratic equation
        t1 = (-b + np.sqrt(discriminant)) / (2 * a)
        t2 = (-b - np.sqrt(discriminant)) / (2 * a)
    return t2, t1
    # Filter the time values to return the minimum value greater than 0, unless both are less than 0
    # if t1 < 0 and t2 < 0:
    #     min_t_value = max(t1,t2)
    # else:
    #     min_t_value = min(t for t in [t1, t2] if t > 0)
    # return min_t_value
