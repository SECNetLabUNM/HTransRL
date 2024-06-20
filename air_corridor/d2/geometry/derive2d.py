from air_corridor.tools.util import *

# def overlap_rectangle_circle(rect, circle):
#     # Compute the corners of the rectangle
#     corners = rectangle_corners(rect)
#
#     # Compute the closest point on the rectangle to the center of the circle
#     closest_point = closest_point_on_rectangle_to_point(rect, circle.center)
#
#     # Check if the closest point is inside the circle
#     if distance_between_points(closest_point, circle.center) <= circle.radius:
#         return True
#
#     # Check if any edge of the rectangle intersects the circle
#     for edge in rectangle_edges(rect):
#         if edge_intersects_circle(edge, circle):
#             return True
#
#     # The rectangle and the circle do not overlap
#     return False
#
# def overlap_rectangles(rect1, rect2):
#     # Compute the corners of the rectangles
#     corners1 = rectangle_corners(rect1)
#     corners2 = rectangle_corners(rect2)
#
#     # Check if any corner of one rectangle lies inside the other rectangle
#     if any(is_point_inside_rectangle(corner, rect2) for corner in corners1):
#         return True
#     if any(is_point_inside_rectangle(corner, rect1) for corner in corners2):
#         return True
#
#     # Check if any edge of one rectangle intersects an edge of the other rectangle
#     for edge1 in rectangle_edges(rect1):
#         for edge2 in rectangle_edges(rect2):
#             if edges_intersect(edge1, edge2):
#                 return True
#
#     # The rectangles do not overlap
#     return False