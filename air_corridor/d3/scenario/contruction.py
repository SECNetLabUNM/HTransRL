from air_corridor.tools.util import *

minor_radius = 2
major_radius = 10

global_direction_rad = [random.random() * np.pi, (random.random() - 0.5) * 2 * np.pi]  # theta, phi
global_direction = polar_to_unit_normal(global_direction_rad)
rotation_maxtrix_to_remote = vec2vec_rotation(np.array([0, 0, 1]), vec_2=global_direction)
rotation_matrix_to_base = vec2vec_rotation(global_direction, np.array([0, 0, 1]))
# generate info for torus 1
torus_1_y = global_direction
torus_1_x_base = random_unit(2)
torus_1_x = np.dot(rotation_maxtrix_to_remote, torus_1_x_base)
torus_1_anchor = np.array([0, 0, 0]) - torus_1_x * major_radius
torus_1_direction = np.cross(torus_1_x, torus_1_y)
torus_1_begin = np.arctan2(torus_1_x[1], torus_1_x[0])
torus_1_end = torus_1_begin + np.pi / 2

# generate info for torus 1
torus_2_x = -global_direction
torus_2_y_base = random_unit(2)
torus_2_y = np.dot(rotation_maxtrix_to_remote, torus_2_y_base)
torus_2_anchor = np.array([0, 0, 0]) - torus_2_y * major_radius
torus_2_direction = np.cross(torus_1_x, torus_1_y)
torus_2_end = np.arctan2(torus_1_x[1], torus_1_x[0])
torus_2_begin = torus_2_end - np.pi / 2
DirectionalPartialTorusCorridor(name='B',
                                anchor_point=torus_1_anchor,
                                direction_rad=torus_1_direction,
                                major_radius=10,
                                minor_radius=2,
                                begin_rad=torus_1_begin,
                                end_rad=torus_1_end,
                                connections=['C'],
                                reduce_space=True)

corridors = {
    'B': DirectionalPartialTorusCorridor(name='B',
                                         anchor_point=torus_1_anchor,
                                         direction_rad=torus_1_direction,
                                         major_radius=10,
                                         minor_radius=2,
                                         begin_rad=torus_1_begin,
                                         end_rad=torus_1_end,
                                         connections=['C'],
                                         reduce_space=True),
    'C': DirectionalPartialTorusCorridor(name='C',
                                         anchor_point=torus_2_anchor,
                                         direction_rad=torus_2_direction,
                                         major_radius=10,
                                         minor_radius=2,
                                         begin_rad=torus_2_begin,
                                         end_rad=torus_2_end,
                                         connections=['D'],
                                         reduce_space=True),
}
cor_graph = corridors['B'].convert2graph(self.corridors)
wait()
# corridors = {'A': CylinderCorridor(name='A',
#                                    anchor_point=np.array([0, 0, 0]),
#                                    direction_rad=direction_rad,
#                                    length=random.random() * 10 + 5,
#                                    width=4,
#                                    connections=['B'],
#                                    reduce_space=True),
#              'B': DirectionalPartialTorusCorridor(name='B',
#                                                   anchor_point=torus_1_anchor,
#                                                   direction_rad=torus_1_direction,
#                                                   major_radius=10,
#                                                   minor_radius=2,
#                                                   begin_rad=begin_rad,
#                                                   end_rad=end_rad,
#                                                   connections=['C'],
#                                                   reduce_space=True),
#              'C': DirectionalPartialTorusCorridor(name='C',
#                                                   anchor_point=np.array([0, 0, 0]),
#                                                   direction_rad=direction_rad,
#                                                   major_radius=10,
#                                                   minor_radius=2,
#                                                   begin_rad=begin_rad,
#                                                   end_rad=end_rad,
#                                                   connections=['D'],
#                                                   reduce_space=True),
#              'D': CylinderCorridor(name='D',
#                                    anchor_point=np.array([0, 0, 0]),
#                                    direction_rad=direction_rad,
#                                    length=random.random() * 10 + 5,
#                                    width=4,
#                                    connections=[''], reduce_space=True)
#              }
