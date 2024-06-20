import pickle

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from moviepy.editor import VideoFileClip

from air_corridor.d3.corridor.corridor import CylinderCorridor, DirectionalPartialTorusCorridor

matplotlib.use('TkAgg')
from air_corridor.tools.util import *


class Visualization():
    def __init__(self, max_rounds=None, to_base=False):

        self.size = 30
        self.ax = None
        self.max_rounds = max_rounds if max_rounds is not None else 1
        self.animate_rounds = [{'corridor': None, 'uav': {}, 'ncfo': {}} for _ in range(self.max_rounds)]
        self.line = None
        self.current_corridor = []  # None#[]
        self.to_base = to_base

    def put_data(self, round, agents, ncfos, corridors=None):
        if self.animate_rounds[round]['corridor'] is None and corridors is not None:
            self.animate_rounds[round]['corridor'] = corridors
        for agent in agents:
            if agent in self.animate_rounds[round]['uav']:
                self.animate_rounds[round]['uav'][agent].append(agent.position)
            else:
                self.animate_rounds[round]['uav'][agent] = [agent.position]
            if self.to_base:
                self.animate_rounds[round]['uav'][agent][-1] = self.animate_rounds[round]['corridor'][
                    'A'].rotate_to_base(
                    self.animate_rounds[round]['uav'][agent][-1] - self.animate_rounds[round]['corridor'][
                        'A'].anchor_point)  #
        for ncfo in ncfos:
            if ncfo in self.animate_rounds[round]['ncfo']:
                self.animate_rounds[round]['ncfo'][ncfo].append(ncfo.position)
            else:
                self.animate_rounds[round]['ncfo'][ncfo] = [ncfo.position]
            # if self.to_base:
            #     self.animate_rounds[round]['ncfo'][agent][-1] = self.animate_rounds[round]['corridor'][
            #         'A'].rotate_to_base(
            #         self.animate_rounds[round]['ncfo'][agent][-1] - self.animate_rounds[round]['corridor'][
            #             'A'].anchor_point)
            # self.animate_rounds[round]['uav'][agent][-1] = self.animate_rounds[round]['corridor'][
            #     'A'].rotate_to_base(
            #     self.animate_rounds[round]['uav'][agent][-1] )  #

    def save_data(self, file_name):
        # Serialize with Pickle
        with open(f'{file_name}.dl', 'wb') as f:
            pickle.dump(self.animate_rounds, f)

    def read_data(self, file_name):
        with open(f'{file_name}.dl', 'rb') as f:
            self.animate_rounds = pickle.load(f)
        self.max_rounds = len(self.animate_rounds)

    def animate(self, frame_data):
        # round_index, frame_index=kwargs
        round_index, frame_index = frame_data

        if frame_index == 0:
            self.num_agents = len(self.animate_rounds[round_index]['uav'])
            self.num_ncfos = len(self.animate_rounds[round_index]['ncfo'])
            self.lines = ([self.ax.plot([0, 0, 0], [0, 0, 0], [0, 0, 0], linewidth=3)[0] for _ in
                           range(self.num_agents)] +
                          [self.ax.plot([0, 0, 0], [0, 0, 0], [0, 0, 0], linewidth=4, c='r')[0] for _ in
                           range(self.num_ncfos)])
            self.plot_corridor(round_index)

        lines = self.lines
        current_round_UAV_frames = self.animate_rounds[round_index]['uav']
        start_idx = max(0, frame_index - 20)
        for line, (agent, traj) in zip(lines[:self.num_agents], current_round_UAV_frames.items()):
            end_idx = min(frame_index + 1, len(traj))
            line.set_data(traj[start_idx:end_idx, 0], traj[start_idx:end_idx, 1])
            line.set_3d_properties(traj[start_idx:end_idx, 2])

        current_round_NCFO_frames = self.animate_rounds[round_index]['ncfo']
        for line, (ncfo, traj) in zip(lines[self.num_agents:], current_round_NCFO_frames.items()):
            line.set_data(traj[start_idx:end_idx, 0], traj[start_idx:end_idx, 1])
            line.set_3d_properties(traj[start_idx:end_idx, 2])

        # for line in lines:
        #     print(f"line: {line.get_data()}")
        return lines

    def show_animation(self, gif=True, load_file_name=None, save_to=None):
        if load_file_name:
            self.read_data(load_file_name)
        fig = plt.figure()
        self.ax = fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_xlim(-self.size, self.size)
        self.ax.set_ylim(-self.size, self.size)
        self.ax.set_zlim(-self.size, self.size)
        ani = FuncAnimation(fig, self.animate, frames=self.frame_locate(), interval=100)
        #ani.save(f"{save_to}.gif",writer='imagemagick', fps=30)
        ani.save(f"{save_to}.gif", writer='pillow', fps=30)
        # if mp4 or gif:
        #     ani.save(f"{save_to}.mp4", writer='ffmpeg', bitrate=2000, fps=30)
        #     if gif:
        #         clip = VideoFileClip(f"{save_to}.mp4")
        #         clip.write_gif(f"{save_to}.gif")
        plt.show()

    def frame_locate(self, tail=20):
        # frames_in_each_round = [max([len(single_round_data['uav'][agent]) for agent in single_round_data['uav']]) for single_round_data in
        #                         self.animate_rounds]
        frames_in_each_round = []
        for single_round_data in self.animate_rounds:
            longest_frames = 0
            for agent in single_round_data['uav']:
                longest_frames = max(longest_frames, len(single_round_data['uav'][agent]))
                single_round_data['uav'][agent] = np.array(single_round_data['uav'][agent])
            for ncfo in single_round_data['ncfo']:
                # longest_frames = max(longest_frames, len(single_round_data['ncfo'][ncfo]))
                single_round_data['ncfo'][ncfo] = np.array(single_round_data['ncfo'][ncfo])
            frames_in_each_round.append(longest_frames)
        for round_index, num_frame in enumerate(frames_in_each_round):
            for frame_index in range(num_frame + tail):
                yield (round_index, frame_index)

    def plot_corridor(self, round):
        def torus(R, r, begin_rad, end_rad, R_res=100, r_res=100):
            u = np.linspace(begin_rad, end_rad, R_res)
            v = np.linspace(0, 2 * np.pi, r_res)
            u, v = np.meshgrid(u, v)
            x = (R + r * np.cos(v)) * np.cos(u)
            y = (R + r * np.cos(v)) * np.sin(u)
            z = r * np.sin(v)
            return x, y, z

        def cylinder(r, h, theta_res=100, z_res=100):
            theta = np.linspace(0, 2 * np.pi, theta_res)
            z = np.linspace(-h / 2, h / 2, z_res)
            theta, z = np.meshgrid(theta, z)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            return x, y, z

        for surface in self.current_corridor:
            if surface in self.ax.collections:
                surface.remove()
        plt.draw()
        for name, corridor in self.animate_rounds[round]['corridor'].items():

            if isinstance(corridor, CylinderCorridor):
                Xt, Yt, Zt = cylinder(r=corridor.radius,
                                      h=corridor.length)

                if self.to_base:
                    rotation_matrix = np.eye(3)
                else:
                    rotation_matrix = vec2vec_rotation(Z_UNIT, corridor.orientation_vec)
            elif isinstance(corridor, DirectionalPartialTorusCorridor):
                if self.to_base:
                    Xt, Yt, Zt = torus(R=corridor.major_radius,
                                       r=corridor.minor_radius,
                                       begin_rad=0,
                                       end_rad=corridor.end_rad - corridor.begin_rad)
                    rotation_matrix = np.eye(3)
                else:
                    Xt, Yt, Zt = torus(R=corridor.major_radius,
                                       r=corridor.minor_radius,
                                       begin_rad=corridor.begin_rad,
                                       end_rad=corridor.end_rad)
                    # print(f"radius: {corridor.major_radius}, direction: {corridor.orientation_vec}, "
                    #       f"begin: {corridor.beginCirclePlane.anchor_point}")
                    rotation_matrix = vec2vec_rotation(Z_UNIT, corridor.orientation_vec)
            translate = corridor.anchor_point
            # Apply rotation
            x_rot_torus, y_rot_torus, z_rot_torus = [], [], []
            for a, b, c in zip(Xt, Yt, Zt):
                x_p, y_p, z_p = np.dot(rotation_matrix, np.array([a, b, c]))
                x_rot_torus.append(x_p + translate[0])
                y_rot_torus.append(y_p + translate[1])
                z_rot_torus.append(z_p + translate[2])
            # print(f"plot {name}")
            obj = self.ax.plot_surface(np.array(x_rot_torus), np.array(y_rot_torus),
                                       np.array(z_rot_torus),
                                       edgecolor='royalblue',
                                       lw=0.1, rstride=20, cstride=4, alpha=0.1)
            self.current_corridor.append(obj)
