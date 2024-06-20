import argparse
import glob
import logging
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from functools import reduce

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

path_to_air_corridor = '/home/kun/PycharmProjects/air-corridor_ncfo'
sys.path.insert(0, path_to_air_corridor)

# for i in sys.path:
#     print(i)
import air_corridor.d3.scenario.D3shapeMove as d3
from air_corridor.tools.log_config import setup_logging
from air_corridor.tools.util import save_init_params
from ppo import PPO

print(d3)


def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True', 'true', 'TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
# # parser.add_argument('--ModelIndex', type=float, default=0, help='which model to load')
parser.add_argument('--LoadModel', type=str2bool, default=False, help='Load pretrained model or Not')
# # parser.add_argument('--LoadFolder', type=str, default=None, help='Which folder to load')
#
# parser.add_argument('--ModelIndex', type=float, default=2e7, help='which model to load')
# parser.add_argument('--LoadModel', type=str2bool, default=True, help='Load pretrained model or Not')
# parser.add_argument('--LoadFolder', type=str,
#                     default='/mnt/storage/result/d2move_20240325092534_new_net/width_128epoch4_index_True_state2_cbfFalse_acc0.3_future2_shareTrue_netfc10_horizon8_batch16_enc2_dec2_spaceTrue_level20_capacity6_beta_base1.0_beta_adaptor_coefficient1.1',
#                     help='Which folder to load')
parser.add_argument('--net_model', type=str, default='fc10_3e', help='number of encoders')
# parser.add_argument('--net_model', type=str, default='mask_3e', help='number of encoders')

parser.add_argument('--complexity', type=str, default='simple', help='BWv3, BWHv3, Lch_Cv2, PV0, Humanv2, HCv2')
parser.add_argument('--time', type=str, default=None, help='BWv3, BWHv3, Lch_Cv2, PV0, Humanv2, HCv2')
parser.add_argument('--exp-name', type=str, default="0:0", help='BWv3, BWHv3, Lch_Cv2, PV0, Humanv2, HCv2')
parser.add_argument('--EnvIdex', type=int, default=2, help='BWv3, BWHv3, Lch_Cv2, PV0, Humanv2, HCv2')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--video_turns', type=int, default=100, help='which model to load')
parser.add_argument('--num_agents', type=int, default=5, help='Decay rate of entropy_coef')
parser.add_argument('--variable_agent', type=str2bool, default=False, help='Decay rate of entropy_coef')
parser.add_argument('--dt', type=float, default=1, help='Decay rate of entropy_coef')
parser.add_argument('--reduce_space', type=str2bool, default=True, help='Share feature extraction layers?')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--T_horizon', type=int, default=2048, help='lenth of long trajectory')
parser.add_argument('--distnum', type=int, default=0, help='0:Beta ; 1:GS_ms;  2: GS_m')
parser.add_argument('--Max_train_steps', type=float, default=1e7, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=5e5, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=1e4, help='Model evaluating interval, in steps.')
parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
parser.add_argument('--clip_rate', type=float, default=0.2, help='PPO Clip rate')
parser.add_argument('--K_epochs', type=int, default=10, help='PPO update times')
parser.add_argument('--net_width', type=int, default=128, help='Hidden net width')
parser.add_argument('--activation', type=str, default='tanh', help='activation function')
parser.add_argument('--a_lr', type=float, default=1.5e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=1.5e-5, help='Learning rate of critic')
parser.add_argument('--l2_reg', type=float, default=1e-3, help='L2 regulization coefficient for Critic')
parser.add_argument('--a_optim_batch_size', type=int, default=64, help='lenth of sliced trajectory of actor')
parser.add_argument('--c_optim_batch_size', type=int, default=64, help='lenth of sliced trajectory of critic')
parser.add_argument('--entropy_coef', type=float, default=1e-3, help='Entropy coefficient of Actor')
parser.add_argument('--entropy_coef_decay', type=float, default=0.99, help='Decay rate of entropy_coef')
parser.add_argument('--share_layer_flag', type=str2bool, default=True, help='Share feature extraction layers?')
parser.add_argument('--multiply_horrizion', type=int, default=1, help='Share feature extraction layers?')
parser.add_argument('--multiply_batch', type=int, default=2, help='Share feature extraction layers?')
parser.add_argument('--reduce_epoch', type=str2bool, default=False, help='Share feature extraction layers?')
parser.add_argument('--curriculum', type=str2bool, default=True, help='gradually increase range')
parser.add_argument('--consider_boid', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--token_query', type=str2bool, default=True, help='tokenize s1 for query')
parser.add_argument('--trans_position', type=str2bool, default=False, help='token input with position')
parser.add_argument('--num_enc', type=int, default=2, help='number of encoders')
parser.add_argument('--num_dec', type=int, default=2, help='number of encoders')

parser.add_argument('--liability', type=str2bool, default=True, help='number of encoders')
parser.add_argument('--collision_free', type=str2bool, default=False, help='number of encoders')
parser.add_argument('--beta_adaptor_coefficient', type=float, default=1.05, help='number of encoders')
parser.add_argument('--beta_base', type=float, default=1.0, help='number of encoders')
parser.add_argument('--level', type=int, default=13, help='Share feature extraction layers?')
parser.add_argument('--num_corridor_in_state', type=int, default=2, help='number of encoders')
parser.add_argument('--corridor_index_awareness', type=str, default='1111', help='indicate the corridor index')
parser.add_argument('--acceleration_max', type=float, default=0.3, help='Learning rate of actor')
parser.add_argument('--velocity_max', type=float, default=1.5, help='Learning rate of actor')
parser.add_argument('--base_difficulty', type=float, default=0.2, help='Learning rate of actor')
parser.add_argument('--ratio', type=float, default=0.5, help='How much percent for torus?')
parser.add_argument('--uniform_state', type=str2bool, default=False, help='number of encoders')
parser.add_argument('--dynamic_minor_radius', type=str2bool, default=False, help='Share feature extraction layers?')
parser.add_argument('--num_obstacles', type=int, default=2, help='number of encoders')
parser.add_argument('--num_ncfos', type=int, default=3, help='number of encoders')
parser.add_argument('--rotate_for_cylinder', type=str2bool, default=True, help='number of encoders')
parser.add_argument('--state_choice', type=int, default=2, help='number of encoders')
parser.add_argument('--rest_awareness', type=str2bool, default=True, help='number of encoders')
parser.add_argument('--cbf', type=str2bool, default=False, help='number of encoders')
parser.add_argument('--with_corridor_index', type=str2bool, default=True, help='number of encoders')
parser.add_argument('--visibility', type=float, default=4.5, help='Learning rate of actor')

opt = parser.parse_args()

# if not opt.reduce_space:
#     assert not opt.rotate_for_cylinder

if opt.corridor_index_awareness:
    opt.corridor_index_awareness = [int(i) for i in opt.corridor_index_awareness]
opt.T_horizon *= opt.multiply_horrizion
opt.a_optim_batch_size *= opt.multiply_batch
opt.c_optim_batch_size *= opt.multiply_batch
opt.save_interval = 2.5e5
# opt.eval_interval = 2.0e4
# opt.eval_interval = 5e4
opt.Max_train_steps = int(opt.Max_train_steps)



if opt.net_model == 'dec':
    opt.num_dec = 3


# opt.c_optim_batch_size *= multiplier


def main():
    write = opt.write  # Use SummaryWriter to record the training.
    render = opt.render
    exp_name = opt.exp_name.split(':')
    if opt.time is None:
        run_name = f"d2move_{int(time.time())}_{exp_name[0]}"
    else:
        run_name = f"d2move_{opt.time}_{exp_name[0]}"

    env = d3.parallel_env(render_mode='rgb_array')
    exp_name = ''.join(exp_name[1:])

    if exp_name is None:
        dir = f'{run_name}'
    else:
        dir = f'{run_name}/{exp_name}'

    if not os.path.exists(dir): os.makedirs(dir)
    logger = setup_logging(f"{run_name}/process_log.txt", logging.INFO)

    max_steps = 500
    T_horizon = opt.T_horizon  # lenth of long trajectory

    Dist = ['Beta', 'GS_ms', 'GS_m']  # type of probility distribution
    distnum = opt.distnum

    Max_train_steps = opt.Max_train_steps
    save_interval = opt.save_interval  # in steps
    eval_interval = opt.eval_interval  # in steps

    seed = opt.seed
    logger.info("Random Seed: {}".format(seed))
    torch.manual_seed(seed)
    #  env.seed(seed)
    # eval_env.seed(seed)
    np.random.seed(seed)

    if write:
        timenow = str(datetime.now())[0:-10]
        # timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]

        if opt.LoadModel:
            opt.ModelIndex = int(opt.ModelIndex)
            file_pattern = f"{opt.LoadFolder}/events.out.tfevents.*"
            files = glob.glob(file_pattern)
            summary_file = files[0]

            if os.path.exists(summary_file):
                logger.info('Summary exists')
                src = summary_file
                dst = f"{run_name}/{exp_name}"
                if not os.path.exists(dst): os.makedirs(dst)
                logger.info(f"src: {src} \n"
                            f"dst: {dst}")
                shutil.copy(src=summary_file, dst=dst)
                writepath = dst
        else:
            logger.info('did not find summary')
            if exp_name is None:
                writepath = f"{run_name}"
            else:
                writepath = f"{run_name}/{exp_name}"
        writer = SummaryWriter(f"{writepath}")
    kwargs = {
        "state_dim": 26,
        "s2_dim": 22,
        "action_dim": 3,
        "env_with_Dead": True,
        "gamma": opt.gamma,
        "lambd": opt.lambd,  # For GAE
        "clip_rate": opt.clip_rate,  # 0.2
        "K_epochs": opt.K_epochs,
        "net_width": opt.net_width,
        "a_lr": opt.a_lr,
        "c_lr": opt.c_lr,
        "dist": Dist[distnum],
        "l2_reg": opt.l2_reg,  # L2 regulization for Critic
        "a_optim_batch_size": opt.a_optim_batch_size,
        "c_optim_batch_size": opt.c_optim_batch_size,
        "entropy_coef": opt.entropy_coef,
        # Entropy Loss for Actor: Large entropy_coef for large exploration, but is harm for convergence.
        "entropy_coef_decay": opt.entropy_coef_decay,
        'activation': opt.activation,
        'share_layer_flag': opt.share_layer_flag,
        'anneal_lr': True,
        'totoal_steps': opt.Max_train_steps,
        'with_position': opt.trans_position,
        'token_query': opt.token_query,
        'num_enc': opt.num_enc,
        'num_dec': opt.num_dec,
        'dir': dir,
        "writer": writer,
        'logger': logger,
        'net_model': opt.net_model,
        'beta_base': opt.beta_base,
    }

    if opt.LoadModel:
        model = PPO(**kwargs)
        model.load(folder=opt.LoadFolder,
                   global_step=opt.ModelIndex)
        total_steps = opt.ModelIndex

        source_path = os.path.join(opt.LoadFolder, 'net_params.json')
        destination_path = os.path.join(os.getcwd(), 'net_params.json')
        shutil.copy(source_path, destination_path)

        source_path = os.path.join(opt.LoadFolder, 'main_params.json')
        destination_path = os.path.join(os.getcwd(), 'main_params.json')
        shutil.copy(source_path, destination_path)
    else:
        save_init_params(name='net_params', **kwargs)
        opt_dict = vars(opt)
        opt_dict['dir'] = dir
        save_init_params(name='main_params', **vars(opt))
        model = PPO(**kwargs)
        total_steps = 0

    traj_lenth = 0

    videoing = False
    ready_for_train = False
    ready_for_log = True
    # some train lap could be longer than eval
    # trained_between_evaluations = True

    # Configure logging for process 1

    logger.info(opt)

    episodes = 0
    total_episode = 0
    trained_times = 0
    extra_save_index = 0
    start_time = time.time()
    epsilon = 0.1

    # base_difficulty = 1 / pow(3, 7)

    if opt.curriculum:
        env_options = {'difficulty': opt.base_difficulty}
    else:
        env_options = {'difficulty': 1}

    while total_steps < Max_train_steps:
        # active_agents = [{'terminated': False, 'trajectory': []} for _ in range(num_agents)]
        steps = 0
        if opt.variable_agent:
            opt.num_agents = np.random.randint(6, 13)
            capacity = 12 + opt.num_obstacles + opt.num_ncfos
        else:
            capacity = opt.num_agents + opt.num_obstacles + opt.num_ncfos
        s, init_info = env.reset(seed=seed,
                                 options=env_options,
                                 num_agents=opt.num_agents,
                                 reduce_space=opt.reduce_space,
                                 level=opt.level,
                                 ratio=opt.ratio,
                                 collision_free=opt.collision_free,
                                 liability=opt.liability,
                                 beta_adaptor_coefficient=opt.beta_adaptor_coefficient,
                                 num_corridor_in_state=opt.num_corridor_in_state,
                                 dt=opt.dt,
                                 capacity=capacity,
                                 corridor_index_awareness=opt.corridor_index_awareness,
                                 velocity_max=opt.velocity_max,
                                 acceleration_max=opt.acceleration_max,
                                 uniform_state=opt.uniform_state,
                                 dynamic_minor_radius=opt.dynamic_minor_radius,
                                 num_obstacles=opt.num_obstacles,
                                 num_ncfo=opt.num_ncfos,
                                 rotate_for_cylinder=opt.rotate_for_cylinder,
                                 epsilon=epsilon,
                                 state_choice=opt.state_choice,
                                 cbf=opt.cbf,
                                 rest_awareness=opt.rest_awareness,
                                 with_corridor_index=opt.with_corridor_index,
                                 visibility=opt.visibility)
        s1 = {agent: s[agent]['self'] for agent in env.agents}
        s2 = {agent: s[agent]['other'] for agent in env.agents}
        if ready_for_log:
            # model.weights_track(total_steps)
            ready_for_log = False
            # trained_between_evaluations = False
            videoing = True
            turns = 0  # opt.video_turns
            scores = 0
            ave_speed = 0
            # ani = vl(opt.video_turns)
            lst_std_variance = []
            env.anima_recording = True
            status_summary = defaultdict(list)
        episodes += 1
        total_episode += 1
        '''Interact & trian'''
        while env.agents:
            traj_lenth += 1
            steps += 1
            s1_lst = [state for agent, state in s1.items()]
            s2_lst = [state for agent, state in s2.items()]
            if render:
                env.render()
                a_lst, logprob_a_lst = model.evaluate(s1_lst, s2_lst)
            else:
                a_lst, logprob_a_lst, alpha, beta = model.select_action(s1_lst, s2_lst)
            logprob_a = {agent: logprob for agent, logprob, in zip(env.agents, logprob_a_lst)}
            a = {agent: a for agent, a in zip(env.agents, a_lst)}

            s_prime, r, terminated, truncated, info = env.step(a)
            s1_prime = {agent: s_prime[agent]['self'] for agent in env.agents}
            s2_prime = {agent: s_prime[agent]['other'] for agent in env.agents}

            done = {agent: terminated[agent] | truncated[agent] for agent in env.agents}

            '''distinguish done between dead|win(dw) and reach env._max_episode_steps(rmax); done = dead|win|rmax'''
            '''dw for TD_target and Adv; done for GAE'''

            dw = {agent: done[agent] and steps != max_steps for agent in env.agents}
            for agent in env.agents:
                agent.trajectory.append([s1[agent],
                                         s2[agent],
                                         a[agent],
                                         r[agent],
                                         s1_prime[agent],
                                         s2_prime[agent],
                                         logprob_a[agent],
                                         done[agent],
                                         dw[agent]])
                if done[agent]:
                    for transition in agent.trajectory:
                        model.put_data(agent, transition)
                        # model.put_data( transition)
            if videoing:
                # ani.put_data(round=turns, agents={agent: agent.position for agent in env.agents})
                for agent in env.agents:
                    if done[agent]:
                        status_summary[init_info['corridor_seq']].append(agent.status)
                        scores += agent.accumulated_reward
                        ave_speed += agent.trajectory_ave_speed if agent.trajectory_ave_speed > 0 else 0

                ## calculate beta distribution variance
                alpha = np.array(alpha.to('cpu'))
                beta = np.array(beta.to('cpu'))
                variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
                # Calculate the standard deviation for each element
                std_dev = np.sqrt(variance)
                # Calculate the average standard deviation for each dimension
                # std_dev = np.squeeze(std_dev)
                lst_std_variance.append(std_dev)

                if all(done.values()):
                    turns += 1
                if turns == opt.video_turns:
                    videoing = False
                    average_score = scores / opt.video_turns / opt.num_agents
                    status_lst = reduce(lambda x, y: x + y, status_summary.values())
                    counter = Counter(status_lst)

                    ave_won_speed = ave_speed / max(1, counter['won'])

                    ## average beta distribution variance
                    stacked_std = np.vstack(lst_std_variance)
                    mean_array = np.mean(stacked_std, axis=0)
                    total_cases = sum(counter.values())
                    if write:
                        for key, values in status_summary.items():
                            writer.add_scalar(f"scenario/{key}", Counter(values)['won'] / len(values), total_steps)

                        won_percent = counter['won'] / total_cases
                        writer.add_scalar('charts/reward_steps', average_score, total_steps)
                        writer.add_scalar('charts/reward_episodes', average_score, total_episode)
                        writer.add_scalar("charts/won_percent", counter['won'] / total_cases, total_steps)
                        writer.add_scalar("fail_reasons/collide_percent", counter['collided'] / total_cases,
                                          total_steps)
                        writer.add_scalar("fail_reasons/collided_UAV", counter['collided_UAV'] / total_cases,
                                          total_steps)
                        writer.add_scalar("fail_reasons/breached_wall", counter['breached_wall'] / total_cases,
                                          total_steps)
                        writer.add_scalar("fail_reasons/breached_c", counter['breached_c'] / total_cases, total_steps)
                        writer.add_scalar('charts/difficulty', env_options['difficulty'], total_steps)
                        writer.add_scalar('charts/uni_r_steps', average_score * min(1, env_options['difficulty']),
                                          total_steps)
                        writer.add_scalar("charts/uni_won_percent", won_percent * min(1, env_options['difficulty']),
                                          total_steps)
                        writer.add_scalar("charts/ave_won_speed", ave_won_speed, total_steps)
                        writer.add_scalar("charts/unified_speed", ave_speed / total_cases, total_steps)
                        writer.add_scalar("dist/beta_std_phi", mean_array[2], total_steps)
                        writer.add_scalar("dist/beta_std_theta", mean_array[1], total_steps)
                        writer.add_scalar("dist/beta_std_r", mean_array[0], total_steps)
                    logger.info(
                        f"seed:{seed}, steps:{int(total_steps / 1000)}k, {env_options['difficulty']}, won:{round(counter['won'] / total_cases * 100, 1)}%, s:{round(average_score, 2)}, won_speed:{round(ave_won_speed, 3)}, status: {counter}, \n {opt.exp_name}")

                    if opt.curriculum:
                        # difficulty 1 is np.pi/2;   1.2 corresponds to slightly larger than
                        maxDiff = 1.0
                        diffThreshold = 0.6
                        if env_options['difficulty'] < maxDiff and won_percent >= diffThreshold:
                            if env_options['difficulty'] == 1 and epsilon > 1e-5:
                                epsilon /= 2
                            else:
                                epsilon = 0
                            env_options['difficulty'] = min(env_options['difficulty'] + 0.1, maxDiff)

            # remove
            env.agents = [agent for agent in env.agents if not done[agent]]
            s1 = {agent: s1_prime[agent] for agent in env.agents}
            s2 = {agent: s2_prime[agent] for agent in env.agents}

            '''update if its time'''

            if traj_lenth % T_horizon == 0:
                ready_for_train = True
            if ready_for_train and not env.agents:
                ready_for_train = False
                # trained_between_evaluations = True
                if opt.reduce_epoch:
                    epoches = int(np.power(episodes, 1 / 2.3))
                else:
                    epoches = opt.K_epochs

                model.train(total_steps, epoches)
                # model.save(total_steps, extra_save_index)
                extra_save_index += 1
                trained_times += 1
                logger.info(f"{episodes}, {epoches}, {opt.exp_name}")
                writer.add_scalar('weights/episodes', episodes, total_steps)
                writer.add_scalar('weights/epoches', epoches, total_steps)
                traj_lenth = 0
                episodes = 0
            total_steps += 1

            '''record & log'''
            if total_steps % save_interval == 0:
                model.save(total_steps)
            if trained_times == 2:
                ready_for_log = True
                trained_times = 0

            # if total_steps % 10 == 0:
            #     logger.info(total_steps, exp_name)
    env.close()


if __name__ == '__main__':
    log_file = "error_log.txt"
    log_format = "%(asctime)s [%(levelname)s]: %(message)s"

    main()
