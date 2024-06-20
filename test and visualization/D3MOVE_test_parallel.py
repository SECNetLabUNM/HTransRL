# from pettingzoo.mpe import simple_adversary_v3
# file setup for federated test

import collections
import json
import multiprocessing
import time
from itertools import product

import air_corridor.d3.scenario.D3shapeMove as d3
from air_corridor.tools.util import load_init_params
from rl_multi_3d_trans.ppo import PPO

env = d3.parallel_env(render_mode="")
max_round = 1000


def process_combination(combination):
    model_key, num_agents, level, turbulence_variance, visibility = combination
    result = '../trained_models/basic/'
    if model_key == '3-1':
        loadFolder = f"{result}HD-F"
        modelIndex = ''
        net_model = 'dec'
        trained_level = 19
    if model_key == '3-2':
        loadFolder = f"{result}HD-T"
        modelIndex = ''
        net_model = 'dec'
        trained_level = 19
    if model_key == '3-3':
        loadFolder = f"{result}HTransRL-F"
        modelIndex = ''
        net_model = 'fc10_3e'
        trained_level = 19
    if model_key == '3-4':
        loadFolder = f"{result}HTransRL-T"
        modelIndex = ''
        net_model = 'fc10_3e'
        trained_level = 19
    if model_key == '3-5':
        loadFolder = f"{result}DS-F"
        modelIndex = ''
        net_model = 'fc12'
        trained_level = 19
    if model_key == '3-6':
        loadFolder = f"{result}DS-T"
        modelIndex = ''
        net_model = 'fc12'
        trained_level = 19

    kwargs = load_init_params(name='net_params', dir=loadFolder)
    opt = load_init_params(name='main_params', dir=loadFolder)
    kwargs['net_model'] = net_model
    model = PPO(**kwargs)
    try:
        model.load(folder=loadFolder, global_step=modelIndex)
    except:
        return
    # ani = vl(max_round, to_base=False)
    status = {}
    accumulated_won_speed = 0
    accumulated_won_time = 0
    for z in range(max_round):
        # print(f"{i}/{max_round}")
        '''
        training scenarios can be different from test scenarios, so num_corridor_in_state and corridor_index_awareness
        need to match the setting for UAV during training.
        '''
        s, infos = env.reset(num_agents=num_agents,
                             num_obstacles=4,
                             num_ncfo=3,
                             level=level,
                             dt=opt['dt'],
                             beta_adaptor_coefficient=opt['beta_adaptor_coefficient'],
                             test=True,
                             turbulence_variance=turbulence_variance,
                             visibility=visibility)
        step = 0
        # ani.put_data(agents={agent: agent.position for agent in env.agents}, corridors=env.corridors, round=z)
        while env.agents:
            s1 = {agent: s[agent]['self'] for agent in env.agents}
            s2 = {agent: s[agent]['other'] for agent in env.agents}
            s1_lst = [state for agent, state in s1.items()]
            s2_lst = [state for agent, state in s2.items()]
            a_lst, logprob_a_lst = model.evaluate(s1_lst, s2_lst)
            # a_lst, logprob_a_lst, alpha, beta = model.select_action(s1_lst, s2_lst)
            actions = {agent: a for agent, a in zip(env.agents, a_lst)}
            s, rewards, terminations, truncations, infos = env.step(actions)
            # ani.put_data(round=z, agents={agent: agent.position for agent in env.agents})
            for agent in env.agents:
                if agent.status != 'Normal' and agent not in status:
                    status[agent] = agent.status
                    if agent.trajectory_ave_speed > 0:
                        accumulated_won_speed += agent.trajectory_ave_speed
                        accumulated_won_time += agent.travel_time

            env.agents = [agent for agent in env.agents if not agent.terminated]
            step += 1
            # print(step)

    state_count = collections.Counter(status.values())
    ave_won_speed = accumulated_won_speed / max(1, state_count['won'])
    ave_won_time = accumulated_won_time / max(1, state_count['won'])
    total_agents = num_agents * max_round
    won_rate = round(state_count['won'] / max(total_agents, 1), 2)
    print(
        f"{combination}  {trained_level}  {state_count['won']}/{total_agents}  {won_rate}, speed:{round(ave_won_speed, 3)}")
    # return model_key, num_agents, level, state_count['won'], won_rate, state_count, ave_won_speed
    return combination, state_count['won'], won_rate, state_count, ave_won_speed, ave_won_time


# ani.show_animation()
# Specify the file name where you want to save the array

def main():
    multiprocessing.set_start_method('spawn', force=True)

    # Your existing setup code

    # dic_id_name = {'1-1': '10-circle',
    #                '1-2': '10-grid',
    #                '2-1': '3e-T6',
    #                '2-2': '3e-F6',
    #                '2-3': 'HD-T6',
    #                '2-4': 'HD-F6',
    #                '3-1': 'HD-F',
    #                '3-2': 'HD-T',
    #                '3-3': 'HTransRL-F',
    #                '3-4': 'HTransRL-T',
    #                '3-5': 'DS-F',
    #                '3-6': 'DS-T',
    #                }

    # for HTransRL vanilla
    model_set = reversed(['3-1', '3-2', '3-3', '3-4', '3-5', '3-6'])

    # 4-(5-8) fed models
    # model_set = reversed(['4-1', '4-5', '4-6', '4-7', '4-8'])
    level_set = reversed([20, 21])
    agents_dic = {6: 0, 9: 1, 12: 2, 18: 3, 24: 4, 36: 5}
    model_dic = {j: i for i, j in enumerate(model_set)}
    level_dic = {j: i for i, j in enumerate(level_set)}
    turbulence_dic = {j: i for i, j in enumerate([0])}
    visibility_dic = {j: i for i, j in enumerate([4.0])}
    combinations = list(product(model_dic.keys(),
                                agents_dic.keys(),
                                level_dic.keys(),
                                turbulence_dic.keys(),
                                visibility_dic.keys()))

    # Limit the number of processes to 4
    with multiprocessing.Pool(processes=9) as pool:
        results = pool.map(process_combination, combinations)
        # print(results)

    with open('plot/test_data_time.json', 'w') as file:
        json.dump(results, file, indent=4)

    # with open('./plot/test_data_ori.json', 'r') as file:
    #     results = json.load(file)

    # won_matrix = np.zeros([len(model_dic.keys()), len(agents_dic.keys()), len(level_dic.keys())])
    # won_rate_matrix = np.zeros([len(model_dic.keys()), len(agents_dic.keys()), len(level_dic.keys())])
    # for result in results:
    #     if result:
    #         model_key, num_agents, level_key, won_value, won_rate, status_count, ave_won_speed = result
    #         won_matrix[model_dic[model_key], agents_dic[num_agents], level_dic[level_key]] = won_value
    #         won_rate_matrix[model_dic[model_key], agents_dic[num_agents], level_dic[level_key]] = won_rate
    #
    # arrays = {'model': model_dic,
    #           'agent': agents_dic,
    #           'level': level_dic,
    #           'won_times': won_matrix,
    #           'won_rate': won_rate_matrix,
    #           'status_count': status_count,
    #           'ave_won_speed': ave_won_speed}
    # with open('./plot/array.pkl', 'wb') as f:
    #     pickle.dump(arrays, f)


if __name__ == "__main__":
    begin = time.time()
    main()
    print(time.time() - begin)
