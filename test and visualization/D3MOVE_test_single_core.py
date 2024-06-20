# from pettingzoo.mpe import simple_adversary_v3
import collections
import json

import air_corridor.d3.scenario.D3shapeMove as d3
from air_corridor.tools.util import load_init_params
from air_corridor.tools.visualization import Visualization as vl
from rl_multi_3d_trans.ppo import PPO

env = d3.parallel_env(render_mode="")

# trained model info
loadModel = True
if loadModel:
    result = '../trained_models/basic/'
    loadFolder = f"{result}HTransRL-T"
    modelIndex = ''
    net_model = 'fc10_3e'
    trained_level = 19

kwargs = load_init_params(name='net_params', dir=loadFolder)
opt = load_init_params(name='main_params', dir=loadFolder)
with open(f"{loadFolder}/net_params.json", 'r') as json_file:
    kwargs1 = json.load(json_file)
    # opt=json.load
kwargs['net_model'] = net_model
model = PPO(**kwargs)
model.load(folder=loadFolder, global_step=modelIndex, dir=loadFolder)

opt = load_init_params(name='main_params', dir=loadFolder)
max_round = 10

ani_bool = True
if ani_bool:
    ani = vl(max_round + 1, to_base=False)

status = {}
num_agents = 12
# level 14: cttc;
# level 20: cttcttc
# level 21: cttcttcttc
level_key = 21
level_dic = {14: 'cttc',
             20: 'cttcttc',
             21: 'cttcttcttc'}
for num_agents in [12]:
    print(f"simulation in {level_dic[level_key]} with {num_agents} UAVs")
    for i in range(max_round + 1):

        '''
        training scenarios can be different from test scenarios, so num_corridor_in_state and corridor_index_awareness
        need to match the setting for UAV during training.
        '''
        # level 14: cttc;
        # level 20: cttcttc
        # level 21: cttcttcttc
        s, infos = env.reset(num_agents=num_agents,
                             num_obstacles=4,
                             num_ncfo=3,
                             level=level_key,
                             dt=opt['dt'],
                             beta_adaptor_coefficient=opt['beta_adaptor_coefficient'],
                             test=True)
        current_actions = {}
        step = 0
        agents = env.agents
        if ani_bool:
            ani.put_data(agents={agent: agent.position for agent in env.agents},
                         ncfos={ncfo: ncfo.position for ncfo in env.ncfos}, corridors=env.corridors, round=i)

        while env.agents:
            if loadModel:
                s1 = {agent: s[agent]['self'] for agent in env.agents}
                s2 = {agent: s[agent]['other'] for agent in env.agents}
                s1_lst = [state for agent, state in s1.items()]
                s2_lst = [state for agent, state in s2.items()]
                a_lst, logprob_a_lst = model.evaluate(s1_lst, s2_lst, deterministic=True)
                actions = {agent: a for agent, a in zip(env.agents, a_lst)}
            else:
                actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            action_updated = False
            s, rewards, terminations, truncations, infos = env.step(actions)
            if ani_bool:
                ani.put_data(round=i, agents={agent: agent.position for agent in env.agents},
                             ncfos={ncfo: ncfo.position for ncfo in env.ncfos})
            # print(rewards)

            for agent in env.agents:
                if agent.status != 'Normal' and agent not in status:
                    status[agent] = agent.status
            env.agents = [agent for agent in env.agents if not agent.terminated]
            step += 1
            # print(step)
        if 1:  # i % 10 != 0:
            state_count = collections.Counter(status.values())
            print(f"{i}/{max_round} - {state_count}")
            print(state_count['won'] / max(sum(state_count.values()), 1))
            #
    if ani_bool:
        ani.show_animation(gif=True, save_to=f"{level_dic[level_key]}_{num_agents}")
env.close()
