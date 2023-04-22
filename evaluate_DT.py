import os
import pickle
import gym
import time
import torch
import torch.nn as nn
import numpy as np
import numpy.random as rd
from torch.nn.modules import loss
from random_generator_battery import ESSEnv
import pandas as pd

from tools import Arguments, test_one_episode_DT, ReplayBuffer, optimization_base_result
from agent import AgentDDPG
from random_generator_battery import ESSEnv


def update_buffer(_trajectory):
    ten_state = torch.as_tensor([item[0]
                                for item in _trajectory], dtype=torch.float32)
    ary_other = torch.as_tensor([item[1] for item in _trajectory])
    ary_other[:, 0] = ary_other[:, 0]   # ten_reward
    # ten_mask = (1.0 - ary_done) * gamma
    ary_other[:, 1] = (1.0 - ary_other[:, 1]) * gamma

    buffer.extend_buffer(ten_state, ary_other)

    _steps = ten_state.shape[0]
    _r_exp = ary_other[:, 0].mean()  # other = (reward, mask, action)
    return _steps, _r_exp


def evaluate_one_episode(model=None,eval_times = 30):

    ratios = []

    for i in range(eval_times):
        args = Arguments()
        agent_name = "DT"
        args.env = ESSEnv()
        args.cwd = agent_name
        # agent.act.load_state_dict(torch.load(act_save_path))
        # print('parameters have been reload and test')
        record = test_one_episode_DT(args.env, "cpu", model)
        # print(record['information'])
        eval_data = pd.DataFrame(record['information'])
        eval_data.columns = ['time_step', 'price', 'netload', 'action', 'real_action',
                            'soc', 'battery', 'gen1', 'gen2', 'gen3', 'unbalance', 'operation_cost']
        eval_data.to_csv(f'{args.cwd}/eval_data.csv', index=False)
        # print(eval_data)

        '''compare with pyomo data and results'''
        if args.compare_with_pyomo:
            month = record['init_info'][0][0]
            day = record['init_info'][0][1]
            initial_soc = record['init_info'][0][3]
            # print(initial_soc)
            base_result = optimization_base_result(
                args.env, month, day, initial_soc)
            # print(base_result)

        if args.save_test_data:
            test_data_save_path = f'{args.cwd}/test_data.pkl'
            with open(test_data_save_path, 'wb') as tf:
                pickle.dump(record, tf)

        args.plot_on = False
        if args.plot_on:
            from plotDRL import PlotArgs, make_dir, plot_evaluation_information, plot_optimization_result
            plot_args = PlotArgs()
            plot_args.feature_change = ''
            args.cwd = agent_name
            plot_dir = make_dir(args.cwd, plot_args.feature_change)
            plot_optimization_result(base_result, plot_dir)
            plot_evaluation_information(args.cwd + '/'+'test_data.pkl', plot_dir)

        '''compare the different cost get from pyomo and DT'''
        ratio = sum(eval_data['operation_cost'])/sum(base_result['step_cost'])
        ratio_unbalance = sum(eval_data['unbalance'])/sum(base_result['step_cost'])
        # print(sum(eval_data['operation_cost']), sum(eval_data['unbalance']))
        # print(sum(base_result['step_cost']), sum(base_result['step_cost']))
        ratios.append(ratio)
        
    print(ratios)
    ratios = np.array(ratios)
    return {"ratio": ratios.mean()}
    # return {"ratio": ratios.mean(), "ratio_unbalance": ratio_unbalance, "unbalance": sum(eval_data['unbalance']), "operation_cost": sum(eval_data['operation_cost']), "base_cost": sum(base_result['step_cost'])}


if __name__ == '__main__':

    args = Arguments()
    '''here record real unbalance'''
    reward_record = {'episode': [], 'steps': [],
                     'mean_episode_reward': [], 'unbalance': []}
    loss_record = {'episode': [], 'steps': [],
                   'critic_loss': [], 'actor_loss': [], 'entropy_loss': []}
    args.visible_gpu = '2'
    for seed in args.random_seed_list:
        args.random_seed = seed
        # set different seed
        args.agent = AgentDDPG()
        agent_name = f'{args.agent.__class__.__name__}'
        args.agent.cri_target = True
        args.env = ESSEnv()
        # creat lists of lists/or creat a long list?

        args.init_before_training(if_main=True)
        '''init agent and environment'''
        agent = args.agent
        env = args.env
        agent.init(
            args.net_dim, env.state_space.shape[0], env.action_space.shape[0], args.learning_rate, args.if_per_or_gae)
        print(
            f'state_dim:{env.state_space.shape[0]},action_dim:{env.action_space.shape[0]}')

        '''init replay buffer'''
        buffer = ReplayBuffer(max_len=args.max_memo, state_dim=env.state_space.shape[0],
                              action_dim=env.action_space.shape[0])
        '''start training'''
        cwd = args.cwd
        gamma = args.gamma
        batch_size = args.batch_size  # how much data should be used to update net
        target_step = args.target_step  # how manysteps of one episode should stop
        # reward_scale=args.reward_scale# here we use it as 1# we dont need this in our model
        # how many times should update for one batch size data
        repeat_times = args.repeat_times
        # if_allow_break = args.if_allow_break
        soft_update_tau = args.soft_update_tau
        # get the first experience from
        agent.state = env.reset()
        # trajectory=agent.explore_env(env,target_step)
        # update_buffer(trajectory)
        '''collect data and train and update network'''
        num_episode = args.num_episode

        ##
        # args.save_network=False
        args.test_network = True
        args.save_test_data = True
        args.compare_with_pyomo = True
        #

        generate_trajectories = False

        trajectory_list = []
        counter = 0

        while generate_trajectories:

            # print(f'counter:{counter}')
            with torch.no_grad():
                trajectory = agent.explore_env(env, target_step)

                trajectory_i = {"observations": [],
                                "actions": [], "rewards": [], "dones": []}

                for state_s in trajectory:
                    trajectory_i["observations"].append(state_s[0])
                    trajectory_i["actions"].append(state_s[1][2:6])

                    reward_mode = 'normal'
                    if reward_mode == 'return_to_go':
                        trajectory_i["rewards"].append(sum(trajectory_i["rewards"]) + state_s[1][0])
                    else:
                        trajectory_i["rewards"].append(state_s[1][0])
                    trajectory_i["dones"].append(state_s[1][1])

                trajectory_i["observations"] = np.array(
                    trajectory_i["observations"])
                trajectory_i["actions"] = np.array(trajectory_i["actions"])
                trajectory_i["rewards"] = np.array(trajectory_i["rewards"])
                trajectory_i["dones"] = np.array(trajectory_i["dones"])
                # print(trajectory_i)
                trajectory_list.append(trajectory_i)
        
                # print(buffer.now_len)
                counter += 1
                if counter % 10000 == 0:
                    print(f'counter:{counter}')
                    f = open('trajectories', 'wb')
                    # source, destination
                    pickle.dump(trajectory_list, f)
                    f.close()

            if counter > 1000000:
                print("====================================")
                print(trajectory_list[0])
                print(len(trajectory_list))
                print('Finished trajectory generating!')

                f = open('trajectories_sh', 'wb')
                # source, destination
                pickle.dump(trajectory_list, f)
                f.close()
                exit(0)


    evaluation_episodes = 1
    ratios = []
    for i in range(evaluation_episodes):

        if args.test_network:
            args.cwd = agent_name
            # agent.act.load_state_dict(torch.load(act_save_path))
            # print('parameters have been reload and test')
            record = test_one_episode_DT(env, agent.device)
            # print(record['information'])
            eval_data = pd.DataFrame(record['information'])
            eval_data.columns = ['time_step', 'price', 'netload', 'action', 'real_action',
                                'soc', 'battery', 'gen1', 'gen2', 'gen3', 'unbalance', 'operation_cost']
            eval_data.to_csv(f'{args.cwd}/eval_data.csv', index=False)
            # print(eval_data)
        if args.save_test_data:
            test_data_save_path = f'{args.cwd}/test_data.pkl'
            with open(test_data_save_path, 'wb') as tf:
                pickle.dump(record, tf)

        '''compare with pyomo data and results'''
        if args.compare_with_pyomo:
            month = record['init_info'][0][0]
            day = record['init_info'][0][1]
            initial_soc = record['init_info'][0][3]
            # print(initial_soc)
            base_result = optimization_base_result(env, month, day, initial_soc)
            print(base_result)

        args.plot_on = True
        if args.plot_on:
            from plotDRL import PlotArgs, make_dir, plot_evaluation_information, plot_optimization_result
            plot_args = PlotArgs()
            plot_args.feature_change = ''
            args.cwd = agent_name
            plot_dir = make_dir(args.cwd, plot_args.feature_change)
            plot_optimization_result(base_result, plot_dir)
            plot_evaluation_information(args.cwd+'/'+'test_data.pkl', plot_dir)

        '''compare the different cost get from pyomo and DT'''
        ration = sum(eval_data['operation_cost'])/sum(base_result['step_cost'])
        # print(sum(eval_data['operation_cost']))
        # print(sum(base_result['step_cost']))
        print(ration)
        ratios.append(ration)
     
    ratios = np.array(ratios)
    print('============================')
    print(ratios)
    print(ratios.mean())
    
    
