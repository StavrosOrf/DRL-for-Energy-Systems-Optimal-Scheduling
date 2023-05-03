import os
import pickle
import time
import torch
import torch.nn as nn
import numpy as np
import numpy.random as rd
from torch.nn.modules import loss
from random_generator_battery import ESSEnv
import pandas as pd
from tqdm import tqdm
import math

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


def generate_best_solutions():
    MONTHS_LEN = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    solutions_list = []
    solutions_number = 10000

    file_name = 'eval_solutions.pkl'

    args = Arguments()
    args.agent = AgentDDPG()
    args.env = ESSEnv()

    args.init_before_training(if_main=True)
    '''init agent and environment'''
    agent = args.agent
    env = args.env
    agent.init(
        args.net_dim, env.state_space.shape[0], env.action_space.shape[0], args.learning_rate, args.if_per_or_gae)
    agent.state = env.reset()

    for counter in tqdm(range(solutions_number)):

        month = np.random.randint(1, 13)
        day = np.random.randint(1, MONTHS_LEN[month-1]-1)
        initial_soc = round(np.random.uniform(0.2, 0.8), 2)
        # print(f'month:{month}, day:{day}, initial_soc:{initial_soc}')

        base_result = optimization_base_result(
            env, month, day, initial_soc)

        total_cost = base_result['step_cost'].sum()
        total_unbalance = abs(
            base_result['load'].sum() - base_result['netload'].sum())

        # print(base_result['step_cost'].sum())
        # print(base_result['load'].sum() - base_result['netload'].sum())
        # print(base_result)
        solution = {'month': month, 'day': day, 'initial_soc': initial_soc,
                    'total_unbalance': total_unbalance, 'total_operation_cost': total_cost}
        solutions_list.append(solution)

    f = open(file_name, 'wb')
    pickle.dump(solutions_list, f)
    f.close()
    print('solutions have been generated and saved')
    print(solutions_list)


def evaluate_one_episode(model=None, state_mean=None,
                         state_std=None, simple_model=False, eval_times=100, use_best_solutions=True, results_in = None):

    ratios_cost = []
    ratios_unbalance = []

    if use_best_solutions:
        dataset_path = f'eval_solutions.pkl'
        with open(dataset_path, 'rb') as f:
            best_solutions = pickle.load(f)

    args = Arguments()
    agent_name = "DT"
    args.env = ESSEnv()
    args.cwd = agent_name

    # for i in tqdm(range(eval_times)):
    for i in range(eval_times):
        # for i in tqdm(range(1000)):

        record = test_one_episode_DT(
            args.env, device="cuda", model_init=model, simple_model=simple_model, month=best_solutions[i]['month'], day=best_solutions[i]['day'], initial_soc=best_solutions[i]['initial_soc'], state_mean=state_mean,
            state_std=state_std)
        # exit()
        eval_data = pd.DataFrame(record['information'])
        eval_data.columns = ['time_step', 'price', 'netload', 'action', 'real_action',
                             'soc', 'battery', 'gen1', 'gen2', 'gen3', 'unbalance', 'operation_cost']
        '''compare with pyomo data and results'''
        if not use_best_solutions:
            month = record['init_info'][0][0]
            day = record['init_info'][0][1]
            initial_soc = record['init_info'][0][3]
            # print(initial_soc)
            base_result = optimization_base_result(
                args.env, month, day, initial_soc)
            # print(base_result)
            ratio = sum(eval_data['operation_cost']) / \
                sum(base_result['step_cost'])
            ratio_unbalance = sum(
                eval_data['unbalance']) / abs(base_result['netload'].sum()-base_result['load'].sum())
        else:
            # print(f"total_operation_cost: {best_solutions[i]['total_operation_cost']} ")
            # print(f"total_unbalance: {best_solutions[i]['total_unbalance']} ")
            # print(f"sum(eval_data['operation_cost']): {sum(eval_data['operation_cost'])} ")
            # print(f"sum(eval_data['unbalance']): {sum(eval_data['unbalance'])} ")

            ratio = abs(sum(eval_data['operation_cost']) /
                        best_solutions[i]['total_operation_cost'])
            ratio_unbalance = abs(sum(
                eval_data['unbalance']) / best_solutions[i]['total_unbalance'])

        ratios_cost.append(ratio)
        ratios_unbalance.append(ratio_unbalance)

    ratios_cost = np.array(ratios_cost)
    ratios_unbalance = np.array(ratios_unbalance)

    # print(
    #     f"index: {ratios_cost.argmin()}, max: {ratios_cost.max()}, min: {ratios_cost.min()}")


    results = {"ratio": ratios_cost.mean(), "ratio_cost_std": ratios_cost.std(), "ratio_cost_median": np.median(ratios_cost),
            "ratio_unbalance": ratios_unbalance.mean(), "ratio_unbalance_std": ratios_unbalance.std(), "ratio_unbalance_median": np.median(ratios_unbalance),
            "ratio_unbalance_max": ratios_unbalance.max(), "ratio_unbalance_min": ratios_unbalance.min(),
            "ratio_cost_max": ratios_cost.max(), "ratio_cost_min": ratios_cost.min()}
    
    if results_in is not None:
        results_in.append(results['ratio'])
        # print(results_in)
    return results


if __name__ == '__main__':

    # generate_best_solutions()
    results = evaluate_one_episode(eval_times=1000, use_best_solutions=True)

    print(results)

    # pickle results
    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)

    # load results
    # with open('results.pkl', 'rb') as f:
    #     results = pickle.load(f)

    # results = pd.DataFrame(results)
    exit(0)

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
            base_result = optimization_base_result(
                env, month, day, initial_soc)
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
