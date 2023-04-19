import numpy as np
import torch

import pandas as pd
import random
from icecream import ic
import matplotlib.pyplot as plt

def instantiate_environment():
    # Set seed for reproducibility
    np.random.seed(42)

    # Set the number of data points to generate (48 points for 24 hours)
    n_points = 48

    # Generate time axis
    time_axis = pd.date_range(start='2023-03-31 00:00:00', end='2023-03-31 23:59:00', freq='30min')

    # Generate the data for each dataset
    pv_generation = np.zeros(n_points)
    for i in range(n_points):
        if time_axis[i].hour >= 6 and time_axis[i].hour <= 18:
            pv_generation[i] = np.random.normal(2500, 500)
        else:
            pv_generation[i] = np.random.normal(500, 100)

    demand = np.zeros(n_points)
    for i in range(n_points):
        if time_axis[i].hour >= 6 and time_axis[i].hour <= 10:
            demand[i] = np.random.normal(6000, 1000)
        elif time_axis[i].hour >= 17 and time_axis[i].hour <= 22:
            demand[i] = np.random.normal(8000, 1000)
        else:
            demand[i] = np.random.normal(4000, 1000)

    price = np.zeros(n_points)
    for i in range(n_points):
        if time_axis[i].hour >= 6 and time_axis[i].hour <= 10:
            price[i] = np.random.normal(50, 5)
        elif time_axis[i].hour >= 17 and time_axis[i].hour <= 22:
            price[i] = np.random.normal(70, 5)
        else:
            price[i] = np.random.normal(30, 5)

    # Plot the data
    # fig, ax = plt.subplots(3, 1, figsize=(16, 10))

    # ax[0].plot(time_axis, pv_generation, color='red')
    # ax[0].set_ylabel('PV Generation')

    # ax[1].plot(time_axis, demand, color='blue')
    # ax[1].set_ylabel('Demand')

    # ax[2].plot(time_axis, price, color='green')
    # ax[2].set_ylabel('Price')
    # ax[2].set_xlabel('Time')

    # plt.show()

    return pv_generation, demand, price

def visualize_charge(pv_generation,demand, price, charge, balance):

    time_axis = pd.date_range(start='2023-03-31 00:00:00', end='2023-03-31 23:59:00', freq='30min')
    # Plot the data
    fig, ax = plt.subplots(5, 1, figsize=(16, 10))

    ax[0].plot(time_axis, pv_generation, color='red')
    ax[0].set_ylabel('PV Generation')

    ax[1].plot(time_axis, demand, color='blue')
    ax[1].set_ylabel('Demand')

    ax[2].plot(time_axis, price, color='green')
    ax[2].set_ylabel('Price')

    ax[3].plot(time_axis, charge, color='gray')
    ax[3].set_ylabel('Battery Charge Level')

    ax[4].plot(time_axis, balance, color='orange')
    ax[4].set_ylabel('Balance')

    ax[4].set_xlabel('Time')

    plt.show()


class Environment():
    def __init__(self):   
        print("Initialize Environment")
        self.t = 0
        self.price = 0
        self.demand = 200
        self.pv_generation = 5  
        self.n_points = 48
        self.pv_generation, self.demand, self.price = instantiate_environment()

    def step(self):
        self.t += 1
        if self.t >= self.n_points:
            self.t = 0      

    def get_price(self):
        return self.price[self.t]

    def get_state(self):
        return (self.t,self.pv_generation[self.t],self.demand[self.t],self.price[self.t])

class BatteryAgent():

    def __init__(self):   
        # print("Initialize Battery")
        self.balance = 0
        self.charge = 0
        self.max_charge = 200
        self.charge_ratio = 30
        self.discharge_ratio = 25   

    def get_state(self,env):
        state_env = env.get_state()
        # print(*state_env,self.balance,self.charge)

        return *state_env, self.balance,self.charge

    def make_action(self,action_index,env):

        price = env.get_price()
        action = ["charge","discharge","do_nothing"][action_index]

        if action == "charge":
            if self.charge + self.charge_ratio > self.max_charge:
                self.charge = self.max_charge
                return -(self.max_charge - self.charge) * price
            else:
                self.charge += self.charge_ratio 
                return -self.charge_ratio * price

        elif action == "discharge":
            if self.charge - self.discharge_ratio < 0:
                self.charge = 0
                return -self.charge * price
            else:
                self.charge -= self.discharge_ratio 
                return self.discharge_ratio * price

        else:
            return 0

    def get_charge(self):
        return self.charge






def evaluate_episode(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    
    env = Environment()
    battery = BatteryAgent()
    state = battery.get_state()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        # state, reward, done, _ = env.step(action)
        reward = battery.make_action(action,env)
        state = battery.get_state()
        done = env.step()

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


def evaluate_episode_rtg(
        # env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
    ):
    

    # print("Start Eval")
    env = Environment()
    battery = BatteryAgent()
    state = np.array(battery.get_state(env))
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    # state = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []
    charge_level = []
    balance_level = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action

        # if t == max_ep_len-1:
        #     ic(actions)
        #     ic(np.argmax(actions.detach().cpu().numpy(),1))                    

        pr_action = action
        action = action.detach().cpu().numpy()
        # ic(pr_action,"=>",action)
        # ic(np.argmax(action))
        # state, reward, done, _ = env.step(action)        
        reward = np.array(battery.make_action(np.argmax(action),env))
        battery.balance += float(reward)
        charge_level.append(battery.charge)
        balance_level.append(battery.balance)

        state = np.array(battery.get_state(env))
        done = env.step()

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = torch.from_numpy(reward).to(device)

        if mode != 'delayed':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break

    # print(charge_level)

    visualize_charge(env.pv_generation,env.demand,env.price,charge_level,balance_level)
    return episode_return, episode_length
