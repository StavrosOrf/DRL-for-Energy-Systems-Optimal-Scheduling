import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import pickle


def instantiate_environment():
    # Set seed for reproducibility
    np.random.seed(42)

    # Set the number of data points to generate (48 points for 24 hours)
    n_points = 48

    # Generate time axis
    time_axis = pd.date_range(
        start='2023-03-31 00:00:00', end='2023-03-31 23:59:00', freq='30min')

    # Generate the data for each dataset``
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
            return True

        return False

    def get_price(self):
        return self.price[self.t]

    def get_state(self):
        return (self.t, self.pv_generation[self.t], self.demand[self.t], self.price[self.t])


class BatteryAgent():

    def __init__(self):
        # print("Initialize Battery")
        self.balance = 0
        self.charge = 0
        self.max_charge = 200
        self.charge_ratio = 30
        self.discharge_ratio = 25

    def get_state(self, env):
        state_env = env.get_state()
        # print(*state_env,self.balance,self.charge)

        return *state_env, self.balance, self.charge

    def make_action(self, action_index, env):

        price = env.get_price()
        action = ["charge", "discharge", "do_nothing"][action_index]

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
                return self.charge * price
            else:
                self.charge -= self.discharge_ratio
                return self.discharge_ratio * price

        else:
            return 0

    def get_charge(self):
        return self.charge


env = Environment()
battery = BatteryAgent()

n_trajectories = 2
generate_trajectories = True

if generate_trajectories:
    trajectory_list = []

    for i in range(n_trajectories):
        battery = BatteryAgent()
        trajectory = {"observations": [],
                      "actions": [], "rewards": [], "dones": []}

        for t in range(env.n_points):
            state = battery.get_state(env)
            action_index = random.randint(0, 2)
            reward = battery.make_action(action_index, env)

            env.step()

            action = np.zeros((1, 3))
            action[0, action_index] = 1

            trajectory["observations"].append(state)
            trajectory["actions"].append((action[0]))
            trajectory["rewards"].append(reward)
            # trajectory["terminals"].append(0)
            if t == env.n_points-1:
                trajectory["dones"].append(1)
            else:
                trajectory["dones"].append(0)

            battery.balance += float(reward)

        print(i)
        trajectory["observations"] = np.array(trajectory["observations"])
        trajectory["actions"] = np.array(trajectory["actions"])
        trajectory["rewards"] = np.array(trajectory["rewards"])
        trajectory["dones"] = np.array(trajectory["dones"])
        trajectory_list.append(trajectory)

    print(len(trajectory_list))

    f = open('trajectories', 'wb')

    # source, destination
    pickle.dump(trajectory_list, f)
    f.close()

dbfile = open('trajectories', 'rb')
trajectory_list = pickle.load(dbfile)
# for keys in trajectory_list:
#     print(keys)
print(len(trajectory_list))
print(trajectory_list[0])
dbfile.close()
