# Write code that uses evolutionary strategies to train a neural network on the environment

import numpy as np
import gym
import time
import pickle
import matplotlib.pyplot as plt
import random
import math
import copy
import os
import sys
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

from evaluate_DT import evaluate_one_episode


# generate a simple NN with 2 hidden layers s input nodes and a single output node
class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x


# for many iterations do the following:
#  1. sample a population of solutions
# 2. evaluate the fitness of each solution
# 3. update the mean and standard deviation of the population
# 4. repeat
# 5. return the best solution found
# 6. plot the fitness of the best solution found at each iteration
action_dim = 4
state_dim = 9

hidden_dim = 64
population_size = 5
epochs = 100
sigma = 0.1
lr = 0.001


def ES(num_iterations, population_size, sigma, learning_rate, render=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # initialize a list of size population size of NNs with random weights
    population = []

    base_model = NN(state_dim, hidden_dim, action_dim).to(device)

    denominator = 0
    for i in range(population_size):
        denominator += math.log(population_size + 0.5) - math.log(i+1)

    zeros = []
    for param in base_model.parameters():
        zeros.append(torch.zeros(param.shape, device=device))

    for i in range(population_size):
        model = NN(state_dim, hidden_dim, action_dim).to(device)
        model.load_state_dict(base_model.state_dict())
        population.append(model)

    for iter in range(num_iterations):
        results = []
        samples = []

        for i in tqdm(range(population_size)):

            gen_samples = []
            for param in population[i].parameters():
                sample = torch.normal(mean=0, std=torch.ones(
                    param.shape, device=device))
                param.data = param.data + sigma * sample
                gen_samples.append(sample)

            samples.append(gen_samples)
            results.append(evaluate_one_episode(
                population[i], simple_model=True)['ratio'])

        print(
            f'iter {iter} mean {np.mean(results)}, Best: {np.max(results)}, Worst: {np.min(results)}')
        # print(samples)
        # scalar x to torch tensor
        results = torch.tensor(results, device=device)

        # generate a list with the ranking of each element in the results list
        # e.g. [0, 1, 2, 3, 4] for population size 5
        # print(results)
        ranking = torch.argsort(results, descending=True)
        # print(ranking)
        weights = [0]*population_size
        for i in range(population_size):
            weights[ranking[i]] = ((math.log(population_size + 0.5) -
                                    math.log(population_size - i)) / denominator)
        # print(weights)

        gradients = [z.detach().clone() for z in zeros]
        for i in range(population_size):
            for j in range(len(gradients)):
                gradients[j] += torch.mul(samples[i][j], weights[i])

        for j, param in enumerate(base_model.parameters()):
            param.data = param.data + learning_rate / \
                (population_size * sigma) * gradients[j]


env = None
ES(num_iterations=epochs, population_size=population_size,
   sigma=sigma, learning_rate=lr, render=False)
