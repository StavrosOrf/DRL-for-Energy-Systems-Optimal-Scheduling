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
import concurrent.futures
import time

from decision_transformer.models.decision_transformer import DecisionTransformer
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
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x


action_dim = 4
state_dim = 9

DECISION_TRANSFORMER = True

hidden_dim = 64
population_size = 100
epochs = 500
sigma = 0.3
lr = 0.01
number_of_threads = 2

def evaluator(model,DECISION_TRANSFORMER):
    result = []
    gen_samples = []

    sigma = 0.2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for param in model.parameters():
        sample = torch.normal(mean=0, std=torch.ones(
            param.shape, device=device))
        param.data = param.data + sigma * sample
        gen_samples.append(sample)

    # thread = Thread(target=evaluate_one_episode, args=(model, True,))
    result = evaluate_one_episode(model, None, None,  not DECISION_TRANSFORMER, 100, True)

    return result, gen_samples


def ES(num_iterations, population_size, sigma, learning_rate, number_of_threads, DECISION_TRANSFORMER):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # initialize a list of size population size of NNs with random weights
    population = []

    best_ratio = 1000000
    state_dim = 9
    act_dim = 4
    K = 24
    max_ep_len = 24
    variant = {
        'embed_dim': 128,
        'n_layer': 4,
        'n_head': 4,
        'activation_function': 'relu',
        'dropout': 0.1,
    }

    if DECISION_TRANSFORMER:
        base_model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        ).to(device)
    else:
        base_model = NN(state_dim, hidden_dim, action_dim).to(device)

    denominator = 0
    for i in range(population_size):
        denominator += math.log(population_size + 0.5) - math.log(i+1)

    zeros = []
    for param in base_model.parameters():
        zeros.append(torch.zeros(param.shape, device=device))

    for i in range(population_size):

        if DECISION_TRANSFORMER:
            model = DecisionTransformer(
                state_dim=state_dim,
                act_dim=act_dim,
                max_length=K,
                max_ep_len=max_ep_len,
                hidden_size=variant['embed_dim'],
                n_layer=variant['n_layer'],
                n_head=variant['n_head'],
                n_inner=4*variant['embed_dim'],
                activation_function=variant['activation_function'],
                n_positions=1024,
                resid_pdrop=variant['dropout'],
                attn_pdrop=variant['dropout'],
            ).to(device)
        else:            
            model = NN(state_dim, hidden_dim, action_dim).to(device)

        model.load_state_dict(base_model.state_dict())
        population.append(model)

    for iter in range(num_iterations):
        results = []
        samples = []

        for i in tqdm(range(population_size)):
            for p in population:
                p.load_state_dict(base_model.state_dict())
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=number_of_threads) as executor:
            pool = {executor.submit(
                evaluator, model,DECISION_TRANSFORMER): model for model in population}
            for thread in concurrent.futures.as_completed(pool):
                samples.append(thread.result()[1])
                results.append(thread.result()[0]['ratio'])

        # print(results)
        if best_ratio > np.min(results):
            best_ratio = np.min(results)
            print('saving best model')
            torch.save(base_model.state_dict(), 'best_model.pt')

        print(
            f'iter {iter} timer {(time.time()-start):.01f} mean {np.mean(results)}, Max: {np.max(results)}, Min: {np.min(results)} \t best: {best_ratio}')

        results = torch.tensor(results, device=device)

        ranking = torch.argsort(results, descending=True)
        # print(ranking)
        weights = [0]*population_size
        for i in range(population_size):
            weights[ranking[i]] = ((math.log(population_size + 0.5) -
                                    math.log(population_size - i)))  # / denominator)
        # print(weights)

        gradients = [z.detach().clone() for z in zeros]
        for i in range(population_size):
            for j in range(len(gradients)):
                gradients[j] += torch.mul(samples[i][j], weights[i])

        for j, param in enumerate(base_model.parameters()):
            param.data = param.data + learning_rate * gradients[j]


ES(num_iterations=epochs, population_size=population_size,
   sigma=sigma, learning_rate=lr, number_of_threads=number_of_threads, DECISION_TRANSFORMER=DECISION_TRANSFORMER)
