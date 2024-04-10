import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from src.simlator.Simulator import *
import matplotlib.pyplot as plt

from src.common.Parameters import *

class PPONet(nn.Module):
    def __init__(self, input_layer, output_layer):
        super(PPONet, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(input_layer, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_pi = nn.Linear(64, output_layer)
        self.fc_v = nn.Linear(64, 1)

    def pi(self, x, softmax_dim=0):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def pi_log_prob(self, x, softmax_dim=0):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc_pi(x)

        # Calculate softmax probabilities
        prob = F.softmax(x, dim=softmax_dim)

        # Calculate log probabilities
        log_prob = F.log_softmax(x, dim=softmax_dim)

        return prob, log_prob
    def v(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        v = self.fc_v(x)
        return v