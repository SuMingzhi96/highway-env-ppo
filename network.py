"""
	This file contains a neural network module for us to
	define our actor and critic networks in PPO.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class FeedForwardNN(nn.Module):
    """
		A standard in_dim-64-64-out_dim Feed Forward Neural Network.
	"""

    def __init__(self, in_dim, out_dim):
        """
			Initialize the network and set up the layers.

			Parameters:
				in_dim - input dimensions as an int
				out_dim - output dimensions as an int

			Return:
				None
		"""
        super(FeedForwardNN, self).__init__()
        self.log_std = nn.Parameter(torch.zeros(1, out_dim))  # We use 'nn.Paremeter' to train log_std automatically
        yinzi = 1
        self.layer1 = nn.Linear(in_dim, 64 * yinzi)
        self.layer2 = nn.Linear(64 * yinzi, 64 * yinzi)
        # self.layer3 = nn.Linear(64*yinzi, 64*yinzi)
        self.layer3 = nn.Linear(64 * yinzi, out_dim)

        orthogonal_init(self.layer1)  ##trick 8
        orthogonal_init(self.layer2)
        orthogonal_init(self.layer3, gain=0.01)

    def forward(self, obs):
        """
			Runs a forward pass on the neural network.

			Parameters:
				obs - observation to pass as input

			Return:
				output - the output of our forward pass
		"""
        # Convert observation to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        # activation3 = F.relu(self.layer3(activation2))
        output = F.tanh(self.layer3(activation2))

        return output

    def get_dist(self, s):
        mean = self.forward(s)
        log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = Normal(mean, std)  # Get the Gaussian distribution
        return dist
