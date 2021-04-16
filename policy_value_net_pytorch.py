"""
An implementation of the policyValueNet in PyTorch
Tested in PyTorch 0.2.0 and 0.3.0
@author: Junxiao Song
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Net(nn.Module):
    """policy-value network module"""
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        # TODO: create common layers

        # TODO: create action policy layers

        # TODO: create state value layers


    def forward(self, state_input):
        x_act = None
        x_val =None
        # TODO: common layers

        # TODO: action policy layers

        # TODO: state value layers

        return x_act, x_val


class PolicyValueNet():
    """policy-value network """
    def __init__(self, board_width, board_height,
                 model_file=None, use_gpu=False):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty
        # the policy value net module
        if self.use_gpu:
            self.policy_value_net = Net(board_width, board_height).cuda()
        else:
            self.policy_value_net = Net(board_width, board_height)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        """ TODO: policy function with batch input
        input: a batch of states
        output: a batch of action probabilities and state values
        """

    def policy_value_fn(self, board):
        """ TODO: policy function with board state input
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """TODO: perform a training step"""

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)