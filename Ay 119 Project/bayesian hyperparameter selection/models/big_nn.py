#! /usr/bin/env python3

r"""
Network of Neural Networks (Big_NN)
"""

from __future__ import annotations
import torch
from typing import Any
from botorch.models.model import Model
from botorch.models import FixedNoiseGP
from botorch import fit_gpytorch_model
from botorch.posteriors import Posterior
from botorch.models.transforms import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor
from bofn.models.neural_network import NN
import numpy as np
import matplotlib.pyplot as plt


class Big_NN():
    r"""
    """

    def __init__(self, dag, architecture, active_input_indices,
                problem,
                trial,
                lr,
                train_iters,
                weight_decay,
                model_name="Big_NN") -> None:

        self.problem = problem
        self.trial = trial
        self.model_name = model_name
        self.train_iters = train_iters
        self.lr = lr
        self.weight_decay = weight_decay

        self.dag = dag
        self.n_nodes = dag.get_n_nodes()
        self.root_nodes = dag.get_root_nodes()
        self.active_input_indices = active_input_indices
        self.node_NNs = [None for k in range(self.n_nodes)]
        for k in range(len(self.root_nodes)):
            architecture[0] = len(active_input_indices[k])
            # make it known that this nn is part of big nn
            nn_model_name = model_name + "_root" + str(k)
            self.node_NNs[k] = NN(architecture=architecture, problem=problem, trial=trial, model_name=nn_model_name,
                                 train_iters=train_iters, lr=lr, weight_decay=weight_decay)

        for k in range(self.n_nodes):
            if self.node_NNs[k] is None:
                architecture[0] = len(self.dag.get_parent_nodes(k)) + len(active_input_indices[k])
                # make it known that this nn is part of big nn
                nn_model_name = model_name + "_root" + str(k)
                self.node_NNs[k] = NN(architecture=architecture, problem=problem, trial=trial, model_name=nn_model_name,
                                   train_iters=train_iters, lr=lr, weight_decay=weight_decay)

    def train_model(self, X, network_output_at_X):
        # for the root nodes, we will call forward, not forward_BigNN, because the problem statement already ensures [0,1] so
        # no need for sigmoid. Thus, we should train using the train_model method, not train_model_BigNN

        # actually no, because a root node could take part of its input from the problem statement and part from the
        # output of another node. So we should train using the train_model_BigNN method for all nodes
        for k in range(len(self.root_nodes)):
            target = network_output_at_X[:, k].unsqueeze(1)
            self.node_NNs[k].train_model(X=X[:, self.active_input_indices[k]], Y=target)
        for k in range(len(self.root_nodes), self.n_nodes):
            parent_nodes = self.dag.get_parent_nodes(k)
            concatenated = torch.cat([network_output_at_X[:, parent_nodes], X[:, self.active_input_indices[k]]], dim=1)
            target = network_output_at_X[:, k].unsqueeze(1)
            self.node_NNs[k].train_model(X=concatenated, Y=target)

    class CustomPosterior:
        def __init__(self, mean: Tensor, variance: Tensor):
            self.mean = mean
            self.variance = variance

    def posterior(self, X: Tensor) -> Tensor:
        with torch.no_grad():
            return self.CustomPosterior(
                mean=self.forward(X),
                variance=None,
            )

    def forward(self, X: Tensor) -> Tensor:
        if len(X.shape) == 3:
            assert X.shape[1] == 1
            X = X.squeeze(dim=1)

        # should every interior call to NNs be forward or embedding? i think still forward cuz embedding is just for training

        # for the root nodes, call forward, not forward_BigNN, because the problem statement already ensures [0,1] so
        # no need for sigmoid

        # actually no, because a root node could take part of its input from the problem statement and part from the
        # output of another node. So we should train using the train_model_BigNN method for all nodes
        network_output = torch.zeros(X.shape[0], self.n_nodes)
        for k in range(len(self.root_nodes)):
            network_output[:, k] = self.node_NNs[k].forward(X[:, self.active_input_indices[k]]).squeeze(dim=1)
        for k in range(len(self.root_nodes), self.n_nodes):
            parent_nodes = self.dag.get_parent_nodes(k)
            node_inputs = network_output[:, parent_nodes]
            active_inputs = X[:, self.active_input_indices[k]]
            input = torch.cat([node_inputs, active_inputs], dim=1)
            network_output[:, k] = self.node_NNs[k].forward(input).squeeze(dim=1)

        # messy way to get the dimensions right
        ret = network_output[:, -1].unsqueeze(1).unsqueeze(1)
        return ret