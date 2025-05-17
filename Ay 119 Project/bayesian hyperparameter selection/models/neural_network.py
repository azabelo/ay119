#!/usr/bin/env python3

from __future__ import annotations

import torch

from sklearn.model_selection import train_test_split
from torch import Tensor

import matplotlib.pyplot as plt
import numpy as np
import os
import math
import sys

# Get script directory
# script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
# sys.path.append(script_dir[:-12])

from experiments.ackley import Ackley
from experiments.dropwave import Dropwave
from experiments.alpine2 import Alpine2
from experiments.rosenbrock import Rosenbrock

class NN(torch.nn.Sequential):
    """
    Architecture is
    [
        input,
        architecture[1] + activation + dropout,
        ...
        architecture[-2] + activation + dropout,
        architecture[-1]
    ]
    """
    act_dict = {
        "relu": torch.nn.ReLU,
        "leaky_relu": torch.nn.LeakyReLU,
        "swish": torch.nn.SiLU,
        "sigmoid": torch.nn.Sigmoid,
        "tanh": torch.nn.Tanh,
        "softmax": torch.nn.Softmax,
    }

    def __init__(
        self,
        architecture,
        problem,
        trial,
        lr,
        train_iters,
        weight_decay,
        model_name="NN",
        activation="leaky_relu",
        # normalize=False,
        # standardize=False
    ):
        super().__init__()

        self.architecture = architecture
        act_layer = self.act_dict[activation.lower()]
        self.activation = activation
        self.problem = problem
        self.trial = trial
        self.model_name = model_name
        self.train_iters = train_iters
        self.lr = lr
        self.weight_decay = weight_decay

        self.efficient = True if model_name[-5:] == "_fast" else False

        self.rpn_multiplier = 0

        # self.normalize = normalize
        # self.standardize = standardize
        #
        # # not always used
        # self.normalization_constant_lower = None
        # self.normalization_constant_upper = None
        # self.standardization_constant_mean = None
        # self.standardization_constant_std = None

        for dim in range(len(architecture) - 1):
            name = str(dim + 1)
            self.add_module(
                "linear" + name,
                torch.nn.Linear(architecture[dim], architecture[dim + 1]).double(),
            )
            # don't dropout/activate from output layer
            if dim + 2 < len(architecture):
                self.add_module(activation + name, act_layer())

            # RPN
            linear = torch.nn.Linear(architecture[dim], architecture[dim + 1]).double()
            linear.weight.requires_grad = False
            linear.bias.requires_grad = False
            self.add_module(
                "linear" + name + "_rpn",
               linear,
            )
            if dim + 2 < len(architecture):
                self.add_module(activation + name + "_rpn", act_layer())

    class CustomPosterior:
        def __init__(self, mean: Tensor, variance: Tensor):
            self.mean = mean
            self.variance = variance

    def posterior(self, X: Tensor) -> Tensor:
        self.eval()
        with torch.no_grad():
            return self.CustomPosterior(
                mean=self(X),
                variance=None,
            )

    # for eval
    def forward(self, x: Tensor, train=False) -> Tensor:
        # print("forward input", x.shape)
        self.eval()
        if train:
            self.train()

        X = x.clone()

        # pass through network
        for dim in range(len(self.architecture) - 1):
            X = self._modules["linear"+str(dim + 1)](X)
            if dim + 2 < len(self.architecture):
                X = self._modules[self.activation+str(dim + 1)](X)

        rpn = x.clone()

        # pass through rpn
        for dim in range(len(self.architecture) - 1):
            rpn = self._modules["linear" + str(dim + 1) + "_rpn"](rpn)
            if dim + 2 < len(self.architecture):
                rpn = self._modules[self.activation + str(dim + 1) + "_rpn"](rpn)

        X = X + rpn * self.rpn_multiplier
        # print(rpn)

        self.eval()
        return X
    
    def get_params(self):
        return [{"params": self.parameters()}]


    def train_model(self, X, Y):

        indices = torch.randperm(X.size(0))
        X = X[indices]
        Y = Y[indices]

        self.train()
        num_iter = self.train_iters

        if not self.efficient:
            loss_path = "results/" + self.problem + "/" + self.model_name + "/trial" + str(self.trial) + "/losses"
            os.makedirs(loss_path, exist_ok=True)

            preds_path = "results/" + self.problem + "/" + self.model_name + "/trial" + str(self.trial) + "/predictions"
            os.makedirs(preds_path, exist_ok=True)

        optimizer = torch.optim.Adam(self.get_params(), lr=self.lr)
        mse = torch.nn.MSELoss()
        losses = []

        # # batch descent
        # for i in range(num_iter):
        #     optimizer.zero_grad()
        #     preds = self.forward(X, train=True)
        #     loss = mse(preds, Y)
        #
        #     # L2 regularization term:
        #     l2_reg = 0.0
        #     for param in self.parameters():
        #         l2_reg += torch.norm(param) ** 2
        #     loss += 0.5 * self.weight_decay * l2_reg
        #
        #     losses.append(loss.item())
        #     loss.backward()
        #     optimizer.step()_iter):
        #     optimizer.zero_grad()
        #     preds = self.forward(X, train=True)
        #     loss = mse(preds, Y)
        #
        #     # L2 regularization term:
        #     l2_reg = 0.0
        #     for param in self.parameters():
        #         l2_reg += torch.norm(param) ** 2
        #     loss += 0.5 * self.weight_decay * l2_reg
        #
        #     losses.append(loss.item())
        #     loss.backward()
        #     optimizer.step()

        # mini-batch descent
        batch_size = 4
        for epoch in range(num_iter):
            # Shuffle at the start of each epoch
            indices = torch.randperm(X.size(0))
            X = X[indices]
            Y = Y[indices]

            for i in range(0, X.size(0), batch_size):
                X_batch = X[i:i + batch_size]
                Y_batch = Y[i:i + batch_size]

                optimizer.zero_grad()
                preds = self.forward(X_batch, train=True)
                loss = mse(preds, Y_batch)

                # L2 regularization term
                l2_reg = sum(torch.norm(param) ** 2 for param in self.parameters())
                loss += 0.5 * self.weight_decay * l2_reg

                losses.append(loss.item())
                loss.backward()
                optimizer.step()

        if not self.efficient:
            np.savetxt(loss_path + "/iter" + str(X.shape[0]) + ".txt", losses)

            preds = self.forward(X)
            np.savetxt(preds_path + "/iter" + str(X.shape[0]) + ".txt", preds.detach().numpy())

        self.eval()
        return self, None




# reset model for cross val
# call embedding without fuckign up the gradients
    def cross_val_preds(self, num_folds, X, Y):
        # should not need training afterwards, because it is getting retrained every fold
        self.eval()
        # for every fold, train on the rest and predict on the fold
        preds = []
        for i in range(num_folds):
            lower = math.floor(i * X.shape[0] / num_folds)
            upper = math.floor((i+1) * X.shape[0] / num_folds)
            # train on the rest
            X_train = torch.cat((X[:lower], X[upper:]))
            Y_train = torch.cat((Y[:lower], Y[upper:]))
            self.train_model(X=X_train, Y=Y_train, save_loss=False)
            # predict on the fold
            preds_fold = self.forward(X[lower:upper])
            preds.append(preds_fold)

        preds_path = "results/" + self.problem + "/" + self.model_name + "/preds-fold" + str(num_folds) + "/trial" + str(self.trial)
        os.makedirs(preds_path, exist_ok=True)

        preds = torch.cat(preds)
        assert preds.shape == Y.shape
        np.savetxt(preds_path + "/iter" + str(X.shape[0]) + ".txt", preds.detach().numpy())


    def random_ball_preds(self, d, X):
        # should already be trained
        self.eval()
        # for every point in X, generate random points with distance d from the point
        pertubations = torch.rand_like(X) * 2 - 1
        # Normalize the random vectors to have length d (not uniform on a sphere, but close enough)
        pertubations = pertubations * (d / torch.norm(pertubations, dim=-1, keepdim=True))
        perturbed_X = X + pertubations
        # dont let the pertubations go outside [0,1]
        perturbed_X = torch.clamp(perturbed_X, 0, 1)

        perturbed_preds = self.forward(perturbed_X)
        perturbed_targets = self.get_ground_truth(perturbed_X, self.problem)

        preds_path = "results/" + self.problem + "/" + self.model_name + "/preds-ball/trial" + str(self.trial)
        os.makedirs(preds_path, exist_ok=True)
        targets_path = "results/" + self.problem + "/" + self.model_name + "/targets-ball/trial" + str(self.trial)
        os.makedirs(targets_path, exist_ok=True)

        np.savetxt(preds_path + "/iter" + str(X.shape[0]) + ".txt", perturbed_preds.detach().numpy())
        np.savetxt(targets_path + "/iter" + str(X.shape[0]) + ".txt", perturbed_targets.detach().numpy())



    def near_preds(self, d, X):
        # should already be trained
        self.eval()
        # almost like random ball preds, but now the target is the output at X, not at the new points
        pertubations = torch.rand_like(X) * 2 - 1
        # Normalize the random vectors to have length d (not uniform on a sphere, but close enough)
        pertubations = pertubations * (d / torch.norm(pertubations, dim=-1, keepdim=True))
        perturbed_X = X + pertubations
        # dont let the pertubations go outside [0,1]
        perturbed_X = torch.clamp(perturbed_X, 0, 1)

        perturbed_preds = self.forward(perturbed_X)
        # here is the difference (compare these targets with objective_at_X to ensure it is correct)
        perturbed_targets = self.get_ground_truth(X, self.problem)

        preds_path = "results/" + self.problem + "/" + self.model_name + "/preds-near/trial" + str(self.trial)
        os.makedirs(preds_path, exist_ok=True)
        targets_path = "results/" + self.problem + "/" + self.model_name + "/targets-near/trial" + str(self.trial)
        os.makedirs(targets_path, exist_ok=True)

        np.savetxt(preds_path + "/iter" + str(X.shape[0]) + ".txt", perturbed_preds.detach().numpy())
        np.savetxt(targets_path + "/iter" + str(X.shape[0]) + ".txt", perturbed_targets.detach().numpy())

    def random_preds(self, X):
        # should already be trained
        self.eval()

        rand = torch.rand(X.size())
        preds = self.forward(rand)
        targets = self.get_ground_truth(rand, self.problem)

        preds_path = "results/" + self.problem + "/" + self.model_name + "/preds-rand/trial" + str(self.trial)
        os.makedirs(preds_path, exist_ok=True)
        targets_path = "results/" + self.problem + "/" + self.model_name + "/targets-rand/trial" + str(self.trial)
        os.makedirs(targets_path, exist_ok=True)

        np.savetxt(preds_path + "/iter" + str(X.shape[0]) + ".txt", preds.detach().numpy())
        np.savetxt(targets_path + "/iter" + str(X.shape[0]) + ".txt", targets.detach().numpy())


    def get_ground_truth(self, X, problem):
        if problem == "rosenbrock":
            input_dim = 4
            rosenbrock = Rosenbrock(dim=input_dim)
            return rosenbrock.evaluate(X=X)[..., -1]
        elif problem == "ackley":
            ackley = Ackley(input_dim=6)
            return ackley.evaluate(X=X)[..., -1]
        elif problem == "alpine2":
            alpine2 = Alpine2(n_nodes=6)
            return alpine2.evaluate(X=X)[..., -1]
        elif problem == "dropwave":
            dropwave = Dropwave()
            return dropwave.evaluate(X=X)[..., -1]


