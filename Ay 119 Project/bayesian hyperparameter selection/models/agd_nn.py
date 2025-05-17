#!/usr/bin/env python3

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

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


from experiments.agd import AGD

class AGD_NN(torch.nn.Sequential):

    def __init__(
            self,
            architecture,
            problem,
            trial,
            train_iters,
            model_name="NN",
            bias=False
    ):
        super().__init__()

        self.architecture = architecture
        self.problem = problem
        self.trial = trial
        self.model_name = model_name
        self.train_iters = train_iters

        self.normalize = True
        self.standardize = True

        # hard code normalization to [0,1]
        # self.normalization_constant_lower = None
        # self.normalization_constant_upper = None
        self.standardization_constant_mean = None
        self.standardization_constant_std = None

        input_dim = architecture[0]
        output_dim = architecture[-1] # should be 1
        super(AGD_NN, self).__init__()

        self.initial = nn.Linear(input_dim, architecture[1], bias=bias)
        self.layers = nn.ModuleList(
            [nn.Linear(architecture[i], architecture[i + 1], bias=bias) for i in range(1, len(architecture) - 2)])
        self.final = nn.Linear(architecture[-2], output_dim, bias=bias)



    # for eval
    def forward(self, x: Tensor) -> Tensor:
        # print("forward input", x.shape)
        self.eval()

        # normalize input X
        # if self.normalize:
        #     x = ((x - self.normalization_constant_lower) / (self.normalization_constant_upper - self.normalization_constant_lower))

        # escape if x contains outside [0,1]
        if torch.any(x < 0) or torch.any(x > 1):
            print("Input x contains values outside [0,1]. - from forward")
            exit(1)


        # pass through network
        x = self.initial(x)
        x = F.relu(x) * math.sqrt(2)

        for layer in self.layers:
            x = layer(x)
            x = F.relu(x) * math.sqrt(2)

        x = self.final(x)


        # convert back to original mean and std
        if self.standardize:
            x = x * self.standardization_constant_std + self.standardization_constant_mean

        self.eval()
        # print("forward", x.shape)
        return x

    # for big_nn, we must use sigmoid since the inputs are no longer gauranteed to be in the range 0,1
    # def forward_BigNN(self, x: Tensor) -> Tensor:
    #     self.eval()
    #
    #     # normalize input X by passing it through sigmoid
    #     x = torch.sigmoid(x)
    #
    #     # escape if x contains outside [0,1]
    #     if torch.any(x < 0) or torch.any(x > 1):
    #         print("Input x contains values outside [0,1]. - from forward_BigNN")
    #         exit(1)
    #
    #     # pass through network
    #     for dim in range(len(self.architecture) - 1):
    #         x = self._modules["linear" + str(dim + 1)](x)
    #         if dim + 2 < len(self.architecture):
    #             x = self._modules[self.activation + str(dim + 1)](x)
    #
    #     # convert back to original mean and std
    #     if self.standardize:
    #         x = x * self.standardization_constant_std + self.standardization_constant_mean
    #
    #     self.eval()
    #     return x

    # for training (dont unstandardize)
    def embedding(self, x: Tensor) -> Tensor:
        self.train()

        # normalize input X
        # if self.normalize:
        #     x = ((x - self.normalization_constant_lower) / (self.normalization_constant_upper - self.normalization_constant_lower))

        # escape if x contains outside [0,1]
        if torch.any(x < 0) or torch.any(x > 1):
            print("Input x contains values outside [0,1]. - from embedding")
            exit(1)

        # pass through network
        x = x.view(x.shape[0], -1)

        x = self.initial(x)
        x = F.relu(x) * math.sqrt(2)

        for layer in self.layers:
            x = layer(x)
            x = F.relu(x) * math.sqrt(2)

        x = self.final(x)

        self.eval()
        # print("embedding", x.shape)
        return x

    # for big_nn, we must use sigmoid since the inputs are no longer gauranteed to be in the range 0,1
    # def embedding_BigNN(self, x: Tensor) -> Tensor:
    #     self.train()
    #
    #     # normalize input X by passing it through sigmoid
    #     x = torch.sigmoid(x)
    #
    #     # normalize input X
    #     # if self.normalize:
    #     #     x = ((x - self.normalization_constant_lower) / (self.normalization_constant_upper - self.normalization_constant_lower))
    #
    #     # escape if x contains outside [0,1]
    #     if torch.any(x < 0) or torch.any(x > 1):
    #         print("Input x contains values outside [0,1]. - from embedding_BigNN")
    #         exit(1)
    #
    #     # pass through network
    #     for dim in range(len(self.architecture) - 1):
    #         x = self._modules["linear" + str(dim + 1)](x)
    #         if dim + 2 < len(self.architecture):
    #             x = self._modules[self.activation + str(dim + 1)](x)
    #
    #     self.eval()
    #     return x

    def get_params(self):
        return [{"params": self.parameters()}]


    def train_model(self, X, Y, save_loss=True):
        self.train()
        num_iter = self.train_iters

        agd = AGD(self, gain=1.0) # may need to play around with gain
        agd.init_weights()

        # if self.normalize:
        #     # normalize input X and save normalization constants
        #     self.normalization_constant_lower = X.min(dim=0).values
        #     self.normalization_constant_upper = X.max(dim=0).values
        #     X = (X - self.normalization_constant_lower) / (self.normalization_constant_upper - self.normalization_constant_lower)

        # escape if x contains outside [0,1]
        if torch.any(X < 0) or torch.any(X > 1):
            print("Input x contains values outside [0,1]. - from train_model")
            exit(1)

        # standardize output Y
        if self.standardize:
            self.standardization_constant_mean = Y.mean()
            self.standardization_constant_std = Y.std()
            Y = (Y - self.standardization_constant_mean) / self.standardization_constant_std

        if save_loss:
            loss_path = "results/" + self.problem + "/" + self.model_name + "/losses/trial" + str(self.trial)
            os.makedirs(loss_path, exist_ok=True)
        preds_path = "results/" + self.problem + "/" + self.model_name + "/preds-in/trial" + str(self.trial)
        os.makedirs(preds_path, exist_ok=True)

        # optimizer = torch.optim.Adam(self.get_params(), lr=self.lr)
        # mse = torch.nn.MSELoss()
        losses = []
        # for i in range(num_iter):
        #     optimizer.zero_grad()
        #     preds = self.embedding(X)
        #     loss = mse(preds, Y)
        #
        #     # L2 regularization term
        #     l2_reg = 0.0
        #     for param in self.parameters():
        #         l2_reg += torch.norm(param) ** 2
        #     loss += 0.5 * self.weight_decay * l2_reg
        #
        #     losses.append(loss.item() / len(Y))
        #     loss.backward()
        #     optimizer.step()
        dataloader = [(X, Y)]
        for epoch in range(num_iter):
            optim = agd
            num_minibatches = len(dataloader)

            epoch_loss = 0
            epoch_log = 0

            for data, target in dataloader:
                output = self.embedding(data)
                loss = (output - target).square().mean()
                loss.backward()
                epoch_log += optim.step()
                self.zero_grad()
                epoch_loss += loss.item()

            loss = epoch_loss / num_minibatches
            losses.append(loss / len(Y))

        if save_loss:
            np.savetxt(loss_path + "/iter" + str(X.shape[0]) + ".txt", losses)

        preds = self.forward(X)
        np.savetxt(preds_path + "/iter" + str(X.shape[0]) + ".txt", preds.detach().numpy())

        self.eval()
        return self, None

    # for big_nn, we must normalize since the inputs are no longer gauranteed to be in the range 0,1
    # def train_model_BigNN(self, X, Y, save_loss=True):
    #     self.train()
    #     num_iter = self.train_iters
    #
    #     # normalize input X and save normalization constants
    #     # self.normalization_constant_lower = X.min(dim=0).values
    #     # self.normalization_constant_upper = X.max(dim=0).values
    #     # X = (X - self.normalization_constant_lower) / (self.normalization_constant_upper - self.normalization_constant_lower)
    #
    #     # unnecessary because embedding_BigNN will make sure to apply sigmoid to X
    #     # # escape if x contains outside [0,1]
    #     # if torch.any(X < 0) or torch.any(X > 1):
    #     #     print("Input x contains values outside [0,1]. - from train_model_BigNN")
    #     #     exit(1)
    #
    #     # standardize output Y
    #     if self.standardize:
    #         self.standardization_constant_mean = Y.mean()
    #         self.standardization_constant_std = Y.std()
    #         Y = (Y - self.standardization_constant_mean) / self.standardization_constant_std
    #
    #     if save_loss:
    #         loss_path = "results/" + self.problem + "/" + self.model_name + "/losses/trial" + str(self.trial)
    #         os.makedirs(loss_path, exist_ok=True)
    #     preds_path = "results/" + self.problem + "/" + self.model_name + "/preds-in/trial" + str(self.trial)
    #     os.makedirs(preds_path, exist_ok=True)
    #
    #     optimizer = torch.optim.Adam(self.get_params(), lr=self.lr)
    #     mse = torch.nn.MSELoss()
    #     losses = []
    #     for i in range(num_iter):
    #         optimizer.zero_grad()
    #         preds = self.embedding_BigNN(X)
    #         loss = mse(preds, Y)
    #
    #         # L2 regularization term
    #         l2_reg = 0.0
    #         for param in self.parameters():
    #             l2_reg += torch.norm(param) ** 2
    #         loss += 0.5 * self.weight_decay * l2_reg
    #
    #         losses.append(loss.item() / len(Y))
    #         loss.backward()
    #         optimizer.step()
    #
    #     if save_loss:
    #         np.savetxt(loss_path + "/iter" + str(X.shape[0]) + ".txt", losses)
    #
    #     preds = self.forward_BigNN(X)
    #     np.savetxt(preds_path + "/iter" + str(X.shape[0]) + ".txt", preds.detach().numpy())
    #
    #     self.eval()
    #     return self, None

    # reset model for cross val
    # call embedding without fuckign up the gradients
    def cross_val_preds(self, num_folds, X, Y):
        # should not need training afterwards, because it is getting retrained every fold
        self.eval()
        # for every fold, train on the rest and predict on the fold
        preds = []
        for i in range(num_folds):
            lower = math.floor(i * X.shape[0] / num_folds)
            upper = math.floor((i + 1) * X.shape[0] / num_folds)
            # train on the rest
            X_train = torch.cat((X[:lower], X[upper:]))
            Y_train = torch.cat((Y[:lower], Y[upper:]))
            self.train_model(X=X_train, Y=Y_train, save_loss=False)
            # predict on the fold
            preds_fold = self.forward(X[lower:upper])
            preds.append(preds_fold)

        preds_path = "results/" + self.problem + "/" + self.model_name + "/preds-fold" + str(
            num_folds) + "/trial" + str(self.trial)
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


