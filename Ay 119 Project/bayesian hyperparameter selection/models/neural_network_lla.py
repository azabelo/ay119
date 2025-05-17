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

from laplace import Laplace


class NN_LLA(torch.nn.Sequential):
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

        self.normalize = True
        self.standardize = True

        # hard code normalization to [0,1]
        # self.normalization_constant_lower = None
        # self.normalization_constant_upper = None
        self.standardization_constant_mean = None
        self.standardization_constant_std = None

        for dim in range(len(architecture) - 1):
            name = str(dim + 1)
            self.add_module(
                "linear" + name,
                torch.nn.Linear(architecture[dim], architecture[dim + 1]).double(),
            )
            # don't dropout/activate from output layer
            if dim + 2 < len(architecture):
                self.add_module(activation + name, act_layer())

    # for eval
    def forward(self, x: Tensor) -> Tensor:
        self.eval()

        # normalize input X
        # if self.normalize:
        #     x = ((x - self.normalization_constant_lower) / (self.normalization_constant_upper - self.normalization_constant_lower))

        print(self.model_name)
        # escape if x contains outside [0,1]
        # if torch.any(x < 0) or torch.any(x > 1):
        #     print("Input x contains values outside [0,1]. - from forward")
        #     exit(1)

        # pass through network
        for dim in range(len(self.architecture) - 1):
            x = self._modules["linear" + str(dim + 1)](x)
            if dim + 2 < len(self.architecture):
                x = self._modules[self.activation + str(dim + 1)](x)

        # convert back to original mean and std
        if self.standardize:
            x = x * self.standardization_constant_std + self.standardization_constant_mean

        self.eval()
        return x

    # for big_nn, we must use sigmoid since the inputs are no longer gauranteed to be in the range 0,1

    # for training (dont unstandardize)
    def embedding(self, x: Tensor) -> Tensor:
        self.train()

        # normalize input X
        # if self.normalize:
        #     x = ((x - self.normalization_constant_lower) / (self.normalization_constant_upper - self.normalization_constant_lower))

        # escape if x contains outside [0,1]
        # if torch.any(x < 0) or torch.any(x > 1):
        #     print("Input x contains values outside [0,1]. - from embedding")
        #     exit(1)

        # pass through network
        for dim in range(len(self.architecture) - 1):
            x = self._modules["linear" + str(dim + 1)](x)
            if dim + 2 < len(self.architecture):
                x = self._modules[self.activation + str(dim + 1)](x)

        self.eval()
        return x

    def get_params(self):
        return [{"params": self.parameters()}]

    def train_model(self, X, Y, save_loss=True):
        self.train()
        num_iter = self.train_iters

        # if self.normalize:
        #     # normalize input X and save normalization constants
        #     self.normalization_constant_lower = X.min(dim=0).values
        #     self.normalization_constant_upper = X.max(dim=0).values
        #     X = (X - self.normalization_constant_lower) / (self.normalization_constant_upper - self.normalization_constant_lower)

        # escape if x contains outside [0,1]
        # if torch.any(X < 0) or torch.any(X > 1):
        #     print("Input x contains values outside [0,1]. - from train_model")
        #     exit(1)

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

        optimizer = torch.optim.Adam(self.get_params(), lr=self.lr)
        mse = torch.nn.MSELoss()
        losses = []
        for i in range(num_iter):
            optimizer.zero_grad()
            preds = self.embedding(X)
            loss = mse(preds, Y)

            # L2 regularization term
            l2_reg = 0.0
            for param in self.parameters():
                l2_reg += torch.norm(param) ** 2
            loss += 0.5 * self.weight_decay * l2_reg

            losses.append(loss.item() / len(Y))
            loss.backward()
            optimizer.step()

        if save_loss:
            np.savetxt(loss_path + "/iter" + str(X.shape[0]) + ".txt", losses)

        preds = self.forward(X)
        np.savetxt(preds_path + "/iter" + str(X.shape[0]) + ".txt", preds.detach().numpy())

        self.eval()

        # Compute Laplace approximation
        la = Laplace(self, likelihood='regression', subset_of_weights='all')

        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(X, Y)
        train_loader = DataLoader(dataset, batch_size=len(X), shuffle=True)

        la.fit(train_loader)
        la.optimize_prior_precision()

        return self, la


def predict_with_uncertainty(model, la, X_test):
    f_mu, f_var = la(X_test, pred_type='nn')
    f_std = torch.sqrt(f_var)
    return f_mu, f_std

if __name__ == "__main__":
    input_size = 2
    hidden_size = 10
    output_size = 1

    architecture = [input_size, hidden_size, output_size]
    problem = "ackley"
    trial = 1
    lr = 0.001
    train_iters = 1000
    weight_decay = 1e-4

    model = NN_LLA(architecture, problem, trial, lr, train_iters, weight_decay)

    # Create synthetic data for example
    X = torch.randn(100, input_size).double()
    Y = torch.randn(100, output_size).double()
    print(X)
    print(Y)

    trained_model, laplace_approximation = model.train_model(X, Y)

    print(type(laplace_approximation))

    # Example usage
    mean, std = predict_with_uncertainty(trained_model, laplace_approximation, X)
    print("Posterior Mean:", mean)
    print("Posterior Std Dev:", std)