import numpy as np
import random
import torch

from typing import Optional, Tuple
from torch import Tensor

from gpytorch.distributions import MultivariateNormal
from botorch.posteriors.gpytorch import GPyTorchPosterior


class CustomPosterior:
    def __init__(self, mean: Tensor, variance: Tensor):
        self.mean = mean
        self.variance = variance


class EnsembleNN:
    def __init__(self, base_model_class, n_networks=10, subset_fraction=0.8, **model_kwargs):
        """
        Initialize the ensemble of neural networks.

        Parameters:
        - base_model_class: The class of the neural network to be used in the ensemble.
        - n_networks: Number of neural networks in the ensemble.
        - subset_fraction: Fraction of the training data to be used for each neural network.
        - model_kwargs: Additional arguments to initialize each neural network.
        """
        self.n_networks = n_networks
        self.subset_fraction = subset_fraction
        self.models = [base_model_class(**model_kwargs) for _ in range(n_networks)]

        for i, model in enumerate(self.models):
            model.model_name += f"_ensemble{i}"

        self.trained = [False] * n_networks
        self.num_outputs = 1

    def fit(self, X, y):
        """
        Train each neural network in the ensemble on a random subset of the training data.

        Parameters:
        - X: The training features, as a numpy array.
        - y: The training labels, as a numpy array.
        """
        n_samples = len(X)
        subset_size = int(self.subset_fraction * n_samples)

        for i, model in enumerate(self.models):
            # Randomly select a subset of the training data
            indices = random.sample(range(n_samples), subset_size)
            X_subset, y_subset = X[indices], y[indices]

            # Train the model on this subset
            model.train_model(X_subset, y_subset)
            self.trained[i] = True

    def forward(self, X: Tensor) -> Tensor:

        return self.posterior(X).mean

    def posterior(
            self,
            X: Tensor,
            observation_noise: bool = False,
            posterior_transform: Optional = None,
    ) -> GPyTorchPosterior:
        """
        Compute the posterior over model outputs at the provided points.

        Args:
            X: A tensor of shape `(batch_shape) x q x d`, where `d` is the dimension
                of the feature space and `q` is the number of points considered jointly.
            observation_noise: If True, add observation noise to the posterior.
            posterior_transform: An optional transformation to apply to the posterior.

        Returns:
            A GPyTorchPosterior object representing `batch_shape` joint distributions
            over `q` points. Includes observation noise if specified.
        """


        predictions = [model.forward(X) for model in self.models]  # shape: [n_networks, (batch_shape) x q x 1]

        # Step 2: Stack the predictions to get shape [n_networks, (batch_shape) x q x 1]
        predictions = torch.stack(predictions, dim=0)  # shape: [n_networks, (batch_shape) x q x 1]

        # Step 3: Compute the mean and variance of the predictions
        mean = predictions.mean(dim=0)  # shape: (batch_shape) x q x 1
        variance = predictions.var(dim=0)  # shape: (batch_shape) x q x 1

        # no normal
        posterior = CustomPosterior(mean=mean, variance=variance)

        # Apply any posterior transformations, if provided
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)

        return posterior

    def posterior_mean(
            self,
            X: Tensor):
        return self.posterior(X).mean