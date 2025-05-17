import torch
from torch import nn
from typing import List
from torch import Tensor
import os
import numpy as np

from bofn.models.neural_network import NN

class Joint_Big_NN():
    """
    Joint training of neural networks forming a directed acyclic graph (DAG).
    """

    def __init__(self, dag, architecture, active_input_indices,
                 problem, trial, lr, train_iters, weight_decay, final_node_loss_weight, model_name="JointBig_NN") -> None:
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
        self.final_node_loss_weight = final_node_loss_weight
        self.node_NNs = nn.ModuleList(
            [None for _ in range(self.n_nodes)])  # Use nn.ModuleList for PyTorch compatibility

        # Initialize the NN for each node
        for k in range(len(self.root_nodes)):
            architecture[0] = len(active_input_indices[k])
            nn_model_name = f"{model_name}_root{k}"
            self.node_NNs[k] = NN(architecture=architecture, problem=problem, trial=trial,
                                  model_name=nn_model_name, train_iters=train_iters, lr=lr, weight_decay=weight_decay)

        for k in range(self.n_nodes):
            if self.node_NNs[k] is None:
                architecture[0] = len(self.dag.get_parent_nodes(k)) + len(active_input_indices[k])
                nn_model_name = f"{model_name}_root{k}"
                self.node_NNs[k] = NN(architecture=architecture, problem=problem, trial=trial,
                                      model_name=nn_model_name, train_iters=train_iters, lr=lr,
                                      weight_decay=weight_decay)

    def train_model(self, X: Tensor, network_output_at_X: Tensor):
        for node in self.node_NNs:
            node.train()

        """
        Jointly train the entire network of neural networks using the sum of the node-specific losses.

        Args:
            X (Tensor): Input data of shape (n_samples, n_features).
            network_output_at_X (Tensor): Target outputs for each node, shape (n_samples, n_nodes).
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()  # Using mean squared error loss for each node's output

        loss_path = "results/" + self.problem + "/" + self.model_name + "/trial" + str(self.trial) + "/losses"
        os.makedirs(loss_path, exist_ok=True)

        preds_path = "results/" + self.problem + "/" + self.model_name + "/trial" + str(self.trial) + "/predictions"
        os.makedirs(preds_path, exist_ok=True)

        losses = []
        for epoch in range(self.train_iters):
            optimizer.zero_grad()  # Zero the gradients for every epoch
            total_loss = 0.0

            # Compute the forward pass through the entire network to get outputs
            network_output = torch.zeros(X.shape[0], self.n_nodes)
            for k in range(len(self.root_nodes)):
                # Forward pass through each root node
                network_output[:, k] = self.node_NNs[k].forward(X[:, self.active_input_indices[k]], train=True).squeeze(dim=1)
            for k in range(len(self.root_nodes), self.n_nodes):
                # For each non-root node, concatenate inputs from parent nodes and active inputs
                parent_nodes = self.dag.get_parent_nodes(k)
                node_inputs = network_output[:, parent_nodes]
                active_inputs = X[:, self.active_input_indices[k]]
                node_input = torch.cat([node_inputs, active_inputs], dim=1)
                network_output[:, k] = self.node_NNs[k].forward(node_input, train=True).squeeze(dim=1)

            # Compute loss for each node and accumulate
            for k in range(self.n_nodes):
                target = network_output_at_X[:, k].unsqueeze(1)  # Get target values for the node
                node_output = network_output[:, k].unsqueeze(1)  # Output from the node

                # Apply the final node's loss weight
                if k == self.n_nodes - 1:
                    loss = self.final_node_loss_weight * criterion(node_output, target)
                else:
                    loss = criterion(node_output, target)

                total_loss += loss  # Accumulate loss across all nodes

            losses.append(total_loss.item())
            np.savetxt(loss_path + "/iter" + str(X.shape[0]) + ".txt", losses)

            preds = self.forward(X).squeeze(1)
            np.savetxt(preds_path + "/iter" + str(X.shape[0]) + ".txt", preds.detach().numpy())

            # Backpropagation and optimization step for the entire network
            total_loss.backward()
            optimizer.step()

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
        """
        Forward pass through the entire network, computing outputs at each node.

        Args:
            X (Tensor): Input data of shape (n_samples, n_features).

        Returns:
            Tensor: Final output at the last node, with shape (n_samples, 1, 1).
        """
        for node in self.node_NNs:
            node.eval()

        if len(X.shape) == 3:
            assert X.shape[1] == 1
            X = X.squeeze(dim=1)

        # Forward pass through each node of the network, considering the DAG structure
        network_output = torch.zeros(X.shape[0], self.n_nodes)
        for k in range(len(self.root_nodes)):
            network_output[:, k] = self.node_NNs[k].forward(X[:, self.active_input_indices[k]]).squeeze(dim=1)
        for k in range(len(self.root_nodes), self.n_nodes):
            parent_nodes = self.dag.get_parent_nodes(k)
            node_inputs = network_output[:, parent_nodes]
            active_inputs = X[:, self.active_input_indices[k]]
            node_input = torch.cat([node_inputs, active_inputs], dim=1)
            network_output[:, k] = self.node_NNs[k].forward(node_input).squeeze(dim=1)

        # Return the output of the final node as the network's prediction
        return network_output[:, -1].unsqueeze(1).unsqueeze(1)

    def parameters(self):
        """
        Return the parameters of all node NNs for joint optimization.
        """
        params = []
        for nn in self.node_NNs:
            params.extend(nn.parameters())
        return params
