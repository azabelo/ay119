import numpy as np
import os
import sys
import time
import torch
from botorch.acquisition import ExpectedImprovement, qExpectedImprovement, qKnowledgeGradient, LogExpectedImprovement
from botorch.acquisition import PosteriorMean as GPPosteriorMean
from botorch.sampling.normal import SobolQMCNormalSampler
from torch import Tensor
from typing import Callable, List, Optional

from bofn.acquisition_function_optimization.optimize_acqf import optimize_acqf_and_get_suggested_point
from bofn.utils.dag import DAG
from bofn.utils.initial_design import generate_initial_design
from bofn.utils.fit_gp_model import fit_gp_model
from bofn.models.gp_network import GaussianProcessNetwork
from bofn.models.neural_network import NN
from bofn.models.big_nn import Big_NN
from bofn.utils.posterior_mean import PosteriorMean, PosteriorVar
from botorch.models.deterministic import GenericDeterministicModel

from bofn.models.agd_nn import AGD_NN
#from bofn.utils.laplace_bayesopt.laplace_bayesopt.botorch import LaplaceBoTorch

from bofn.models.ensemble_nn import EnsembleNN

# from bofn.models.RPN import rpn_bo_acquisitions, rpn_bo_optimizers, rpn_bo_models, rpn_bo_utilities, rpn_bo_architectures, rpn_bo_dataloaders
from bofn.models.RPN.rpn_bo_dataloaders import BootstrapLoader
from bofn.models.RPN.rpn_bo_models import EnsembleRegression
from bofn.models.RPN.rpn_bo_acquisitions import MCAcquisition

from jax import vmap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from bofn.models.joint_big_nn import Joint_Big_NN

def bofn_trial(
    problem: str,
    algo: str,
    trial:int,
    n_init_evals: int,
    n_bo_iter: int,
    restart: bool,
    function_network: Callable,
    dag: DAG,
    active_input_indices: List[List[Optional[int]]],
    input_dim: int,
    network_to_objective_transform: Callable,
) -> None:
    # Get script directory
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    results_folder = script_dir + "/results/" + problem + "/" + algo + "/trial" + str(trial) + "/"

    efficient = True if algo[-5:] == "_fast" else False

    if restart and False:
        # Check if training data is already available
        try:
            # Current available evaluations
            X = torch.tensor(np.loadtxt(results_folder +
                                        "X/X_" + str(trial) + ".txt"))
            network_output_at_X = torch.tensor(np.loadtxt(
                results_folder + "network_output_at_X/network_output_at_X_" + str(trial) + ".txt"))
            objective_at_X = torch.tensor(np.loadtxt(
                results_folder + "objective_at_X/objective_at_X_" + str(trial) + ".txt"))

            # Historical best observed objective values and running times
            hist_best_obs_vals = list(np.loadtxt(
                results_folder + "best_obs_vals_" + str(trial) + ".txt"))
            runtimes = list(np.loadtxt(
                results_folder + "runtimes/runtimes_" + str(trial) + ".txt"))

            # Current best observed objective value
            best_obs_val = torch.tensor(hist_best_obs_vals[-1])

            init_batch_id = len(hist_best_obs_vals)
            print("Restarting experiment from available data.")

        except:

            # Initial evaluations
            X = generate_initial_design(
                num_samples=n_init_evals, input_dim=input_dim, seed=trial)
            network_output_at_X = function_network(X)
            objective_at_X = network_to_objective_transform(
                network_output_at_X)

            # Current best objective value
            best_obs_val = objective_at_X.max().item()

            # Historical best observed objective values and running times
            hist_best_obs_vals = [best_obs_val]
            runtimes = []

            init_batch_id = 1
    else:
        # Initial evaluations (random)
        X = generate_initial_design(
            num_samples=n_init_evals, input_dim=input_dim, seed=trial)
        network_output_at_X = function_network(X)
        objective_at_X = network_to_objective_transform(network_output_at_X)

        # Current best objective value
        best_obs_val = objective_at_X.max().item()

        # Historical best observed objective values and running times
        hist_best_obs_vals = [best_obs_val]
        runtimes = []

        init_batch_id = 1

    for iteration in range(init_batch_id, n_bo_iter + 1):
        print("Experiment: " + problem)
        print("Sampling policy: " + algo)
        print("Trial: " + str(trial))
        print("Iteration: " + str(iteration))

        # New suggested point
        t0 = time.time()
        new_x = get_new_suggested_point(
            algo=algo,
            problem=problem,
            trial=trial,
            X=X,
            network_output_at_X=network_output_at_X,
            objective_at_X=objective_at_X,
            network_to_objective_transform=network_to_objective_transform,
            dag=dag,
            active_input_indices=active_input_indices,
            function_network=function_network,
            results_folder=results_folder
        )
        t1 = time.time()
        runtimes.append(t1 - t0)

        # Evalaute network at new point
        network_output_at_new_x = function_network(new_x)

        # Evaluate objective at new point
        objective_at_new_x = network_to_objective_transform(
            network_output_at_new_x)

        # Update training data
        X = torch.cat([X, new_x], 0)
        network_output_at_X = torch.cat(
            [network_output_at_X, network_output_at_new_x], 0)
        objective_at_X = torch.cat([objective_at_X, objective_at_new_x], 0)

        # Update historical best observed objective values
        best_obs_val = objective_at_X.max().item()
        hist_best_obs_vals.append(best_obs_val)
        print("Best value found so far: " + str(best_obs_val))

        # Save data
        # np.savetxt(results_folder + "X/X_" + str(trial) + ".txt", X.numpy())
        # np.savetxt(results_folder + "network_output_at_X/network_output_at_X_" +
        #            str(trial) + ".txt", network_output_at_X.numpy())
        # np.savetxt(results_folder + "objective_at_X/objective_at_X_" +
        #            str(trial) + ".txt", objective_at_X.numpy())
        # np.savetxt(results_folder + "best_obs_vals_" +
        #            str(trial) + ".txt", np.atleast_1d(hist_best_obs_vals))
        # np.savetxt(results_folder + "runtimes/runtimes_" +
        #            str(trial) + ".txt", np.atleast_1d(runtimes))

        os.makedirs(results_folder, exist_ok=True)

        np.savetxt(results_folder + "X.txt", X.numpy())
        np.savetxt(results_folder + "network_output_at_X.txt", network_output_at_X.numpy())
        np.savetxt(results_folder + "objective_at_X.txt", objective_at_X.numpy())
        np.savetxt(results_folder + "best_obs_vals.txt", np.atleast_1d(hist_best_obs_vals))
        np.savetxt(results_folder + "runtimes.txt", np.atleast_1d(runtimes))

    dir = 'results/' + problem + '/' + algo + '/trial' + str(trial) + '/'

    if "NN" in algo and not efficient:
        plot_loss(dir, trial)

def get_new_suggested_point(
    algo: str,
    problem: str,
    trial,
    X: Tensor,
    network_output_at_X: Tensor,
    objective_at_X: Tensor,
    network_to_objective_transform: Callable,
    dag: DAG,
    active_input_indices: List[int],
    function_network,
    results_folder
) -> Tensor:
    input_dim = X.shape[-1]

    lb = np.array([0.0 for _ in range(input_dim)])
    ub = np.array([1.0 for _ in range(input_dim)])
    bounds = (lb, ub)
    
    standardized_network_output_at_X =  (network_output_at_X - network_output_at_X.mean(dim=0))/network_output_at_X.std(dim=0)
    standardized_objective_at_X = (objective_at_X - objective_at_X.mean())/objective_at_X.std()

    sub_models = None

    efficient = True if algo[-5:] == "_fast" else False

    if algo == "Random":
        return torch.rand([1, input_dim])

    elif algo[:7] == "GPFN_PM":
        model = GaussianProcessNetwork(train_X=X, train_Y=network_output_at_X, dag=dag,
                                       active_input_indices=active_input_indices)
        qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        # could also do this using a property decorator i think
        qmc_sampler.batch_range_override = (0, 1)
        acquisition_function = PosteriorMean(
            model=model,
            sampler=qmc_sampler,
            objective=network_to_objective_transform,
        )
        posterior_mean_function = acquisition_function

    elif algo == "SingleGPFN_PM":
        model = GaussianProcessNetwork(train_X=X, train_Y=network_output_at_X, dag=dag,
                                       active_input_indices=active_input_indices, single_task=True)
        qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        # could also do this using a property decorator i think
        qmc_sampler.batch_range_override = (0, 1)
        acquisition_function = PosteriorMean(
            model=model,
            sampler=qmc_sampler,
            objective=network_to_objective_transform,
        )
        posterior_mean_function = acquisition_function

    elif algo[:5] == "GP_PM":
        model = fit_gp_model(X=X, Y=objective_at_X)
        def temp_function(x):
            return model.posterior(x).mean
        posterior_sample = GenericDeterministicModel(f=temp_function)
        acquisition_function = GPPosteriorMean(model=posterior_sample)
        posterior_mean_function = GPPosteriorMean(model=posterior_sample)

    elif algo == "SingleGP_PM":
        model = fit_gp_model(X=X, Y=objective_at_X, single_task=True)
        def temp_function(x):
            return model.posterior(x).mean
        posterior_sample = GenericDeterministicModel(f=temp_function)
        acquisition_function = GPPosteriorMean(model=posterior_sample)
        posterior_mean_function = GPPosteriorMean(model=posterior_sample)



    elif algo[:7] == "GPFN_EI":
        # Model
        model = GaussianProcessNetwork(train_X=X, train_Y=network_output_at_X, dag=dag,
                          active_input_indices=active_input_indices)
        # Sampler
        # qmc_sampler = SobolQMCNormalSampler(num_samples=128) #for older botorch versions
        qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        # could also do this using a property decorator i think
        qmc_sampler.batch_range_override = (0, 1)

        # Acquisition function
        acquisition_function = qExpectedImprovement(
            model=model,
            best_f=objective_at_X.max().item(),
            sampler=qmc_sampler,
            objective=network_to_objective_transform,

        )
        posterior_mean_function = PosteriorMean(
            model=model,
            sampler=qmc_sampler,
            objective=network_to_objective_transform,
        )

    elif algo == "SingleGPFN_EI":
        # Model
        model = GaussianProcessNetwork(train_X=X, train_Y=network_output_at_X, dag=dag,
                          active_input_indices=active_input_indices, single_task=True)
        # Sampler
        # qmc_sampler = SobolQMCNormalSampler(num_samples=128) #for older botorch versions
        qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        # could also do this using a property decorator i think
        qmc_sampler.batch_range_override = (0, 1)

        # Acquisition function
        acquisition_function = qExpectedImprovement(
            model=model,
            best_f=objective_at_X.max().item(),
            sampler=qmc_sampler,
            objective=network_to_objective_transform,

        )
        posterior_mean_function = PosteriorMean(
            model=model,
            sampler=qmc_sampler,
            objective=network_to_objective_transform,
        )

    elif algo == "GPCF_EI":
        model = fit_gp_model(X=X, Y=network_output_at_X)
        #qmc_sampler = SobolQMCNormalSampler(num_samples=128) #for older botorch versions
        qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        acquisition_function = qExpectedImprovement(
            model=model,
            best_f=objective_at_X.max().item(),
            sampler=qmc_sampler,
            objective=network_to_objective_transform,

        )
        posterior_mean_function = PosteriorMean(
            model=model,
            sampler=qmc_sampler,
            objective=network_to_objective_transform,
        )

    elif algo[:5] == "GP_EI":
        # initializes a regular GP model based on the current data
        model = fit_gp_model(X=X, Y=objective_at_X)
        # this acquisition function tells us the expected improvement at a set of points
        acquisition_function = ExpectedImprovement(
        # acquisition_function = LogExpectedImprovement(
            model=model, best_f=objective_at_X.max().item())
        # not sure exactly how this posterior mean works
        posterior_mean_function = GPPosteriorMean(model=model)

    elif algo == "SingleGP_EI":
        # initializes a regular GP model based on the current data
        model = fit_gp_model(X=X, Y=objective_at_X, single_task=True)
        # this acquisition function tells us the expected improvement at a set of points
        acquisition_function = ExpectedImprovement(
        # acquisition_function = LogExpectedImprovement(
            model=model, best_f=objective_at_X.max().item())
        # not sure exactly how this posterior mean works
        posterior_mean_function = GPPosteriorMean(model=model)


    # "BigNN_PM_{iters}_{lr}_{weight_decay}"
    # example: "BigNN_PM_100_0.01_0.001
    if algo[:9] == "BigNN_PM_":
        # maybe do this inside the big_nn class instead
        info = algo.split("_")
        train_iters = int(info[2])
        lr = float(info[3])
        weight_decay = float(info[4])
        architecture = [None,  64, 32, 16, 1]    #?????
        model = Big_NN(dag=dag, active_input_indices=active_input_indices, architecture=architecture,
                       problem=problem, trial=trial, model_name=algo,
                       train_iters=train_iters, lr=lr, weight_decay=weight_decay)

        model.train_model(X=X, network_output_at_X=standardized_network_output_at_X)
        posterior_sample = GenericDeterministicModel(f=model.forward)
        acquisition_function = GPPosteriorMean(model=posterior_sample)
        posterior_mean_function = GPPosteriorMean(model=posterior_sample)

        sub_models = model.node_NNs

    # "NN_PM_{iters}_{lr}_{weight_decay}"
    # example: "NN_PM_100_0.01_0.001
    # later add normalization?
    elif algo[:6] == "NN_PM_":
        info = algo.split("_")
        train_iters = int(info[2])
        lr = float(info[3])
        weight_decay = float(info[4])
        architecture = [X.shape[1], 64, 32, 16, 1]
        model = NN(architecture=architecture, problem=problem, trial=trial, model_name=algo,
                   train_iters=train_iters, lr=lr, weight_decay=weight_decay)
        target = standardized_objective_at_X.unsqueeze(1)
        model.train_model(X=X, Y=target)

        # literally just a wrapper for the model with no added functionality
        posterior_sample = GenericDeterministicModel(f=model.forward)
        # passes to a AnalyticAcquisitionFunction, which will calculate mean from the posterior when given data X
        acquisition_function = GPPosteriorMean(model=posterior_sample)
        # same
        posterior_mean_function = GPPosteriorMean(model=posterior_sample)

        # # do this now so that you can do the eval stuff right after
        # new_x = optimize_acqf_and_get_suggested_point(
        #     acq_func=acquisition_function,
        #     bounds=torch.tensor([[0. for i in range(input_dim)], [
        #         1. for i in range(input_dim)]]),
        #     batch_size=1,
        #     posterior_mean=posterior_mean_function,
        # )
        #
        # model.random_preds(X=X)
        # model.random_ball_preds(X=X, d=0.1)
        # model.near_preds(X=X, d=0.1)
        # model.cross_val_preds(X=X, Y=target, num_folds=5)
        #
        # return new_x

    elif algo[:10] == "randNN_PM_":
        info = algo.split("_")
        train_iters = int(info[2])
        lr = float(info[3])
        weight_decay = float(info[4])

        import random
        # +- 1 or 2 to each layer at random
        architecture = [X.shape[1], 64, 32, 16, 1]
        for i in range(1, len(architecture) - 1):
            architecture[i] += random.choice([-2, -1, 0, 1, 2])

        model = NN(architecture=architecture, problem=problem, trial=trial, model_name=algo,
                   train_iters=train_iters, lr=lr, weight_decay=weight_decay)
        target = standardized_objective_at_X.unsqueeze(1)
        model.train_model(X=X, Y=target)

        # literally just a wrapper for the model with no added functionality
        posterior_sample = GenericDeterministicModel(f=model.forward)
        # passes to a AnalyticAcquisitionFunction, which will calculate mean from the posterior when given data X
        acquisition_function = GPPosteriorMean(model=posterior_sample)
        # same
        posterior_mean_function = GPPosteriorMean(model=posterior_sample)

    elif algo[:13] == "JointBigNN_PM":

        info = algo.split("_")
        train_iters = int(info[2])
        lr = float(info[3])
        weight_decay = float(info[4])
        architecture = [None, 64, 32, 16, 1]
        model = Joint_Big_NN(dag=dag, active_input_indices=active_input_indices, architecture=architecture,
                             problem=problem, trial=trial, model_name=algo,
                             train_iters=train_iters, lr=lr, weight_decay=weight_decay,
                             final_node_loss_weight=0.5)  # 8 * X.shape[0] / 100
        print(X.shape[0])

        model.train_model(X=X, network_output_at_X=standardized_network_output_at_X)
        posterior_sample = GenericDeterministicModel(f=model.forward)
        acquisition_function = GPPosteriorMean(model=posterior_sample)
        posterior_mean_function = GPPosteriorMean(model=posterior_sample)


    elif algo[:13] == "EnsembleNN_PM":
        info = algo.split("_")
        train_iters = int(info[2])
        lr = float(info[3])
        weight_decay = float(info[4])
        architecture = [X.shape[1], 64, 32, 16, 1]

        model_kwargs = {
            'architecture': architecture,
            'problem': problem,
            'trial': trial,
            'model_name': algo,
            'train_iters': train_iters,
            'lr': lr,
            'weight_decay': weight_decay
        }
        model = EnsembleNN(NN, 10, 0.8, **model_kwargs)
        target = standardized_objective_at_X.unsqueeze(1)
        model.fit(X, target)

        posterior_sample = GenericDeterministicModel(f=model.forward)
        acquisition_function = GPPosteriorMean(model=posterior_sample)
        posterior_mean_function = GPPosteriorMean(model=posterior_sample)


    elif algo[:16] == "EnsembleBigNN_PM":
        info = algo.split("_")
        train_iters = int(info[2])
        lr = float(info[3])
        weight_decay = float(info[4])
        architecture = [X.shape[1], 64, 32, 16, 1]

        model_kwargs = {
            'architecture': architecture,
            'problem': problem,
            'trial': trial,
            'model_name': algo,
            'train_iters': train_iters,
            'lr': lr,
            'weight_decay': weight_decay,
            'dag': dag,
            'active_input_indices': active_input_indices
        }
        model = EnsembleNN(Big_NN, 10, 0.8, **model_kwargs)
        model.fit(X, standardized_network_output_at_X)

        posterior_sample = GenericDeterministicModel(f=model.forward)
        acquisition_function = GPPosteriorMean(model=posterior_sample)
        posterior_mean_function = GPPosteriorMean(model=posterior_sample)

    elif algo[:21] == "EnsembleJointBigNN_PM":
        info = algo.split("_")
        train_iters = int(info[2])
        lr = float(info[3])
        weight_decay = float(info[4])
        architecture = [X.shape[1],  64, 32, 16, 1]

        model_kwargs = {
            'architecture': architecture,
            'problem': problem,
            'trial': trial,
            'model_name': algo,
            'train_iters': train_iters,
            'lr': lr,
            'weight_decay': weight_decay,
            'dag': dag,
            'active_input_indices': active_input_indices,
            'final_node_loss_weight': 1
        }
        model = EnsembleNN(Joint_Big_NN, 10, 0.8, **model_kwargs)
        model.fit(X, standardized_network_output_at_X)

        posterior_sample = GenericDeterministicModel(f=model.forward)
        acquisition_function = GPPosteriorMean(model=posterior_sample)
        posterior_mean_function = GPPosteriorMean(model=posterior_sample)


    elif algo[:13] == "EnsembleNN_EI":

        info = algo.split("_")
        train_iters = int(info[2])
        lr = float(info[3])
        weight_decay = float(info[4])
        architecture = [X.shape[1], 64, 32, 16, 1]

        model_kwargs = {
            'architecture': architecture,
            'problem': problem,
            'trial': trial,
            'model_name': algo,
            'train_iters': train_iters,
            'lr': lr,
            'weight_decay': weight_decay
        }
        model = EnsembleNN(NN, 10, 0.5, **model_kwargs)
        target = standardized_objective_at_X.unsqueeze(1)
        model.fit(X, target)

        acquisition_function = ExpectedImprovement(
            model=model, best_f=standardized_objective_at_X.max().item()
        )

        posterior_mean_function = GPPosteriorMean(model=model)

        # print(posterior_mean_function(test))

    elif algo[:16] == "EnsembleBigNN_EI":

        info = algo.split("_")
        train_iters = int(info[2])
        lr = float(info[3])
        weight_decay = float(info[4])
        architecture = [None,  64, 32, 16, 1]

        model_kwargs = {
            'architecture': architecture,
            'problem': problem,
            'trial': trial,
            'model_name': algo,
            'train_iters': train_iters,
            'lr': lr,
            'weight_decay': weight_decay,
            'dag': dag,
            'active_input_indices': active_input_indices
        }
        model = EnsembleNN(Big_NN, 10, 0.8, **model_kwargs)
        model.fit(X, standardized_network_output_at_X)

        acquisition_function = ExpectedImprovement(
            model=model, best_f=standardized_objective_at_X.max().item()
        )
        posterior_mean_function = GPPosteriorMean(model=model)


    elif algo[:21] == "EnsembleJointBigNN_EI":

        info = algo.split("_")
        train_iters = int(info[2])
        lr = float(info[3])
        weight_decay = float(info[4])
        architecture = [None,  64, 32, 16, 1]

        model_kwargs = {
            'architecture': architecture,
            'problem': problem,
            'trial': trial,
            'model_name': algo,
            'train_iters': train_iters,
            'lr': lr,
            'weight_decay': weight_decay,
            'dag': dag,
            'active_input_indices': active_input_indices,
            'final_node_loss_weight':1
        }
        model = EnsembleNN(Joint_Big_NN, 10, 0.8, **model_kwargs)
        model.fit(X, standardized_network_output_at_X)
        acquisition_function = ExpectedImprovement(
            model=model, best_f=standardized_objective_at_X.max().item()
        )

        # this seems to work on some problems
        posterior_mean_function = GPPosteriorMean(model=model)


    elif algo == "GP_KG":
        model = fit_gp_model(X=X, Y=objective_at_X)
        acquisition_function = qKnowledgeGradient(
            model=model, num_fantasies=8)
        posterior_mean_function = GPPosteriorMean(model=model)
    # self explanatory, pretty complicated, given all the above, finds the next point to evaluate

    if algo[:6] == "RPN_EI":
        print(X.shape)

        # create and train the RPN ensemble model
        info = algo.split("_")
        target = objective_at_X.unsqueeze(1)

        batch_size_loc = 4
        ensemble_size = 128
        fraction = 0.5
        dataset = BootstrapLoader(X.numpy(), target.numpy(), batch_size_loc, ensemble_size, fraction, 1)
        (mu_X, sigma_X), (mu_y, sigma_y) = dataset.norm_const

        # layers is another name for architecture

        # not sure if i should be normalizing it in some way here as well
        layers = [input_dim, 64, 32, 16, 1]
        nIter_RPN = 100
        model = EnsembleRegression(layers, ensemble_size)
        model.train(dataset, nIter=nIter_RPN)

        # RPN repo has its own entire acquistion model for the next point, so i could forcibly get the acuqisition function and
        # posterior mean function from the model, or i could just use theirs

        # LCB or EI by default?

        kappa = 1
        args = (kappa,)

        # Fit GMM if needed for weighted acquisition functions
        weights_fn = lambda x: np.ones(x.shape[0], )

        def predict(x):

            # accepts and returns un-normalized data
            # x = np.tile(x[np.newaxis,:,:], (ensemble_size, 1, 1))
            # x = normalize(x, mu_X, sigma_X)
            params = vmap(model.get_params)(model.opt_state)
            params_prior = vmap(model.get_params)(model.prior_opt_state)
            opt_params = (params, params_prior)

            samples = model.posterior_no_vmap(opt_params, x)
            # samples = denormalize(samples, mu_y, sigma_y)

            return samples

        acq_model = MCAcquisition(predict,
                                  bounds,
                                  *args,
                                  acq_fn='LCB',
                                  output_weights=weights_fn)

        num_restarts_acq = 1
        new_x = acq_model.next_best_point(q=1, num_restarts=num_restarts_acq)
        new_x = new_x.reshape(1, input_dim)

        new_x = torch.tensor(np.asarray(new_x))
        print(new_x)
        return new_x

    elif algo[:6] == "LLA_NN":
        info = algo.split("_")
        train_iters = int(info[2])
        lr = float(info[3])
        weight_decay = float(info[4])
        architecture = [X.shape[1], 64, 32, 16, 1]
        model = NN(architecture=architecture, problem=problem, trial=trial, model_name=algo,
                   train_iters=train_iters, lr=lr, weight_decay=weight_decay)
        target = objective_at_X.unsqueeze(1)
        # model.train_model(X=X, Y=target)
        def get_net():
            return model

        #model = LaplaceBoTorch(get_net, X, target)

        # same as GP_EI
        acquisition_function = ExpectedImprovement(
            model=model, best_f=objective_at_X.max().item())
        posterior_mean_function = GPPosteriorMean(model=model)

    elif algo[:6] == "AGD_NN":
        info = algo.split("_")
        train_iters = int(info[2])
        architecture = [X.shape[1], 16, 16, 1]
        model = AGD_NN(architecture=architecture, problem=problem, trial=trial, model_name=algo, train_iters=train_iters)

        target = objective_at_X.unsqueeze(1)
        model.train_model(X=X, Y=target)
        posterior_sample = GenericDeterministicModel(f=model.forward)
        acquisition_function = GPPosteriorMean(model=posterior_sample)
        posterior_mean_function = GPPosteriorMean(model=posterior_sample)

        # do this now so that you can do the eval stuff right after
        new_x = optimize_acqf_and_get_suggested_point(
            acq_func=acquisition_function,
            bounds=torch.tensor([[0. for i in range(input_dim)], [
                1. for i in range(input_dim)]]),
            batch_size=1,
            posterior_mean=posterior_mean_function,
        )

        model.random_preds(X=X)
        model.random_ball_preds(X=X, d=0.1)
        model.near_preds(X=X, d=0.1)
        model.cross_val_preds(X=X, Y=target, num_folds=5)

    new_x = optimize_acqf_and_get_suggested_point(
        acq_func=acquisition_function,
        bounds=torch.tensor([[0. for i in range(input_dim)], [
                            1. for i in range(input_dim)]]),
        batch_size=1,
        posterior_mean=posterior_mean_function,
    )



    if "GPFN"  in algo or efficient:
        return new_x

    # try:

        # # 100 near points
        # num_samples = 100
        # distance = 0.1
        #
        # indices = torch.randint(0, X.shape[0], (num_samples,))
        # displacements = torch.empty(num_samples, X.shape[1]).uniform_(-distance, distance)
        # X_test = X[indices] + displacements
        # X_test = torch.clamp(X_test, min=0.0, max=1.0)
        # plot_diagonal(predictions=model.posterior(X_test).mean,
        #               targets=network_to_objective_transform(function_network(X_test)),
        #               variances=model.posterior(X_test).variance,
        #               file_path=f"experiments/results_new/{problem}/{algo}/trial_{trial}/near_diag/points_{X.shape[0]}.png"
        #               )


    # 100 random out of sample points
    X_test = torch.rand(100, X.shape[1])

    if 'GP' in algo:
        # we didn't use standardized_network_output_at_X or standardized_objective_at_X, so don't use it in visualization

        # in sample points
        plot_diagonal(predictions=model.posterior(X).mean,
                      targets=objective_at_X,
                      variances=model.posterior(X).variance,
                      file_path=f"{results_folder}/in_sample_diag/points_{X.shape[0]}.png"
                      )

        plot_diagonal(predictions=(model.posterior(X).mean - model.posterior(X).mean.mean()) / model.posterior(X).mean.std(),
                      targets=(objective_at_X - objective_at_X.mean()) / objective_at_X.std(),
                      variances=torch.ones_like(model.posterior(X).variance),
                      file_path=f"{results_folder}/in_sample_diag_normed/points_{X.shape[0]}.png"
                      )
        # out sample
        plot_diagonal(predictions=model.posterior(X_test).mean,
                  targets=network_to_objective_transform(function_network(X_test)),
                  variances=model.posterior(X_test).variance,
                  file_path=f"{results_folder}/out_sample_diag/points_{X.shape[0]}.png"
                  )
        plot_diagonal(predictions=(model.posterior(X_test).mean - model.posterior(X_test).mean.mean()) / model.posterior(X_test).mean.std(),
                      targets=(network_to_objective_transform(function_network(X_test)) - network_to_objective_transform(function_network(X_test)).mean()) / network_to_objective_transform(function_network(X_test)).std(),
                      variances=torch.ones_like(model.posterior(X_test).variance),
                      file_path=f"{results_folder}/out_sample_diag_normed/points_{X.shape[0]}.png"
                      )

        plot_1d(X, model, f"{results_folder}/1d_plots/", acquisition_function, function_network, network_to_objective_transform, new_x)
    else:
        standardized_targets = (objective_at_X - objective_at_X.mean()) / objective_at_X.std()
        # in sample points
        plot_diagonal(predictions=model.posterior(X).mean,
                      targets=standardized_targets,
                      variances=model.posterior(X).variance,
                      file_path=f"{results_folder}/in_sample_diag/points_{X.shape[0]}.png"
                      )
        # out sample
        at_test = network_to_objective_transform(function_network(X_test))
        standardized_objective_at_test = (at_test - at_test.mean()) / at_test.std()
        plot_diagonal(predictions=model.posterior(X_test).mean,
                      targets=standardized_objective_at_test,
                      variances=model.posterior(X_test).variance,
                      file_path=f"{results_folder}/out_sample_diag/points_{X.shape[0]}.png"
                      )
        plot_1d(X, model, f"{results_folder}/1d_plots/", acquisition_function, function_network,
                network_to_objective_transform, new_x)

        # if sub_models is not None:
        #     for sub_model in sub_models:
        #         plot_1d(X, sub_model, f"{results_folder}/1d_plots/", acquisition_function, function_network, network_to_objective_transform, new_x)

    # except Exception as e:
    #     pass

    return new_x





def plot_diagonal(predictions: torch.Tensor, targets: torch.Tensor, variances: torch.Tensor = None, file_path: str = None):
    """
    Create a diagonal plot comparing predictions to targets and save it to a file.

    Parameters:
    predictions (torch.Tensor): A 1D tensor of predicted values.
    targets (torch.Tensor): A 1D tensor of true target values.
    variances (torch.Tensor, optional): A 1D tensor of variances for horizontal error bars.
    file_path (str): File path where the plot will be saved.

    Returns:
    None
    """
    # Convert inputs to numpy arrays for processing
    predictions = predictions.detach().squeeze(-1).squeeze(-1).cpu().numpy()
    targets = targets.detach().cpu().numpy()

    # Create the plot
    plt.figure(figsize=(12, 12))

    # Plotting the diagonal line y = x
    print(predictions.shape)
    print(targets.shape)
    lims = [np.min([predictions, targets]) - 2, np.max([predictions, targets]) + 2]
    plt.plot(lims, lims, 'k--', lw=2)  # Diagonal line

    # Scatter plot of predictions vs targets
    if variances is not None:
        variances = variances.detach().squeeze(-1).squeeze(-1).cpu().numpy()
        plt.errorbar(predictions, targets, xerr=variances, fmt='o', alpha=0.7, capsize=5)
    else:
        plt.scatter(predictions, targets, alpha=0.7)

    # Setting labels and title
    plt.xlabel('Predictions', fontsize=14)
    plt.ylabel('Targets', fontsize=14)
    plt.title('Diagonal Plot of Predictions vs Targets', fontsize=16)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.grid(True)
    plt.axhline(0, color='grey', lw=0.5, ls='--')
    plt.axvline(0, color='grey', lw=0.5, ls='--')

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save the plot to the specified file path
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()

def plot_1d(X, model, file_path, acquisition_function, function_network, network_to_objective_transform, new_x):

    dims = X.shape[1]
    for i in range(dims):
        # A = torch.full((101,dims), 0.5)
        A = new_x.repeat(101, 1)
        A[:, i] = torch.linspace(0, 1, steps=101)

        predictions = (model.posterior(A).mean).squeeze().detach()

        variances = (model.posterior(A).variance).squeeze().detach() if model.posterior(A).variance is not None else None

        x_values = np.linspace(0, 1, 101)
        fig = go.Figure()

        # Add scatter plot for means with error bars representing the variance
        fig.add_trace(go.Scatter(
            x=x_values,
            y=predictions,
            mode='markers',
            name='Mean Predictions',
        ))

        network_output_at_A = function_network(A)
        objective_at_A = network_to_objective_transform(
            network_output_at_A)

        standardized_objective_at_A = (objective_at_A - objective_at_A.mean()) / objective_at_A.std()

        fig.add_trace(go.Scatter(
            x=x_values,
            y=standardized_objective_at_A,
            mode='markers',
            name='ground truth function value',
        ))

        A = new_x.repeat(101, 1)
        A[:, i] = torch.linspace(0, 1, steps=101)
        A = A.unsqueeze(1)

        acq_values = acquisition_function(A).squeeze().detach()

        fig.add_trace(go.Scatter(
            x=x_values,
            y=acq_values,
            mode='markers',
            name='acq func',
        ))


        if variances is not None:
            fig.add_trace(go.Scatter(
                x=x_values,
                y=predictions,
                mode='lines',  # Connect the points with lines for error bars
                name='Error Bars',
                line=dict(width=0),  # Make the line invisible
                error_y=dict(
                    type='data',
                    array=np.sqrt(variances),  # Use the square root of variances for error bars
                    visible=True,
                )
            ))

        # Set plot titles and labels
        fig.update_layout(
            title="Predictions with Variance Error Bars" + str(new_x),
            xaxis_title="Input Range (0 to 1)",
            yaxis_title="Predicted Mean",
            showlegend=True
        )

        # Create directories if they don't exist
        filename = file_path + "dim" + str(i) + "/points_" + str(X.shape[0]) + ".png"
        os.makedirs(file_path + "dim" + str(i) + "/", exist_ok=True)
        fig.write_image(filename)

def plot_loss(dir, trial):
    # Directory containing the txt files
    directory = dir + "losses/"

    # Initialize lists to store data from all files
    all_losses = []

    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                # Read loss data from the file
                loss_data = [float(line.strip()) for line in file.readlines()]
                all_losses.append(loss_data)

    # get ten evenly spaced from all_losses
    all_losses = [all_losses[i] for i in range(0, len(all_losses), len(all_losses)//10)]

    # Plot all loss curves
    fig, ax = plt.subplots(figsize=(10, 6))
    num_curves = len(all_losses)
    color_map = plt.get_cmap('coolwarm')
    for i, loss_data in enumerate(all_losses, start=1):
        color = color_map((i - 1) / (num_curves - 1))  # Adjust color based on position
        ax.plot(loss_data, label=f'Curve {i}', color=color)

    ax.set_xlabel('Iterations (log scale)')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Loss Curves (log scale)')
    ax.set_yscale('log')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(vmin=0, vmax=num_curves - 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Curve Index')

    plt.grid(True)
    plt.savefig(dir + "loss_curves" + str(trial) + ".png")
    plt.close()

