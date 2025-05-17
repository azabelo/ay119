import torch
from torch import Tensor
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.generation.gen import get_best_candidates
from botorch.optim import optimize_acqf
from botorch.optim.initializers import (
    gen_batch_initial_conditions,
    gen_one_shot_kg_initial_conditions,
)

# from bofn.acquisition_function_optimization.custom_acqf_optimizer import custom_optimize_acqf


def optimize_acqf_and_get_suggested_point(
    acq_func,
    bounds,
    batch_size,
    posterior_mean=None,
    ) -> Tensor:
    """Optimizes the acquisition function, and returns a new candidate."""
    input_dim = bounds.shape[1]
    num_restarts=5*input_dim
    raw_samples=50*input_dim



    ic_gen = (
        gen_one_shot_kg_initial_conditions
        if isinstance(acq_func, qKnowledgeGradient)
        else gen_batch_initial_conditions
    )
    batch_initial_conditions = ic_gen(
        acq_function=acq_func,
        bounds=bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options={"batch_limit": num_restarts},
    )
    baseline_candidate = None


    if False:
        baseline_candidate, baseline_candidate_acquisition_value = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        batch_initial_conditions=batch_initial_conditions,
        options={
            "batch_limit": 5,
            "init_batch_limit": 50,
            "maxiter": 100,
            "nonnegative": False,
            "method": "L-BFGS-B",
        },
        return_best_only=True,
    )



        if isinstance(acq_func, qKnowledgeGradient):
            augmented_q_batch_size = acq_func.get_augmented_q_batch_size(batch_size)
            baseline_candidate = baseline_candidate.detach().repeat(1, augmented_q_batch_size, 1)
        else:
            baseline_candidate = baseline_candidate.detach().view(torch.Size([1, batch_size, input_dim]))
        
        batch_initial_conditions = torch.cat([batch_initial_conditions, baseline_candidate], 0)
        num_restarts += 1

    candidates, acq_values = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        batch_initial_conditions=batch_initial_conditions,
        options={
            "batch_limit": 5,
            "init_batch_limit": 50,
            "maxiter": 100,
            "nonnegative": False,
            "method": "L-BFGS-B",
        },
        return_best_only=False,
    )

    candidates = candidates.detach()
    acq_values_sorted, indices = torch.sort(acq_values.squeeze(), descending=True)

    new_x = get_best_candidates(batch_candidates=candidates, batch_values=acq_values)

    print("chosen:", new_x)

    return new_x
