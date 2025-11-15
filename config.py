from tools import utils
WRITING_ROOT = "tmp/"

#Model info
DIMS = [1, 2, 5, 10, 20, 50]
MAX_DIM = max(DIMS)
RANK = MAX_DIM//2
ROW_NORM_TARGET = .9
# EXPECTATION_SAMPLE_SIZE = 50000
# INITIAL_NPOINTS = 50
EXPECTATION_SAMPLE_SIZE = 50
INITIAL_NPOINTS = 2
NPOINTS = [min(INITIAL_NPOINTS * dim * dim, 1000) for dim in DIMS]
NSTEPS = int(1e5)
Y_MIN = 0.0
Y_MAX = 1.0
COST_FN = None
IS_THERE_DEFAULT = True
IS_Y_PARAMETER = True
TEMP = 500
MODEL_KWARGS = {
        "npoints": INITIAL_NPOINTS,
        "kernel": utils.linear_kernel,
        "y_dim": MAX_DIM,
        "is_Y_parameter": IS_Y_PARAMETER,
        "cost_fn": COST_FN,
        "y_min": Y_MIN,
        "y_max": Y_MAX,
        "is_there_default": IS_THERE_DEFAULT,
        "temp": TEMP,
    }


#modes
MODES = ["soft", "ste"]

#OPTIMIZERS
ADAM = {
    "lr": 1e-2,
    "betas": (0.9, 0.95),
    "weight_decay": .0,
    "amsgrad": True,
}

SGD = {
    "lr": 5e-4,
    "momentum": 0.9,
    "nesterov": True,
}
OPTIMIZERS_KWARGS_DICT = {
    "soft": ADAM,
    "hard": ADAM,
    "ste": SGD
}

# Scheduler
REDUCE_ON_PLATEAU = {
    "patience": 200,
    "threshold": 1e-4,
    "factor": .5,
    "cooldown": 200,
    "eps": 1e-8,
}
COSINE_ANNEALING = {
    "T_max": 2000,
    "eta_min": 1e-9,
}
SCHEDULERS_KWARGS_DICT = {
    "soft": REDUCE_ON_PLATEAU,
    "hard": REDUCE_ON_PLATEAU,
    "ste": COSINE_ANNEALING,
}


#Train KWARGS
WINDOW = 400
CONSTRAINT_FNS = [utils.model_min_constraint, utils.model_max_constraint]
USE_WANDB = False
INITIAL_PENALTY_FACTOR = 1000.
TRAIN_KWARGS = {
    "constraint_fns": CONSTRAINT_FNS,
    "initial_penalty_factor": INITIAL_PENALTY_FACTOR,
    "epsilon": 5e-5,
    "nsteps": NSTEPS,
    "use_wandb": USE_WANDB,
    "convergence_tolerance": 1e-7,
    "steps_per_snapshot": 1000,
    "switch_threshold": 0.99,
    "switch_patience": 1000,
}

PATH_RELEVANT_KWARGS = {
    "sample_size": EXPECTATION_SAMPLE_SIZE,
    "kernel": utils.linear_kernel,
    "max_dim": MAX_DIM,
    "is_there_default": IS_THERE_DEFAULT,    
}