import sys
import os
from functools import partial

# Determine the absolute path to the project root directory
script_dir   = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
utils_path   = os.path.join(project_root, 'utils')
sys.path.append(utils_path)

# IMPORT LIBRARIES/MODULES
from imports import *
from helper_functions import *

#################################################################################################################
########################################### OPTUNA RUN SETTINGS #################################################
#################################################################################################################

# GLOBALS
SEED                = None #111
all_samplers        = ['grid', 'random', 'qmc', 'tpe'] # hyperparameter samplers  
L1_hyperparam_range = [1e-10,10]                        # lasso hyperparameter range to sample from
L2_hyperparam_range = [1e-10,10]                        # ridge hyperparameter range to sample from

number_of_simulations_per_trial = 50                   # number of simulations for each sampler
number_of_optuna_trials         = [100]                 # number of samplings/iterations with each sampler. (Use single entry for now)

# SET DIRECTORIES BELOW
BASE_DIRECTORY         = "."
DF_PREPROCESSED_DIR    = BASE_DIRECTORY + "/datasets/House_Price_Prediction/house_price_prediction.pickle" # Data Directory
WORKING_BASE_DIRECTORY = BASE_DIRECTORY  + "/outputs"  #/plots"                                            # Directory to output models

###################################################################

# SAVING INCUMBENT LOSSES and HYPERPARAMS - INITIALIZING (NOT YET READY FOR MULTIPLE ENTRIES IN "number_of_optuna_trials")
model_steps = {} 
for sampler_ in all_samplers:
    model_steps["optuna_steps"+sampler_]       = {}
    model_steps["optuna_hyperparams"+sampler_] = {}
    for sim_ in range(number_of_simulations_per_trial):
        model_steps["optuna_steps"+sampler_][f"{sim_}"]       = []
        model_steps["optuna_hyperparams"+sampler_][f"{sim_}"] = []

#################################################################################################################
#################################################################################################################

def optuna_optimizer(trial, sampler, datasets):

    hyper_l1 = trial.suggest_float('l1', L1_hyperparam_range[0], L1_hyperparam_range[1]) # L1 - Lasso
    hyper_l2 = trial.suggest_float('l2', L2_hyperparam_range[0], L2_hyperparam_range[1]) # L2 - Ridge

    [x_train, x_val, x_test, y_train, y_val, y_test] = datasets
    model, optimal_weights, optimal_bias, _          = elastic_net_regression(x_train, y_train, hyper_l1, hyper_l2)
    val_loss_unregularized                           = evaluate_loss(x_val, y_val, optimal_weights, optimal_bias)

    score = val_loss_unregularized
    if score < optuna_optimizer.best_score:
        model_steps["optuna_steps"+sampler][f"{SIM_NUM}"].append([optimal_weights, optimal_bias])
        model_steps["optuna_hyperparams"+sampler][f"{SIM_NUM}"].append([hyper_l1, hyper_l2])
        optuna_optimizer.best_score = score
        with open( WORKING_BASE_DIRECTORY + "/best_optuna_model.pickle", "wb") as fout:
            pkl.dump(model, fout)

    return score

def optuna_training(num_trials, datasets, sampler):

    time1 = datetime.now()
    if sampler == 'grid':
        search_space       = {}
        num_grid_trials    = int(np.ceil(num_trials**0.5))
        search_space["l1"] = list( np.linspace( L1_hyperparam_range[0], L1_hyperparam_range[1], num_grid_trials ) )
        search_space["l2"] = list( np.linspace( L2_hyperparam_range[0], L2_hyperparam_range[1], num_grid_trials ) )
        study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space, seed=SEED), direction = "minimize") # seed=SEED
    elif sampler == 'random':
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=SEED), direction = "minimize")
    elif sampler == 'qmc':
        study = optuna.create_study(sampler=optuna.samplers.QMCSampler(seed=SEED), direction = "minimize")
    elif sampler == 'tpe':
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=SEED), direction = "minimize")
    else:
        print(" Using default sampler.... TPESampler")
        study = optuna.create_study(direction="minimize")

    study.optimize(partial(optuna_optimizer, sampler=sampler, datasets=datasets), n_trials=num_trials)

    trial = study.best_trial
    time2 = datetime.now()
    delta = time2 - time1

    with open( WORKING_BASE_DIRECTORY + "/best_optuna_model.pickle", "rb") as fin:
        best_clf = pkl.load(fin)

    optuna_model_weights = best_clf.coef_
    optuna_model_bias    = best_clf.intercept_
    init_hyperparameters = [tf.Variable(value) for _ , value in trial.params.items()]

    return optuna_model_weights, optuna_model_bias, init_hyperparameters, delta, trial.number

def run_optuna_training(sampler, datasets, trials, simulation_num):

    global SIM_NUM
    SIM_NUM = simulation_num

    optuna_optimizer.best_score = float('inf')  # initial score set to a very large number
    optuna_model_weights, optuna_model_bias, init_hyperparameters, optuna_time, optimal_trial_number = optuna_training(trials, datasets, sampler)
    model__ = [optuna_model_weights, optuna_model_bias, init_hyperparameters, optuna_time, optimal_trial_number] # output into backup_dictionary

    return model__

if __name__ == "__main__":

    backup_dictionary = {}

    x_train, x_val, x_test, y_train, y_val, y_test = load_dataset(DF_PREPROCESSED_DIR)
    datasets = [x_train, x_val, x_test, y_train, y_val, y_test]

    for num_trials in number_of_optuna_trials:
        for sampler in all_samplers:
            for simulation_num in range(number_of_simulations_per_trial):
                model_info                    = run_optuna_training(sampler, datasets, num_trials, simulation_num)
                backup_dictionary[f'{sampler}_{num_trials}_{simulation_num}'] = model_info

    # SAVING THE MODEL AND INCUMBENT LOSS INFO
    with open( WORKING_BASE_DIRECTORY + f"/all_optuna_models_{number_of_simulations_per_trial}_simulations.pickle", "wb") as fout:
            pkl.dump(backup_dictionary, fout)
    with open( WORKING_BASE_DIRECTORY + f"/optuna_steps_dict.pickle", "wb") as fout:
            pkl.dump(model_steps, fout)