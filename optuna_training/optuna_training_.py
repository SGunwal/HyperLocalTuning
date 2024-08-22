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
###########################################  OPTUNA RUNS   ######################################################
#################################################################################################################

def optuna_optimizer(trial, optuna_setting_file):

    L1_hyperparam_range            = optuna_setting_file["Hyperparameter Range"]["L1"]
    L2_hyperparam_range            = optuna_setting_file["Hyperparameter Range"]["L2"]
    sampler                        = optuna_setting_file["Current Sampler"]
    simulation_number              = optuna_setting_file["Current Simulation Number"]

    hyper_l1 = trial.suggest_float('l1', L1_hyperparam_range[0], L1_hyperparam_range[1]) # L1 - Lasso
    hyper_l2 = trial.suggest_float('l2', L2_hyperparam_range[0], L2_hyperparam_range[1]) # L2 - Ridge

    [x_train, x_val, x_test, y_train, y_val, y_test] = optuna_setting_file["Datasets"]
    model, optimal_weights, optimal_bias, _          = elastic_net_regression(x_train, y_train, hyper_l1, hyper_l2)
    val_loss_unregularized                           = evaluate_loss(x_val, y_val, optimal_weights, optimal_bias)

    score = val_loss_unregularized
    if score < optuna_optimizer.best_score:
        incumbent_solutions_dict = optuna_setting_file["Incumbent Solutions Dictionary"]
        incumbent_solutions_dict["optuna_steps"+sampler][f"{simulation_number}"].append([optimal_weights, optimal_bias])
        incumbent_solutions_dict["optuna_hyperparams"+sampler][f"{simulation_number}"].append([hyper_l1, hyper_l2])
        
        optuna_optimizer.best_score = score
        with open( optuna_setting_file["Working Directory Path"] + "/best_optuna_model.pickle", "wb") as fout:
            pkl.dump(model, fout)

    return score

def optuna_training(Optuna_Setting_Dictionary):

    total_number_of_optuna_trials  = Optuna_Setting_Dictionary["Number of Optuna Trials/Epochs"]
    sampler                        = Optuna_Setting_Dictionary["Current Sampler"]
    random_seed_for_sampling       = Optuna_Setting_Dictionary["Random Seed"]
    L1_hyperparam_range            = Optuna_Setting_Dictionary["Hyperparameter Range"]["L1"]
    L2_hyperparam_range            = Optuna_Setting_Dictionary["Hyperparameter Range"]["L2"]

    time1 = datetime.now()

    if sampler == 'grid':
        search_space       = {}
        num_grid_trials    = int(np.ceil(total_number_of_optuna_trials**0.5))
        search_space["l1"] = list( np.linspace( L1_hyperparam_range[0], L1_hyperparam_range[1], num_grid_trials ) )
        search_space["l2"] = list( np.linspace( L2_hyperparam_range[0], L2_hyperparam_range[1], num_grid_trials ) )
        study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space, seed=random_seed_for_sampling), direction = "minimize")

    elif sampler == 'random':
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=random_seed_for_sampling), direction = "minimize")

    elif sampler == 'qmc':
        study = optuna.create_study(sampler=optuna.samplers.QMCSampler(seed=random_seed_for_sampling), direction = "minimize")

    elif sampler == 'tpe':
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=random_seed_for_sampling), direction = "minimize")

    else:
        print(" Using default sampler.... TPESampler")
        study = optuna.create_study(direction="minimize")

    study.optimize(partial(optuna_optimizer, optuna_setting_file = Optuna_Setting_Dictionary), n_trials = total_number_of_optuna_trials)

    trial = study.best_trial
    time2 = datetime.now()
    delta = time2 - time1

    with open( Optuna_Setting_Dictionary["Working Directory Path"] + "/best_optuna_model.pickle", "rb") as fin:
        best_clf = pkl.load(fin)

    optuna_model_weights = best_clf.coef_
    optuna_model_bias    = best_clf.intercept_
    init_hyperparameters = [tf.Variable(value) for _ , value in trial.params.items()]

    return optuna_model_weights, optuna_model_bias, init_hyperparameters, delta, trial.number

def setting_up_optuna_training(optuna_settings):

    x_train, x_val, x_test, y_train, y_val, y_test = load_dataset(optuna_settings["Dataset Directory Path"])
    optuna_settings["Datasets"]                    = [x_train, x_val, x_test, y_train, y_val, y_test]

    total_number_of_optuna_trials                  = optuna_settings["Number of Optuna Trials/Epochs"]
    number_of_simulations_per_trial                = optuna_settings["Number of Simlations"]
    all_samplers                                   = optuna_settings["Samplers"]

    # Initiating Dictionary for Tracking Incumbent Optuna Solutions
    model_steps_dictionary = {} 
    for sampler_ in all_samplers:
        model_steps_dictionary["optuna_steps"+sampler_]       = {}
        model_steps_dictionary["optuna_hyperparams"+sampler_] = {}
        for sim_ in range(number_of_simulations_per_trial):
            model_steps_dictionary["optuna_steps"+sampler_][f"{sim_}"]       = []
            model_steps_dictionary["optuna_hyperparams"+sampler_][f"{sim_}"] = []
    optuna_settings["Incumbent Solutions Dictionary"] = model_steps_dictionary

    backup_dictionary = {}
    for sampler in all_samplers:
        for simulation_num in range(number_of_simulations_per_trial):
            optuna_optimizer.best_score = float('inf')
            optuna_settings["Current Simulation Number"] = simulation_num
            optuna_settings["Current Sampler"]           = sampler
            backup_dictionary[f'{sampler}_{total_number_of_optuna_trials}_{simulation_num}'] = optuna_training(optuna_settings)

    # SAVING THE MODEL AND INCUMBENT LOSS INFO
    with open( optuna_settings["Working Directory Path"] + f"/all_optuna_models_{number_of_simulations_per_trial}_simulations.pickle", "wb") as fout:
            pkl.dump(backup_dictionary, fout)
    with open( optuna_settings["Working Directory Path"] + f"/optuna_steps_dict.pickle", "wb") as fout:
            pkl.dump(optuna_settings["Incumbent Solutions Dictionary"], fout)

# if __name__ == "__main__":

#     ################################# INPUTS #################################
#     BASE_DIRECTORY         = "."
#     DF_PREPROCESSED_DIR    = BASE_DIRECTORY + "/datasets/Simulation_Dataset/simulation_dataset.pickle" # Data Path
#     WORKING_BASE_DIRECTORY = BASE_DIRECTORY  + "/outputs"   

#     optuna_model_settings = { "Random Seed"          : None,
#                               "Samplers"             : [ 'grid', 'random',  'tpe', 'qmc' ], 
#                               "Hyperparameter Range" : { "L1": [1e-10,1], 
#                                                          "L2": [1e-10,1] 
#                                                         }, 
#                               "Number of Optuna Trials/Epochs" : 100,                     # Total number of optuna trials to run
#                               "Number of Simlations"           : 50,                       # Total number of simulations
#                               "Dataset Directory Path"         : DF_PREPROCESSED_DIR,     # dataset path
#                               "Working Directory Path"         : WORKING_BASE_DIRECTORY   # all models will be saved here
#                             }
    
#     ##################################################################
    
#     setting_up_optuna_training(optuna_model_settings)