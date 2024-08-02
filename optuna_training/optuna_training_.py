import sys
import os

# Determine the absolute path to the project root directory
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))

# Add the utils directory to the Python path
utils_path = os.path.join(project_root, 'utils')
sys.path.append(utils_path)


# IMPORT LIBRARIES/MODULES

from imports import *
from helper_functions import *

# SET RANDOM SEET
SEED = None
# np.random.seed(SEED)
# tf.random.set_seed(SEED)

# SET DIRECTORIES
BASE_DIRECTORY = "."
DF_PREPROCESSED_DIR = BASE_DIRECTORY + "/datasets/House_Price_Prediction/house_price_prediction.pickle"
#################################################################################################################

def optuna_optimizer(trial):

    hyper_l1 = trial.suggest_float('l1', 1e-10, 1.0) # L1 - Lasso
    hyper_l2 = trial.suggest_float('l2', 1e-10, 1.0) # L2 - Ridge

    model, optimal_weights, optimal_bias, _ = elastic_net_regression(x_train, y_train, hyper_l1, hyper_l2)

    train_loss_unregularized = evaluate_loss(x_train, y_train, optimal_weights, optimal_bias)
    val_loss_unregularized   = evaluate_loss(x_val, y_val, optimal_weights, optimal_bias)
    test_loss_unregularized  = evaluate_loss(x_test, y_test, optimal_weights, optimal_bias)

    print("\nTraining:  Loss ", train_loss_unregularized)
    print("Validation: Loss ", val_loss_unregularized)
    print("Test:       Loss ", test_loss_unregularized, "\n\n")

    score = val_loss_unregularized
    if score < optuna_optimizer.best_score:
        optuna_optimizer.best_score = score
        with open( BASE_DIRECTORY + "/outputs/best_optuna_model.pickle", "wb") as fout:
            pkl.dump(model, fout)
        print("Updated best model and training info with new best score: ", score)

    return score

def optuna_training(num_trials):
    time1 = datetime.now()

    if TYPE_OF_SAMPLER == 'grid':
        search_space = { f"l{i}" : list(np.linspace(1e-10,1,2*num_trials)) for i in range(1,3) }
        study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space, seed=SEED), direction = "minimize") # seed=SEED
    elif TYPE_OF_SAMPLER == 'random':
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=SEED), direction = "minimize")
    elif TYPE_OF_SAMPLER == 'qmc':
        study = optuna.create_study(sampler=optuna.samplers.QMCSampler(seed=SEED), direction = "minimize")
    elif TYPE_OF_SAMPLER == 'tpe':
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=SEED), direction = "minimize")
    else:
        print(" Using default sampler.... TPESampler")
        study = optuna.create_study(direction="minimize")
    
    print("\n Using sampler = ", TYPE_OF_SAMPLER, "\n\n" )

    study.optimize(optuna_optimizer, n_trials=num_trials)

    print('\n\n')
    trial = study.best_trial
    print("Best Score: ", trial.value)
    print("Best Params: ")
    for key, value in trial.params.items():
        print("  {}: {}".format(key, value))

    print("\n\n ", "Trial Number: ", trial.number, "\n")

    time2 = datetime.now()
    delta = time2 - time1
    print(f"Time difference is {delta.total_seconds()} seconds")

    with open( BASE_DIRECTORY + "/outputs/best_optuna_model.pickle", "rb") as fin:
        best_clf = pkl.load(fin)

    optuna_model_weights = best_clf.coef_
    optuna_model_bias    = best_clf.intercept_
    init_hyperparameters = [tf.Variable(value) for key, value in trial.params.items()]

    return optuna_model_weights, optuna_model_bias, init_hyperparameters, delta

def run_optuna_training(sampler, trials):

    global TYPE_OF_SAMPLER
    global x_train, x_val, x_test, y_train, y_val, y_test
    # Loading dataset
    x_train, x_val, x_test, y_train, y_val, y_test = load_dataset(DF_PREPROCESSED_DIR)
    # Running Optuna Training
    TYPE_OF_SAMPLER = sampler # 'grid', 'random', 'qmc', 'tpe'
    NUM_OPTUNA_TRIALS = trials
    optuna_optimizer.best_score = float('inf')  # initial score set to a very large number
    optuna_model_weights, optuna_model_bias, init_hyperparameters, optuna_time = optuna_training(NUM_OPTUNA_TRIALS)
    model__ = [optuna_model_weights, optuna_model_bias, init_hyperparameters, optuna_time]

    return model__

if __name__ == "__main__":

    all_samplers = ['grid', 'random', 'qmc', 'tpe']

    number_of_simulations_per_trial = 50
    number_of_optuna_trials = [100] 

    backup_dictionary = {}

    for num_trials in number_of_optuna_trials:
        for sampler in all_samplers:
            for simulation_num in range(number_of_simulations_per_trial):
                model_info                    = run_optuna_training(sampler, num_trials)
                backup_dictionary[f'{sampler}_{num_trials}_{simulation_num}'] = model_info
    with open( BASE_DIRECTORY + f"/outputs/all_optuna_models_{number_of_simulations_per_trial}_simulations.pickle", "wb") as fout:
            pkl.dump(backup_dictionary, fout)