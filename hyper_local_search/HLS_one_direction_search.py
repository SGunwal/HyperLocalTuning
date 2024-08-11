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
from curvature_information import *

########################################### SETTING UP DATA ###########################################

# Loading Datasets as globals
BASE_DIRECTORY         = "."
DF_PREPROCESSED_DIR    = BASE_DIRECTORY + "/datasets/House_Price_Prediction/house_price_prediction.pickle"
BASE_DIRECTORY         = BASE_DIRECTORY + "/outputs"

global x_train, x_val, x_test, y_train, y_val, y_test
x_train, x_val, x_test, y_train, y_val, y_test = load_dataset(DF_PREPROCESSED_DIR)

# RUNNING ALL SIMULATIONS

list_of_all_samplers    = ['grid', 'random', 'qmc', 'tpe']
num_simulation_trials   = 50
number_of_optuna_trials = [100]

##########################################################################################################################

class GoldenSectionSearch:

    def __init__(self, init_hyperparameters, hyperparameter_direction, X, y):

        self.init_hyperparameters = np.array(init_hyperparameters)
        self.hyperparameter_direction = np.array(hyperparameter_direction)/np.linalg.norm(hyperparameter_direction)
        self.X = X # validation
        self.y = y # validation

    def compute_loss(self, weights, bias):
        y_pred = tf.reduce_sum(tf.multiply(self.X,weights), axis=1) + bias
        loss = tf.keras.losses.MeanSquaredError()(y_true=self.y, y_pred=y_pred)
        return loss

    def interval_search(self, init_t, step):
        a = init_t
        b = init_t + step

        a_hyperparams = self.init_hyperparameters + a * self.hyperparameter_direction
        b_hyperparams = self.init_hyperparameters + b * self.hyperparameter_direction
        a_hyperparams = np.maximum(a_hyperparams, 1e-10)
        b_hyperparams = np.maximum(b_hyperparams, 1e-10)

        _ , a_optimal_weights, a_optimal_bias, _ = elastic_net_regression(x_train, y_train, a_hyperparams[0], a_hyperparams[1])
        _ , b_optimal_weights, b_optimal_bias, _ = elastic_net_regression(x_train, y_train, b_hyperparams[0], b_hyperparams[1])

        loss_a = self.compute_loss(a_optimal_weights, a_optimal_bias)
        loss_b = self.compute_loss(b_optimal_weights, b_optimal_bias)

        i = 0
        while loss_b < loss_a:
            i += 1
            a = b
            loss_a = loss_b
            b = init_t + (2 ** i) * step

            b_hyperparams                            = self.init_hyperparameters + b * self.hyperparameter_direction
            b_hyperparams                            = np.maximum(b_hyperparams, 1e-10)
            _ , b_optimal_weights, b_optimal_bias, _ = elastic_net_regression(x_train, y_train, b_hyperparams[0], b_hyperparams[1])
            loss_b                                   = self.compute_loss(b_optimal_weights, b_optimal_bias)
 
        if i == 0:
            return a, b
        elif i == 1:
            return init_t, b
        else:
            return init_t + (2 ** (i - 2)) * step, b

    def golden_section_search(self, delta_1, delta_2, tol=1e-5):
        gr = (1 + 5 ** 0.5) / 2  # Golden ratio

        a = delta_1
        b = delta_2
        c = b - (b - a) / gr
        d = a + (b - a) / gr

        while abs(c - d) > tol:

            c_hyperparams = self.init_hyperparameters + c * self.hyperparameter_direction
            d_hyperparams = self.init_hyperparameters + d * self.hyperparameter_direction
            c_hyperparams = np.maximum(c_hyperparams, 1e-10)
            d_hyperparams = np.maximum(d_hyperparams, 1e-10)

            _ , c_optimal_weights, c_optimal_bias, _ = elastic_net_regression(x_train, y_train, c_hyperparams[0], c_hyperparams[1])
            _ , d_optimal_weights, d_optimal_bias, _ = elastic_net_regression(x_train, y_train, d_hyperparams[0], d_hyperparams[1])

            loss_c = self.compute_loss(c_optimal_weights, c_optimal_bias)
            loss_d = self.compute_loss(d_optimal_weights, d_optimal_bias)


            if loss_c < loss_d:
                b = d
            else:
                a = c

            c = b - (b - a) / gr
            d = a + (b - a) / gr

        optimal_delta = (a + b) / 2

        optimal_hyperparameters               = self.init_hyperparameters + optimal_delta * self.hyperparameter_direction
        optimal_hyperparameters               = np.maximum(optimal_hyperparameters, 1e-10)
        _ , optimal_weights, optimal_bias, _  = elastic_net_regression(x_train, y_train, optimal_hyperparameters[0], optimal_hyperparameters[1])
        optimal_loss                          = self.compute_loss(optimal_weights, optimal_bias)

        return optimal_hyperparameters, optimal_weights, optimal_bias, optimal_loss, optimal_delta

    def find_optimal_weights(self, init_t=0, step=0.1, tol=1e-5):
        delta_1, delta_2 = self.interval_search(init_t, step)
        print(f"delta_1 = {delta_1}, delta_2 = {delta_2}")
        optimal_hyperparameters, optimal_weights, optimal_bias, optimal_loss, optimal_delta = self.golden_section_search(delta_1, delta_2, tol)
        return optimal_hyperparameters, optimal_weights, optimal_bias, optimal_loss, optimal_delta


def hls_direction_search():

    full_results_dictionary = {}
    hls_final_model = {}
    for sampler_type in list_of_all_samplers:
        full_results_dictionary[f"{sampler_type}"] = {}
        hls_final_model["hls_steps"+sampler_type] = []
        hls_final_model["hls_hyperparams"+sampler_type] = [] 

        for optuna_trials in number_of_optuna_trials: # 100

            full_results_dictionary[f"{sampler_type}"][f"{optuna_trials}"] = {} #{ "Training_Losses":[], "Validation_Losses":[], "Testing_Losses":[] }
            tr_optuna_losses, tr_hls_losses, val_optuna_losses, val_hls_losses, test_optuna_losses, test_hls_losses = [], [], [], [], [], []
            optuna_hyperparams, hls_hyperparams = [], []
            optuna_times, hls_times = [], []

            for simulation_number in range(num_simulation_trials):

                # LOADING OPTUNA TRAINED MODELS
                OPTUNA_MODEL_DIRECTORY = BASE_DIRECTORY + f"/all_optuna_models_{num_simulation_trials}_simulations.pickle" 
                with open(OPTUNA_MODEL_DIRECTORY, "rb") as fout:
                        optuna_trained_models = pkl.load(fout)
                model_info = optuna_trained_models[f'{sampler_type}_{optuna_trials}_{simulation_number}']
                [ optuna_model_weights, optuna_model_bias, init_hyperparameters, optuna_time, optimal_trial_number ] = model_info

                ###################################################### HLS #############################################################

                starting_hls_time = datetime.now()
                # GATHERING DIRECTION DATA
                objective_w_b             = exact_gradient_validation(x_val, y_val, optuna_model_weights, optuna_model_bias, batch_size=None)
                constraint_matrix         = constraint_coefficients(x_train, [list(optuna_model_weights)] + [[optuna_model_bias]], init_hyperparameters)
                number_of_hyperparameters = len(init_hyperparameters)

                solution, objective, runtime                                             = Bilevel_Descent_Direction( objective_w_b, constraint_matrix, number_of_hyperparameters, 0.0001 )
                hyperparameter_direction, model_weights_direction , model_bias_direction = unflatten( solution, number_of_hyperparameters )

                # GOLDEN SECTION SEARCH

                GSS_Obj  = GoldenSectionSearch(init_hyperparameters, hyperparameter_direction, x_val, y_val)
                
                optimal_hyperparameters, optimal_weights, optimal_bias, optimal_loss, optimal_delta = GSS_Obj.find_optimal_weights()
                model, optimal_weights, optimal_bias, mse_loss                                      = elastic_net_regression(x_train, y_train, optimal_hyperparameters[0], optimal_hyperparameters[1])

                # Evaluating optimal loss
                optuna_train_loss  = evaluate_loss(x_train, y_train, optuna_model_weights, optuna_model_bias)
                optimal_train_loss = evaluate_loss(x_train, y_train, optimal_weights, optimal_bias)

                optuna_test_loss  = evaluate_loss( x_test, y_test, optuna_model_weights, optuna_model_bias )
                optimal_test_loss = evaluate_loss( x_test, y_test, optimal_weights, optimal_bias )

                optuna_val_loss  = evaluate_loss( x_val, y_val, optuna_model_weights, optuna_model_bias)
                optimal_val_loss = evaluate_loss( x_val, y_val, optimal_weights, optimal_bias)

                end_hls_time   = datetime.now()
                total_hls_time = end_hls_time - starting_hls_time

                optuna_times.append(optuna_time.total_seconds())
                hls_times.append(total_hls_time.total_seconds())

                tr_optuna_losses.append(optuna_train_loss.numpy())
                tr_hls_losses.append(optimal_train_loss.numpy())
                val_optuna_losses.append(optuna_val_loss.numpy())
                val_hls_losses.append(optimal_val_loss.numpy())
                test_optuna_losses.append(optuna_test_loss.numpy())
                test_hls_losses.append(optimal_test_loss.numpy())

                optuna_hyperparams.append([init_hyperparameters[0].numpy(),init_hyperparameters[1].numpy()]) 
                hls_hyperparams.append(optimal_hyperparameters)


            # Updating the HLS steps
            hls_final_model["hls_steps"+sampler_type].append([optimal_weights, optimal_bias])
            hls_final_model["hls_hyperparams"+sampler_type].append(optimal_hyperparameters)

            full_results_dictionary[f"{sampler_type}"][f"{optuna_trials}"]["Training_Losses"]   = [tr_optuna_losses,tr_hls_losses]
            full_results_dictionary[f"{sampler_type}"][f"{optuna_trials}"]["Validation_Losses"] = [val_optuna_losses,val_hls_losses]
            full_results_dictionary[f"{sampler_type}"][f"{optuna_trials}"]["Testing_Losses"]    = [test_optuna_losses,test_hls_losses]
            full_results_dictionary[f"{sampler_type}"][f"{optuna_trials}"]["Runtimes"]          = [optuna_times,hls_times]
            full_results_dictionary[f"{sampler_type}"][f"{optuna_trials}"]["Hyperparams"]       = [optuna_hyperparams,hls_hyperparams]

    return full_results_dictionary, hls_final_model


if __name__ == '__main__':

    full_results_dictionary, hls_final_model = hls_direction_search()

    # Saving the output
    with open( BASE_DIRECTORY + f"/final_hls_output.pickle", "wb") as fout:
        pkl.dump(full_results_dictionary, fout)
    
    # Saving the output for plots
    with open( BASE_DIRECTORY + f"/final_hls_model.pickle", "wb") as fout:
        pkl.dump(hls_final_model, fout)