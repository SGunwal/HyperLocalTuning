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

##########################################################################################################################

class GoldenSectionSearch:

    def __init__(self, init_hyperparameters, hyperparameter_direction, X, y):

        self.init_hyperparameters = np.array(init_hyperparameters)
        self.hyperparameter_direction = np.array(hyperparameter_direction)/np.linalg.norm(hyperparameter_direction)
        self.X = X
        self.y = y

    def compute_loss(self, weights, bias):
        y_pred = tf.reduce_sum(tf.multiply(self.X,weights), axis=1) + bias
        loss = tf.keras.losses.MeanSquaredError()(y_true=self.y, y_pred=y_pred)
        return loss

    def interval_search(self, init_t=0, step=0.1):
        a = init_t
        b = init_t + step

        a_hyperparams = self.init_hyperparameters + a * self.hyperparameter_direction
        b_hyperparams = self.init_hyperparameters + b * self.hyperparameter_direction

        _ , a_optimal_weights, a_optimal_bias, _ = elastic_net_regression(x_train, y_train, a_hyperparams[0], b_hyperparams[1])
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

if __name__ == '__main__':

    # Loading Datasets as globals
    BASE_DIRECTORY         = "."
    DF_PREPROCESSED_DIR = BASE_DIRECTORY + "/datasets/California_Housing_Dataset/preprocessed_housing.pickle"

    global x_train, x_val, x_test, y_train, y_val, y_test
    x_train, x_val, x_test, y_train, y_val, y_test = load_dataset(DF_PREPROCESSED_DIR)
    # Model Type
    sampler    = 'grid'
    num_trials = 100
    simulation_num = 10

    # LOADING OPTUNA TRAINED MODELS
    OPTUNA_MODEL_DIRECTORY = BASE_DIRECTORY + "/all_optuna_models.pickle"
    with open(OPTUNA_MODEL_DIRECTORY, "rb") as fout:
            optuna_trained_models = pkl.load(fout)
    model_info = optuna_trained_models[f'{sampler}_{num_trials}_{simulation_num}']
    [ optuna_model_weights, optuna_model_bias, init_hyperparameters, optuna_time ] = model_info

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

    print("\nTraining losses = ", optuna_train_loss, optimal_train_loss)
    print("Validation losses = ", optuna_val_loss, optimal_val_loss)
    print("Test losses = ", optuna_test_loss, optimal_test_loss)


