import sys
import os

# Determine the absolute path to the project root directory
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))

# Add the utils directory to the Python path
utils_path  = os.path.join(project_root, 'utils')
optuna_path = os.path.join(project_root, 'optuna_training') 
sys.path.append(utils_path)
sys.path.append(optuna_path)

# IMPORT LIBRARIES/MODULES
from imports import *
from helper_functions import *
from curvature_information import *
from optuna_training_ import *

###########################################################################################################
                            # FUNCTIONS FOR HYPER LOCAL SEARCH
###########################################################################################################

class GoldenSectionSearch:

    def __init__(self, init_hyperparameters, hyperparameter_direction, all_dataset):

        self.init_hyperparameters = np.array(init_hyperparameters)
        self.hyperparameter_direction = np.array(hyperparameter_direction)/np.linalg.norm(hyperparameter_direction)
        self.X = all_dataset['x_val'] # validation
        self.y = all_dataset['y_val'] # validation

        self.x_train = all_dataset['x_train']
        self.y_train = all_dataset['y_train']

    def compute_loss(self, weights, bias):
        y_pred = tf.reduce_sum(tf.multiply(self.X,weights), axis=1) + bias
        loss = tf.keras.losses.MeanSquaredError()(y_true=self.y, y_pred=y_pred)
        return loss

    def step_loss(self, step_from_zero ):

        new_hyperparam                         = self.init_hyperparameters + step_from_zero * self.hyperparameter_direction
        new_hyperparam                         = np.maximum(new_hyperparam, 1e-10)
        _ , optimal_weights_, optimal_bias_, _ = elastic_net_regression(self.x_train, self.y_train, new_hyperparam[0], new_hyperparam[1])
        loss_at_step                           = self.compute_loss(optimal_weights_, optimal_bias_)
        return loss_at_step

    def interval_search(self, init_t, step):

        a      = init_t
        b      = init_t + step
        loss_a = self.step_loss(a)
        loss_b = self.step_loss(b)
        i      = 0

        while loss_b < loss_a:
            i      += 1
            a      = b
            loss_a = loss_b
            b      = init_t + (2 ** i) * step
            loss_b = self.step_loss(b)
 
        if i == 0:
            return a, b
        elif i == 1:
            return init_t, b
        else:
            return init_t + (2 ** (i - 2)) * step, b

    def golden_section_search(self, delta_1, delta_2, tol):
        
        gr = (1 + 5 ** 0.5) / 2  # Golden ratio

        a = delta_1
        b = delta_2
        # print("Before GSS starts:: ", self.step_loss(a), self.step_loss(b))
        # print(a,b)
        c = b - (b - a) / gr
        d = a + (b - a) / gr

        # print(c,d)
        
        while abs(c - d) > tol:

            loss_c = self.step_loss(c)
            loss_d = self.step_loss(d)

            # print(loss_c, loss_d)

            if loss_c < loss_d:
                b = d
            else:
                a = c

            c = b - (b - a) / gr
            d = a + (b - a) / gr

        loss_a = self.step_loss(a)
        loss_b = self.step_loss(b)

        # print(loss_a, loss_b)

        if loss_a < loss_b: optimal_delta = a
        else: optimal_delta = b

        optimal_hyperparameters               = self.init_hyperparameters + optimal_delta * self.hyperparameter_direction
        optimal_hyperparameters               = np.maximum(optimal_hyperparameters, 1e-10)
        _ , optimal_weights, optimal_bias, _  = elastic_net_regression(self.x_train, self.y_train, optimal_hyperparameters[0], optimal_hyperparameters[1])
        optimal_loss                          = self.compute_loss(optimal_weights, optimal_bias)

        return optimal_hyperparameters, optimal_weights, optimal_bias, optimal_loss, optimal_delta

    def find_optimal_weights(self, init_t=0, step=1e-3, tol=1e-6):
        delta_1, delta_2 = self.interval_search(init_t, step)
        print(f"delta_1 = {delta_1}, delta_2 = {delta_2}")
        optimal_hyperparameters, optimal_weights, optimal_bias, optimal_loss, optimal_delta = self.golden_section_search(delta_1, delta_2, tol)
        return optimal_hyperparameters, optimal_weights, optimal_bias, optimal_loss, optimal_delta

def hls_direction_search(HLS_Tuning_Settings):

    datasets = HLS_Tuning_Settings["Datasets"]
    full_results_dictionary = {}
    hls_final_model = {}
    for sampler_type in HLS_Tuning_Settings["Samplers"]:

        optuna_trials = HLS_Tuning_Settings["Number of Optuna Trials/Epochs"] 
        full_results_dictionary[f"{sampler_type}"] = {}
        hls_final_model["hls_steps"+sampler_type] = []
        hls_final_model["hls_hyperparams"+sampler_type] = [] 
        
        full_results_dictionary[f"{sampler_type}"][f"{optuna_trials}"] = {} #{ "Training_Losses":[], "Validation_Losses":[], "Testing_Losses":[] }
        tr_optuna_losses, tr_hls_losses, val_optuna_losses, val_hls_losses, test_optuna_losses, test_hls_losses = [], [], [], [], [], []
        optuna_hyperparams, hls_hyperparams = [], []
        optuna_times, hls_times = [], []

        # Loading Optuna Trained Models
        total_number_of_simulations = HLS_Tuning_Settings["Number of Simlations"]
        OPTUNA_MODEL_DIRECTORY      = HLS_Tuning_Settings["Optuna Models Path"]
        with open(OPTUNA_MODEL_DIRECTORY, "rb") as fout:
                optuna_trained_models = pkl.load(fout)

        for simulation_number in range(total_number_of_simulations):

            # Loading simulation specific optuna model
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
            GSS_Obj  = GoldenSectionSearch(init_hyperparameters, hyperparameter_direction, datasets)
            
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

###########################################################################################################
                            # FUNCTIONS FOR SOLUTION ANALYSIS AND PLOTS
###########################################################################################################

def convert_to_dataframe(data, sampler_type, optuna_trials):

    # Extract the relevant data
    results = data[f'{sampler_type}'][str(optuna_trials)]
    
    # Initialize a dictionary to hold DataFrame data
    dataframe_dict = {}
    # Iterate through each key in the results
    for key in results.keys():
        # Create sub-columns for each key
        dataframe_dict[f'{key}_optuna'] = results[key][0]
        dataframe_dict[f'{key}_hls'] = results[key][1]
    # Create the DataFrame
    df = pd.DataFrame(dataframe_dict)
    
    return df

# Function to calculate the 95% confidence interval
def calculate_95_ci(data):
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(len(data))
    ci = 1.96 * std_err
    return mean, ci

def hls_result_analysis(result_dictionary, output_path):
      
    columns_to_process = [
        'Training_Losses_optuna', 'Training_Losses_hls',
        'Validation_Losses_optuna', 'Validation_Losses_hls',
        'Testing_Losses_optuna', 'Testing_Losses_hls',
        'Runtimes_optuna', 'Runtimes_hls'
    ]
    final_results_df = { col:{} for col in columns_to_process }

    for sampler_type in HLS_Tuning_Settings["Samplers"]:

        optuna_trials = HLS_Tuning_Settings["Number of Optuna Trials/Epochs"]
        df = convert_to_dataframe(result_dictionary, sampler_type, optuna_trials)

        for column in columns_to_process:
            mean, ci = calculate_95_ci(df[column])
            final_results_df[column][ sampler_type + "_Mean"] = mean
            final_results_df[column][ sampler_type + "_95% CI"] = ci

    final_results_df = pd.DataFrame(final_results_df)
    final_results_df.to_excel( output_path, index=True)

def plot_2d_losses(datasets, optuna_steps_, hls_steps_, width=800, height=600, sampler_type = None, file_name=None, simulation_num_str = '0'):

    x_train = datasets["x_train"]
    x_val   = datasets["x_val"]
    x_test  = datasets["x_test"]

    y_train = datasets["y_train"]
    y_val   = datasets["y_val"]
    y_test  = datasets["y_test"]

    def extract_losses(steps):
        train_losses = []
        val_losses = []
        test_losses = []
        
        for step in steps:
            optimal_weights, optimal_bias = step
            train_loss = evaluate_loss(x_train, y_train, optimal_weights, optimal_bias)
            val_loss = evaluate_loss(x_val, y_val, optimal_weights, optimal_bias)
            test_loss = evaluate_loss(x_test, y_test, optimal_weights, optimal_bias)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            test_losses.append(test_loss)
        
        return train_losses, val_losses, test_losses

    optuna_train_losses, optuna_val_losses, optuna_test_losses = extract_losses(
        optuna_steps_["optuna_steps"+sampler_type][simulation_num_str])
    hls_train_losses, hls_val_losses, hls_test_losses = extract_losses(
        hls_steps_["hls_steps"+sampler_type])
    
    hls_train_losses = hls_train_losses[-1:]
    hls_val_losses = hls_val_losses[-1:]
    hls_test_losses = hls_test_losses[-1:]
    
    optuna_steps = list(range(1, len(optuna_train_losses) + 1))
    hls_steps = list(range(len(optuna_train_losses) + 1, len(optuna_train_losses) + len(hls_train_losses) + 1))
    
    fig = go.Figure()

    marker_size = 14
    line_width  = 4

    # Plot Optuna steps
    fig.add_trace(go.Scatter(
        x=optuna_steps, y=optuna_train_losses,
        mode='lines+markers+text',
        marker=dict(size=marker_size, color='rgba(0, 0, 255, 0.5)'),
        line=dict(color='rgba(0, 0, 255, 0.5)',width = line_width),
        # text=[f'Step {step}' for step in optuna_steps],
        textposition='top center',
        name='Optuna Training Loss'
    ))

    fig.add_trace(go.Scatter(
        x=optuna_steps, y=optuna_val_losses,
        mode='lines+markers+text',
        marker=dict(size=marker_size, color='rgba(0, 255, 0, 0.5)'),
        line=dict(color='rgba(0, 255, 0, 0.5)',width = line_width),
        # text=[f'Step {step}' for step in optuna_steps],
        textposition='top center',
        name='Optuna Validation Loss'
    ))

    fig.add_trace(go.Scatter(
        x=optuna_steps, y=optuna_test_losses,
        mode='lines+markers+text',
        marker=dict(size=marker_size, color='rgba(255, 0, 0, 0.5)'),
        line=dict(color='rgba(255, 0, 0, 0.5)',width = line_width),
        # text=[f'Step {step}' for step in optuna_steps],
        textposition='top center',
        name='Optuna Test Loss'
    ))

    # Plot HLS steps
    fig.add_trace(go.Scatter(
        x=hls_steps, y=hls_train_losses,
        mode='lines+markers+text',
        marker=dict(size=marker_size, color='blue'),
        line=dict(color='blue',width = line_width),
        # text=[f'Step {step}' for step in hls_steps],
        textposition='top center',
        name='HLS Training Loss'
    ))

    fig.add_trace(go.Scatter(
        x=hls_steps, y=hls_val_losses,
        mode='lines+markers+text',
        marker=dict(size=marker_size, color='green'),
        line=dict(color='green',width = line_width),
        # text=[f'Step {step}' for step in hls_steps],
        textposition='top center',
        name='HLS Validation Loss'
    ))

    fig.add_trace(go.Scatter(
        x=hls_steps, y=hls_test_losses,
        mode='lines+markers+text',
        marker=dict(size=marker_size, color='red'),
        line=dict(color='red',width = line_width),
        # text=[f'Step {step}' for step in hls_steps],
        textposition='top center',
        name='HLS Test Loss'
    ))

    # Connect the final point of HLS with the last point of Optuna using a dotted line
    fig.add_trace(go.Scatter(
        x=[optuna_steps[-1], hls_steps[0]], 
        y=[optuna_train_losses[-1], hls_train_losses[0]],
        mode='lines',
        line=dict(color='blue', dash='dot'),
        showlegend=False
    ))

    # Connect the final point of HLS with the last point of Optuna using a dotted line
    fig.add_trace(go.Scatter(
        x=[optuna_steps[-1], hls_steps[0]], 
        y=[optuna_val_losses[-1], hls_val_losses[0]],
        mode='lines',
        line=dict(color='green', dash='dot'),
        showlegend=False
    ))

    # Connect the final point of HLS with the last point of Optuna using a dotted line
    fig.add_trace(go.Scatter(
        x=[optuna_steps[-1], hls_steps[0]], 
        y=[optuna_test_losses[-1], hls_test_losses[0]],
        mode='lines',
        line=dict(color='red', dash='dot'),
        showlegend=False
    ))

    # Setting title name
    plot_title_text = ""
    if sampler_type == 'grid': plot_title_text = "Grid Search"
    elif sampler_type == 'random': plot_title_text = "Random Search"
    elif sampler_type == 'qmc': plot_title_text = "Quasi Monte Carlo Search"
    elif sampler_type == 'tpe': plot_title_text = "Tree-structured Parzen Estimator"
    

    layout_config = {
    "title": {"text": plot_title_text, "x": 0.5, "xanchor": "center"},
    "xaxis_title": 'Step Number',
    "yaxis_title": 'MSE Loss (B = Billion)',
    "width": width,
    "height": height,
    "xaxis": {"visible": False},
    "font": {"size": 16 }, # Adjust the size of text in the plot
    "showlegend": True,
    "legend": { "x":0.01, "y":0.01, "xanchor": "left", "yanchor": "bottom", 
                "bgcolor": "rgba(255, 255, 255, 0.5)", "font": { "size": 12 }}, # Adjust the size of the legend text  
    "paper_bgcolor": 'rgba(0,0,0,0)',  # Remove the paper background
    "plot_bgcolor": 'rgba(0,0,0,0)'
    }

    fig.update_layout(**layout_config)

    if file_name:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        time.sleep(5)
        # fig.write_html(file_name+f"_sim{simulation_num_str}"+".html")
        fig.write_image(file_name+f"_sim{simulation_num_str}"+".pdf", width=width, height=height)

    # fig.show()


if __name__ == '__main__':

    ################################# INPUTS #################################

    OPTUNA_TRIALS  = 100
    SIMULATIONS    = 50

    BASE_DIRECTORY         = "."
    DF_PREPROCESSED_DIR    = BASE_DIRECTORY + "/datasets/House_Price_Prediction/house_price_prediction.pickle" # "/datasets/Simulation_Dataset/simulation_dataset.pickle"  # 
    WORKING_BASE_DIRECTORY = BASE_DIRECTORY + "/outputs"
    HLS_OUTPUT_PATH        = WORKING_BASE_DIRECTORY + f'/results_summary/results_summary_{OPTUNA_TRIALS}_{SIMULATIONS}.xlsx'

    GIVE_LOSS_PLOTS             = True                                                                          # Bool. If yes, choose the simulation for which to plot losses
    SIMULATION_NUMBER_TO_PLOT   = 0                                                                             # Simulation number out of total "SIMULATIONS" for which to plot training, testing and validation loss movements
    PLOTS_OUTPUT_PATH           =  WORKING_BASE_DIRECTORY + f'/plotly_plots'

    # IGNORE IF WORKING_BASE_DIRECTORY SAME AS OPTUNA'S TRAINING. 
    OPTUNA_MODEL_DIRECTORY        = WORKING_BASE_DIRECTORY + f"/all_optuna_models_{SIMULATIONS}_simulations.pickle" 
    OPTUNA_LOSSES_DICTIONARY_PATH = WORKING_BASE_DIRECTORY +"/optuna_steps_dict.pickle"

    ############################################################################

    ###################### OPTUNA #######################
    optuna_model_settings = { "Random Seed"          : None,
                              "Samplers"             : [ 'grid', 'random',  'tpe', 'qmc' ], 
                              "Hyperparameter Range" : { "L1": [1e-10,1], 
                                                         "L2": [1e-10,1] 
                                                        }, 
                              "Number of Optuna Trials/Epochs" : OPTUNA_TRIALS,                     # Total number of optuna trials to run
                              "Number of Simlations"           : SIMULATIONS,                       # Total number of simulations
                              "Dataset Directory Path"         : DF_PREPROCESSED_DIR,               # dataset path
                              "Working Directory Path"         : WORKING_BASE_DIRECTORY             # all models will be saved here
                            }

    setting_up_optuna_training(optuna_model_settings)

    ##################### HLS ############################

    x_train, x_val, x_test, y_train, y_val, y_test = load_dataset(DF_PREPROCESSED_DIR)

    DATASETS = { 'x_train': x_train, 
                "x_val" : x_val, 
                "x_test": x_test, 
                "y_train": y_train, 
                "y_val": y_val, 
                "y_test": y_test
                }

    HLS_Tuning_Settings    = { "Samplers": [ 'random', 'tpe', 'qmc', 'grid' ], 
                                "Number of Optuna Trials/Epochs": OPTUNA_TRIALS, 
                                "Number of Simlations": SIMULATIONS, 
                                "Datasets":DATASETS, 
                                "Dataset Directory Path": DF_PREPROCESSED_DIR, 
                                "Working Directory Path": WORKING_BASE_DIRECTORY,
                                "Optuna Models Path"    : OPTUNA_MODEL_DIRECTORY 
                                }

    full_results_dictionary, hls_final_model = hls_direction_search(HLS_Tuning_Settings)

    # Writing the final model to a pickle file
    with open(WORKING_BASE_DIRECTORY+"/final_hls_model.pickle", 'wb') as file:
        pkl.dump(hls_final_model, file)

    ############################## HLS OUTPUT ANALYSIS ##############################

    hls_result_analysis(full_results_dictionary, HLS_OUTPUT_PATH)

    ######################## PLOTING LOSS STEPS FOR CHOSEN SIMULATION ############################

    with open(OPTUNA_LOSSES_DICTIONARY_PATH, "rb") as fout:
        optuna_steps_dict = pkl.load(fout)

    for sampler_type in HLS_Tuning_Settings["Samplers"]:

        plot_2d_losses(DATASETS, optuna_steps_dict, hls_final_model, width=800, height=600, sampler_type = sampler_type, 
                        file_name = PLOTS_OUTPUT_PATH + F"/{sampler_type}_2Dloss_movement", 
                        simulation_num_str = f'{SIMULATION_NUMBER_TO_PLOT}'
                        )
