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
sumlation_number_for_plot = 10

DF_PREPROCESSED_DIR = "./datasets/House_Price_Prediction/house_price_prediction.pickle"
x_train, x_val, x_test, y_train, y_val, y_test = load_dataset(DF_PREPROCESSED_DIR)


def plot_3d_losses(optuna_steps_, hls_steps_, show_vertical_lines=False, width=800, height=600, 
                   sampler_ = None, html_file=None, simulation_num = 0
                   ):
    
    def extract_losses(steps, hyperparams):

        print("\n\n", hyperparams, "\n\n")
        hyper_l1 = [params[0] for params in hyperparams]
        hyper_l2 = [params[1] for params in hyperparams]
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
        
        return hyper_l1, hyper_l2, train_losses, val_losses, test_losses

    optuna_hyper_l1, optuna_hyper_l2, optuna_train_losses, optuna_val_losses, optuna_test_losses = extract_losses(
        optuna_steps_["optuna_steps"+sampler_][f'{simulation_num}'], optuna_steps_["optuna_hyperparams"+sampler_][f'{simulation_num}'])
    hls_hyper_l1, hls_hyper_l2, hls_train_losses, hls_val_losses, hls_test_losses = extract_losses(
        hls_steps_["hls_steps"+sampler_], hls_steps_["hls_hyperparams"+sampler_])
    
    # only take final values for HLS
    hls_train_losses = hls_train_losses[-1:]
    hls_val_losses = hls_val_losses[-1:]
    hls_test_losses = hls_test_losses[-1:]
    hls_hyper_l1 = hls_hyper_l1[-1:]
    hls_hyper_l2 = hls_hyper_l2[-1:]
    
    optuna_steps = list(range(1, len(optuna_hyper_l1) + 1))
    hls_steps = list(range(len(optuna_hyper_l1) + 1, len(optuna_hyper_l1) + len(hls_hyper_l1) + 1))
    
    fig = go.Figure()

    # Plot Optuna steps
    fig.add_trace(go.Scatter3d(
        x=optuna_hyper_l1, y=optuna_hyper_l2, z=optuna_train_losses,
        mode='lines+markers+text',
        marker=dict(size=5, color='rgba(0, 0, 255, 0.5)'),
        line=dict(color='rgba(0, 0, 255, 0.5)'),
        # text=[f'Step {step}' for step in optuna_steps],
        textposition='top center',
        name='Optuna Training Loss'
    ))

    fig.add_trace(go.Scatter3d(
        x=optuna_hyper_l1, y=optuna_hyper_l2, z=optuna_val_losses,
        mode='lines+markers+text',
        marker=dict(size=5, color='rgba(0, 255, 0, 0.5)'),
        line=dict(color='rgba(0, 255, 0, 0.5)'),
        # text=[f'Step {step}' for step in optuna_steps],
        textposition='top center',
        name='Optuna Validation Loss'
    ))

    fig.add_trace(go.Scatter3d(
        x=optuna_hyper_l1, y=optuna_hyper_l2, z=optuna_test_losses,
        mode='lines+markers+text',
        marker=dict(size=5, color='rgba(255, 0, 0, 0.5)'),
        line=dict(color='rgba(255, 0, 0, 0.5)'),
        # text=[f'Step {step}' for step in optuna_steps],
        textposition='top center',
        name='Optuna Test Loss'
    ))

    # Plot HLS steps
    fig.add_trace(go.Scatter3d(
        x=hls_hyper_l1, y=hls_hyper_l2, z=hls_train_losses,
        mode='lines+markers+text',
        marker=dict(size=5, color='blue'),
        line=dict(color='blue'),
        # text=[f'Step {step}' for step in hls_steps],
        textposition='top center',
        name='HLS Training Loss'
    ))

    fig.add_trace(go.Scatter3d(
        x=hls_hyper_l1, y=hls_hyper_l2, z=hls_val_losses,
        mode='lines+markers+text',
        marker=dict(size=5, color='green'),
        line=dict(color='green'),
        # text=[f'Step {step}' for step in hls_steps],
        textposition='top center',
        name='HLS Validation Loss'
    ))

    fig.add_trace(go.Scatter3d(
        x=hls_hyper_l1, y=hls_hyper_l2, z=hls_test_losses,
        mode='lines+markers+text',
        marker=dict(size=5, color='red'),
        line=dict(color='red'),
        # text=[f'Step {step}' for step in hls_steps],
        textposition='top center',
        name='HLS Test Loss'
    ))

    steps_ = [f'OPT {step}' for step in (optuna_steps + hls_steps)]
    steps_[-1] = "HLS"

    # Add base points for L1 and L2 with step text
    fig.add_trace(go.Scatter3d(
        x=optuna_hyper_l1 + hls_hyper_l1, y=optuna_hyper_l2 + hls_hyper_l2, z=[0]*(len(optuna_steps) + len(hls_steps)),
        mode='markers+text',
        marker=dict(size=5, color='black'),
        text=steps_,
        textposition='top center',
        name='L1-L2 Base Points'
    ))

    # Add vertical lines from base to training loss points
    if show_vertical_lines:
        for i in range(len(optuna_steps)):
            fig.add_trace(go.Scatter3d(
                x=[optuna_hyper_l1[i], optuna_hyper_l1[i]],
                y=[optuna_hyper_l2[i], optuna_hyper_l2[i]],
                z=[0, optuna_train_losses[i]],
                mode='lines',
                line=dict(color='rgba(0, 0, 255, 0.5)', dash='dash'),
                showlegend=False
            ))
        for i in range(len(hls_steps)):
            fig.add_trace(go.Scatter3d(
                x=[hls_hyper_l1[i], hls_hyper_l1[i]],
                y=[hls_hyper_l2[i], hls_hyper_l2[i]],
                z=[0, hls_train_losses[i]],
                mode='lines',
                line=dict(color='blue', dash='dash'),
                showlegend=False
            ))

    # Connect the final Optuna points with the first HLS points
    fig.add_trace(go.Scatter3d(
        x=[optuna_hyper_l1[-1], hls_hyper_l1[0]], 
        y=[optuna_hyper_l2[-1], hls_hyper_l2[0]], 
        z=[optuna_train_losses[-1], hls_train_losses[0]], 
        mode='lines',
        line=dict(color='rgba(0, 0, 255, 0.5)', dash='dot'),
        showlegend=False
    ))

    fig.add_trace(go.Scatter3d(
        x=[optuna_hyper_l1[-1], hls_hyper_l1[0]], 
        y=[optuna_hyper_l2[-1], hls_hyper_l2[0]], 
        z=[optuna_val_losses[-1], hls_val_losses[0]], 
        mode='lines',
        line=dict(color='rgba(0, 255, 0, 0.5)', dash='dot'),
        showlegend=False
    ))

    fig.add_trace(go.Scatter3d(
        x=[optuna_hyper_l1[-1], hls_hyper_l1[0]], 
        y=[optuna_hyper_l2[-1], hls_hyper_l2[0]], 
        z=[optuna_test_losses[-1], hls_test_losses[0]], 
        mode='lines',
        line=dict(color='rgba(255, 0, 0, 0.5)', dash='dot'),
        showlegend=False
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='L1 Hyperparameter',
            yaxis_title='L2 Hyperparameter',
            zaxis_title='Loss'
        ),
        title='3D Plot of Losses vs. Hyperparameters',
        width=width,
        height=height
    )

    if html_file:
        fig.write_html(html_file+".html")
        fig.write_image(html_file+".pdf", width=1500, height=1000, scale=2)

    # fig.show()

def plot_2d_losses(optuna_steps_, hls_steps_, width=800, height=600, sampler_type = None, html_file=None, simulation_num_str = '0'):

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

    if html_file:

        # Ensure the directory exists
        os.makedirs(os.path.dirname(html_file), exist_ok=True)

        # fig.write_html(html_file+f"_sim{simulation_num_str}"+".html")
        fig.write_image(html_file+f"_sim{simulation_num_str}"+".pdf", width=width, height=height)

    # fig.show()

if __name__ == '__main__':

    # SETTING DIRECTORY
    WORKING_BASE_DIRECTORY = "./outputs"
    #############################################
    list_of_all_samplers    = ['grid', 'random', 'qmc', 'tpe']
    # Loading the losses from optuna trainings
    with open(WORKING_BASE_DIRECTORY+"/optuna_steps_dict.pickle", "rb") as fout:
        optuna_steps_dict = pkl.load(fout)
    with open(WORKING_BASE_DIRECTORY+"/final_hls_model.pickle", "rb") as fout:
        hls_step_dict = pkl.load(fout)


    # PLOTTING
    for sampler_type in list_of_all_samplers:
        # Printing the plots
        # plot_3d_losses(optuna_steps_dict, hls_step_dict, show_vertical_lines=True, width=2000, height=1500, sampler_ = sampler_type, html_file= WORKING_BASE_DIRECTORY + f'/demo_plots/{sampler_type}_3Dloss_movement')
        plot_2d_losses(optuna_steps_dict, hls_step_dict, width=800, height=600, sampler_type = sampler_type, 
                       html_file= WORKING_BASE_DIRECTORY + f'/plotly_plots/{sampler_type}_2Dloss_movement', simulation_num_str = f'{sumlation_number_for_plot}')