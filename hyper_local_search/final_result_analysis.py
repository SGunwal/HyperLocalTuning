import pandas as pd
import numpy as np
import pickle as pkl

def convert_to_dataframe(data, sampler_type, optuna_trials = 100):

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


if __name__ == '__main__':

    # FINAL RESULTS OUTPUT DIRECTORY
    output_directory = "C:\\Users\\Sandeep\\Desktop\\Github-SGunwal\\HyperLocalTuning\\outputs\\HLS_Results\\"

    # LOAD PREPROCESSED DATA
    full_result_path = output_directory + "final_hls_output.pickle"
    with open(full_result_path, "rb") as fin:
        data = pkl.load(fin)

    # SEPARATE SUMMARY FILE FOR EACH MODEL
    SAMPLER_TYPES = ['grid', 'random', 'qmc', 'tpe']
    OPTUNA_TRIALS = [100]

    for sampler_type in SAMPLER_TYPES:
        for trial_type in OPTUNA_TRIALS:

            df = convert_to_dataframe(data, sampler_type, trial_type)
            # Columns to process for before and after
            columns_to_process = [
                'Training_Losses_optuna', 'Training_Losses_hls',
                'Validation_Losses_optuna', 'Validation_Losses_hls',
                'Testing_Losses_optuna', 'Testing_Losses_hls',
                'Runtimes_optuna', 'Runtimes_hls'
            ]
            # Initialize a dictionary to hold the summary statistics
            summary_stats = {}
            # Calculate the mean and 95% CI for each column
            for column in columns_to_process:
                mean, ci = calculate_95_ci(df[column])
                summary_stats[column] = {'Mean': mean, '95% CI': ci}

            # Convert summary statistics to a DataFrame
            summary_df = pd.DataFrame(summary_stats).T

            file_name = "summary_results_" + sampler_type
            summary_df.to_excel( output_directory + file_name + f'{trial_type}.xlsx', index=True)
