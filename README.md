# A Linear Programming-based Hyper Local Search for Tuning Hyperparameters

In this paper, we introduce a novel linear programming-based hyper local search approach for hyperparameter tuning in machine learning models. This method focuses on finetuning continuous hyperparameters and model parameters within a localized region around a pre-trained model to improve generalization. By formulating hyperparameter optimization as a bilevel optimization problem, our approach utilizes a linear program to identify a descent direction that minimizes validation loss, refining the model’s performance. The proposed method complements existing sampling techniques like grid search, random search, TPESampler, and QMCSampler, further enhancing model quality. 

Our results consistently demonstrate improvements in validation performance and, in most cases, test performance as well. For example, in Elastic Net Regression on a house price dataset, hyper local search improved the out-of-sample performance by up to 19.6%. In the MNIST classification task, models tuned with hyper local search showed improvements in 67% of cases, with average performance gains of up to 10.9%. On the CIFAR-10 dataset, CNN models saw an average accuracy boost of 5.5% with minimal runtime overhead. These results underscore the effectiveness of hyper local search in fine-tuning models with diverse hyperparameter configurations. Despite challenges in computing the Hessian matrix for large-scale models, approximations provide a viable solution, making this method both efficient and impactful for widespread application.

---

## Directions for Running Elastic Net Regression with Hyper Local Search

**Note**: Current directions are only available for running the code as a project in Visual Studio Code (VSCode). It should work on other editors as well, but the code has been developed specifically in VSCode. We will update a Jupyter notebook with detailed steps later.

### Overview

- The `datasets` folder contains the dataset used for the experiment.
- The `outputs` folder contains the intermediate and final results from running the experiments. The final outputs are stored as an Excel file inside the `results_summary` folder. This file contains the losses from Optuna training and improved losses using hyper local search, for different sampling methods. For each, the mean and 95% confidence interval values are provided. The files in the `outputs` folder are the exact outputs reported in the paper. A backup of these results is stored in `OUTPUT_BACKUPS`.

- The `utils` folder contains helper functions, Python files to install necessary libraries, and a `versions.txt` file that lists all the packages/libraries and their versions. This file is automatically generated when you run `installs.py`.

- The `optuna_training` folder contains a single Python file for training Optuna models for a given setting. Running this file independently will train the models and save the trained model weights, biases, hyperparameters, and runtime information. The final part of this file is commented out to allow the file to be executed independently while also being accessed by the main module for hyper local search: `HLS_one_direction_search.py`.

- The `hyper_local_search` folder contains the main code for tuning the model. The `curvature_information.py` file computes the gradients and necessary Hessian terms. The `loss_movement_plots.py` file plots the improvements in loss functions from Optuna training and local tuning with our method (though this module is currently not in use). The `HLS_one_direction_search.py` file contains the main functions for hyper local direction search, Golden Section Search (GSS), and analysis of the final outputs.

---

### Steps to Set Up and Run the Model

1. Set up the project in an editor such as Visual Studio Code (VSCode).
   
2. Run the files `utils/installs.py` and `utils/imports.py`. These will import the necessary libraries and packages. You can reference `versions.txt` for the specific versions.

3. Go to the main project file `HLS_one_direction_search.py` and modify the parameters as described below:

   3.1. **Set `OPTUNA_TRIALS` and `SIMULATIONS`**: These control the number of epochs in Optuna training and the number of times the experiments are repeated (both for Optuna training and the local tuning phase).

   3.2. **Directories**: The directories are automatically set to the current project folder. Please do not modify these directories.

   3.3. **Optuna Model Settings**: Check the `optuna_model_settings` dictionary, which contains the settings for Optuna training. The `Random_seed` is set to `None` because there are multiple simulations. If you set it to a number, it will choose a fixed sample for each sampler. The `Samplers` key contains the list of samplers to use. The `Hyperparameter Range` contains the range for simulating the L1 and L2 hyperparameters.

   3.4. **Other Parameters**: The rest of the parameters are automatically set. Run this file and check the `Outputs/results_summary/` folder for the final results.


## Appendix

The models were implemented using Python, with neural network models built in TensorFlow 2.12.0. Initial hyperparameter tuning was conducted using the Optuna 3.1.1 library. To solve the bilevel direction problem, we used the Gurobi 10.0.1 commercial solver under an academic license. The linear problem was solved with default solver parameters, where equality constraints were relaxed to inequalities with a tolerance of 1e-4. After determining the bilevel descent direction, a line search was performed, initially narrowing the interval using the bounding phase method, followed by the Golden section search on the resulting interval. Further details about the parameters used in this line search approach are provided below.

**Elastic-Net Regression**  
For optimizing the lower-level problem during OPTUNA's training phase and hyper local search, we used Sklearn's ElasticNet package with a mean squared error loss function. The bounding phase method was used to identify the interval of interest from initial hyperparameter points with a step size of 1e-3. This was followed by Golden section search, which continued until the interval size was reduced to below 1e-6.

**Multi-Layer Perceptron**  
During OPTUNA training, the Adam optimizer was used with a learning rate of 0.001. For models with 1,000 and 5,000 samples, we trained with a batch size of 128 and 100 epochs for each OPTUNA trial. Hyperparameters were drawn from the interval [1e-6, 1e-1]. The number of trials was set to 20, 100, and 200 for the 1HP, 2HP, and 4HP settings, respectively. The line search was performed using the bounding phase method with a step size of 0.1, and the Golden section search was terminated when the interval size dropped below 1e-5.

**CNN and ResNet50 Models**  
During OPTUNA training, the Adam optimizer was used with a learning rate of 0.001 and 100 epochs for the CNN architecture. For the pre-trained ResNet50 model, we set the learning rate to 0.01 and used only 10 epochs to accommodate the computational budget. The hyperparameters suggested by OPTUNA were sampled from the range [1e-6, 1e-1], and the number of trials for both models was set to 10. To compute the Hessian matrix ∇²<sub>(λ<sub>c</sub>, w)</sub> f(λ<sup>0</sup><sub>c</sub>, w<sup>0</sup>; S<sup>T</sup>), we employed SR1 approximation. The line search used a step size of 0.1 and was terminated when the interval size reached 1e-5.

**Computing Platform**  
Experiments on neural network models were conducted on the Google Colab platform with a pro plus subscription. The allocated CPU model was an Intel(R) Xeon(R) CPU @ 2.20GHz with 12 logical processors and 83.5 GB of system RAM. The GPU employed was Nvidia A100 with 40.0 GB of GPU RAM. Regression models were implemented on an AMD Ryzen 5 3550H processor with 8 computing units and 24 GB RAM, without GPU utilization.
