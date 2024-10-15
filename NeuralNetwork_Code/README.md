# README

This file contains five sections:

- **SEC0**: Contains the details and links for materials.
- **SEC1**: Contains directions to run the models in case the reviewers use the provided Optuna trained models.
- **SEC2**: Contains directions to train the Optuna models and save the pickle file. Afterwards, please follow SEC1 for hyper local tuning.
- **SEC3**: Contains directions for Gurobi license.
- **SEC4**: Provides details of the major libraries used at the time of experiments, along with their versions.

---

## SEC0 => Folder Details

- **Colab_Notebooks**: Contains four Colab notebooks corresponding to Optuna training and hyper local tuning. Details are given in sections below.

- **ResNet50_Optuna_models**: Contains four pickle files for Optuna trained models with ResNet50 setting. The path of these files is required in the `OPTUNA_MODEL_DIRECTORY` variable in the `ResNet50_HLS.ipynb` notebook.

  **Note**: This folder can be accessed from the drive folder [here](https://drive.google.com/drive/folders/16nLo4vAecD6kphYx3Pa5DYE0vciyiZpD?usp=sharing).

- **CIFAR_10_20K**: Contains the data files of the CIFAR-10 dataset. Training, validation, and testing data files provided are the ones used in our experiments.

---

## SEC1: Directions With Optuna Models

There are two models for which notebooks are provided:

- For **Simple CNN Models**: `CNN_HLS.ipynb`
- For **ResNet50 based Models**: `ResNet50_HLS.ipynb`

**NOTE**: In the given notebooks, the specifications are already set to the corresponding models. In the first cell, the libraries will be imported.

The following directions are for both types of models:

1. In the second cell, please provide the directories of the data files, Optuna trained model file, and the full output directory with the Excel (.xlsx) file name.
2. In addition, please specify a Gurobi environment to run the solver. The model is too large for the free version, so at least an academic license is required. 
3. Finally, run all the remaining cells. The output is printed as a log of the final cell and as an Excel file in the output directory.
4. In the case of `CNN_HLS.ipynb`, please mention the number of dense layers of the model (3 or 5).

**NOTE1**: The first row in the output file is the result of Optuna's optimal model t = 0.

**NOTE2**: If you are running this code on a workstation other than Colab Pro Plus, then please note that the runtimes will not match the studies we have provided, as the provided Optuna models are trained on Colab Pro Plus.

---

## SEC2: Directions Without Optuna Models

For both models, notebooks are provided to generate an Optima Optuna model:

- For **Simple CNN Models**: `CNN_OPTUNA.ipynb`
- For **ResNet50 based Models**: `ResNet50_OPTUNA.ipynb`

**NOTE**: In the given notebooks, the specifications are already set to the corresponding models. In the first cell, the libraries will be imported.

The following directions are for both types of models:

1. In the second cell, please provide the directories of the data files and the output directory to save the pickle file in the `OPTUNA_MODEL_DIRECTORY` variable.
2. In addition, please specify the type of sampler (`'grid'`, `'random'`, `'qmc'`, `'tpe'`) and the number of Optuna trials (already set to match our study).
3. Finally, run all the remaining cells. The output is printed as a pickle file in the `OPTUNA_MODEL_DIRECTORY`.
4. In the case of `CNN_HLS.ipynb`, please mention the number of dense layers of the model (3 or 5).
5. After you have the pickle file, please follow the steps in SEC1 for hyper local tuning.

---

## SEC3: GUROBI LICENSE

```python
ENV = gp.Env( empty=True )
ENV.setParam( 'WLSACCESSID', 'xxxxx' )   
ENV.setParam( 'WLSSECRET', 'xxxxx' )    
ENV.setParam( 'LICENSEID', xxxxxx )
ENV.setParam( 'OutputFlag', 0 )      # To Turn-off Logs
ENV.start()
