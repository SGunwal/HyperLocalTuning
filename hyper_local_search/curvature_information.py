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

#########################################################################################################

### ----------- OBJECTIVE COEFFICIENTS FOR DIRECTION LP ------------

def exact_gradient_validation(x_validation, y_validation, w, b, batch_size=None):
    # Convert pandas DataFrame to numpy array
    if not isinstance(x_validation, np.ndarray):
        X = x_validation.values
        y = y_validation.values
    else:
        X = x_validation
        y = y_validation

    y = y.reshape(-1, 1)  # Ensure y is a column vector
    # Number of features
    n_features = X.shape[1]
    # Initialize the gradient for weights and bias
    grad_w = np.zeros((n_features, 1))
    grad_b = 0
    # Number of samples
    n_samples = X.shape[0]
    # Process in batches
    if batch_size is None:
        # Compute residuals
        residuals = y - (np.dot(X, w).reshape(-1,1) + b)
        # Compute gradients
        grad_w = -(2 / n_samples) * np.dot(X.T, residuals)
        grad_b = -(2 / n_samples) * np.sum(residuals)
    else:
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_batch = X[start:end]
            y_batch = y[start:end]
            batch_size_actual = end - start
            # Compute residuals for the batch
            residuals_batch = y_batch - (np.dot(X_batch, w) + b)
            # Update the gradients
            grad_w += -(2 / n_samples) * np.dot(X_batch.T, residuals_batch)
            grad_b += -(2 / n_samples) * np.sum(residuals_batch)
    return np.concatenate([grad_w.flatten(), [grad_b]]) # [w,b]

### ----------- CONSTRAINTS FOR DIRECTION LP ------------

# [\lambda, w, b]
def Grad_L1_term( regularized_weight_vars ): # regularized_weight_vars -> w (not bias b), Regularization_Parameters -> L1,L2
    l1_regularization = tf.reduce_sum(tf.abs(regularized_weight_vars))
    return l1_regularization

def Grad_L2_term( regularized_weight_vars ): # regularized_weight_vars -> w (not bias b), Regularization_Parameters -> L1,L2
    l2_regularization = 0.5 * tf.reduce_sum(tf.square(regularized_weight_vars))
    return l2_regularization

# Reason for using tensorflow: Hypergradient computation for L1 norm
def hessian_part_A( trainable_weights ):

    All_Weights = [ tf.Variable(weight) for weight in trainable_weights ]
    weight_vars = All_Weights[0] # Only weights, not bias, for elastic-net term
    # Gradient
    with tf.GradientTape() as inner_tape:
        grad_l1_regularization_expression = Grad_L1_term( weight_vars )
    hessian_l1_terms = inner_tape.gradient(grad_l1_regularization_expression, All_Weights, unconnected_gradients="zero") # 13 weights, 1 bias

    with tf.GradientTape() as inner_tape:
        grad_l2_regularization_expression = Grad_L2_term( weight_vars )
    hessian_l2_terms = inner_tape.gradient(grad_l2_regularization_expression, All_Weights, unconnected_gradients="zero") # 13 weights, 1 bias

    hessian_l1_terms = np.concatenate((hessian_l1_terms[0].numpy(), hessian_l1_terms[1].numpy())).reshape(-1,1)
    hessian_l2_terms = np.concatenate((hessian_l2_terms[0].numpy(), hessian_l2_terms[1].numpy())).reshape(-1,1)

    hessian_term = np.column_stack((hessian_l1_terms, hessian_l2_terms))#np.concatenate((hessian_l1_terms, hessian_l2_terms), axis=1)

    return hessian_term

def hessian_part_B(X, hyperparams, batch_size = None):

    if not isinstance(X, np.ndarray): X = X.values
    n_features = X.shape[1]
    # Augment X with a column of ones for the bias term
    X_augmented = np.hstack((X, np.ones((X.shape[0], 1))))
    # Initialize the Hessian matrix with the augmented size
    H = np.zeros((n_features + 1, n_features + 1))

    n_samples = X_augmented.shape[0]
    # Process in batches, if batch_size given
    if batch_size is None:
        H = (2 / n_samples) * np.dot(X_augmented.T, X_augmented)
    else:
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_batch = X_augmented[start:end]
            batch_size_actual = end - start
            # Update the Hessian matrix
            H += (2 / n_samples) * np.dot(X_batch.T, X_batch)

    ridge_param = hyperparams[1]
    ridge_term  = ridge_param * np.eye(n_features + 1)
    H += ridge_term

    return H

def constraint_coefficients(df_training, initial_weights, intial_hyperparameters):
    HA = np.array(hessian_part_A( initial_weights )) # 14*2
    HB = hessian_part_B(df_training, intial_hyperparameters)  # 14*14
    return np.hstack((HA, HB)) # 14*16 - [L1,L2,W,b] => (1,1,13,1)

#################### DIRECTION PROBLEM ###########################

def Bilevel_Descent_Direction( GradUObj, Hessian_Follower, number_of_hyperparameters, delta ):

    (Rows_, Columns_) = Hessian_Follower.shape
    # MODEL AND VARIABLE DECLARATION
    m = gp.Model()
    m.setParam( 'OutputFlag', 0 )

    # Objective Function
    Model_Variables = m.addMVar( (Columns_), lb = -100, ub = 100, vtype = 'C' )
    for hindex in range(number_of_hyperparameters):
        Model_Variables[hindex].LB = -1
        Model_Variables[hindex].UB = 1
    # Note: Coefficients are scaled to avoid numerical issues.
    m.setObjective( 1e+4 * (GradUObj @ Model_Variables[number_of_hyperparameters:]), sense = 1 ) # 1 -> Minimize, -1 -> Maximize
    m.addConstr( 1e+5 * (Hessian_Follower @ Model_Variables) <= delta*1e+5 )
    m.addConstr( 1e+5 * (Hessian_Follower @ Model_Variables) >= - delta*1e+5 )
    # OUTPUT HANDLING
    try:
        m.optimize()
        return m.X, m.ObjVal, m.Runtime
    except gp.GurobiError:
        m.computeIIS()
        m.write("IIS_System.ilp")
        return "Error in LB : GurobiError :: ", m.status, 0