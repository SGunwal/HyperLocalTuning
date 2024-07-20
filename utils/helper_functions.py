from imports import *

def evaluate_loss(X, y, weights, bias):
        y_pred = tf.reduce_sum(tf.multiply(X,weights), axis=1) + bias
        loss = tf.keras.losses.MeanSquaredError()(y_true=y, y_pred=y_pred)
        return loss

def elastic_net_regression(X, y, lambda_1, lambda_2):

    # Create ElasticNet model
    model = ElasticNet(alpha=lambda_1 + lambda_2, l1_ratio=lambda_1 / (lambda_1 + lambda_2))
    # Fit model
    model.fit(X, y)
    # Get optimal weights (coefficients) and bias (intercept)
    optimal_weights = model.coef_
    optimal_bias = model.intercept_
    # Compute MSE loss
    y_pred = model.predict(X)
    mse_loss = mean_squared_error(y, y_pred)

    return model, optimal_weights, optimal_bias, mse_loss

def unflatten( full_weight_direction, number_of_hyperparameters ): # Converts flattened directions into weight shapes
    hyperparameter_direction = full_weight_direction[:number_of_hyperparameters]
    model_weights_direction  = full_weight_direction[number_of_hyperparameters:-1]
    model_bias_direction     = full_weight_direction[-1]
    return hyperparameter_direction, model_weights_direction , model_bias_direction

def make_writable(arr):
    """Ensure array is writable."""
    arr = np.array(arr, copy=True)  # Ensure it's a numpy array and make a copy to ensure it's writable
    arr.setflags(write=1)           # Explicitly set writable flag to True
    return arr

def load_dataset(DF_PREPROCESSED_DIR):

    # LOAD PREPROCESSED DATA
    with open(DF_PREPROCESSED_DIR, "rb") as fin:
        data = pkl.load(fin)

    # Ensure data is mutable
    x_train = make_writable(data['x_train'])
    x_val = make_writable(data['x_val'])
    x_test = make_writable(data['x_test'])
    y_train = make_writable(data['y_train'])
    y_val = make_writable(data['y_val'])
    y_test = make_writable(data['y_test'])

    return x_train, x_val, x_test, y_train, y_val, y_test


