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

