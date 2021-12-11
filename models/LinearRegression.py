import math
import numpy as np


class LinearRegression:
    """
    Linear Regression class constructor function

    ---Parameters---

    """

    def __init__(self):
        self.slope = None  # the coefficient of the X also known as the slope (only independent variable)
        self.intercept = None  # y-intercept of the best fit line

    """
    Calculate the total sum of squared residual mean or the total error of the model.
    sum_of_squared_mean = 1/m[sum{(y_acts - y_preds)^2}]
    
    ---Parameters---
    y_acts: The actual target variable
    y_preds: The predicted target
    
    ---Returns---
    
    Returns the total error

    """

    @staticmethod
    def get_sum_of_squared_mean_residuals(y_acts, y_preds):
        sum_of_squared = 0
        residuals = y_acts - y_preds
        for residual in residuals:
            residual = math.pow(residual, 2)
            sum_of_squared = sum_of_squared + residual

        sum_of_squared_mean = sum_of_squared / y_acts.shape[0]
        return sum_of_squared_mean

    """
    Calculate the model parameters b0(y-intercept),b1(slope) using gradient descent
    
    1/m[sum{(y_acts - (b1*x + b0)^2}] = 0
    b0 = y_mean - b1 * x_mean
    b1 = sum[(xi - x_mean)*(yi-y_mean)]/sum[(xi - x_mean)^2]
    
    ---Parameters---
    X: independent variable values
    Y: dependent variable values
    
    ---Returns---
    Returns model paramters bo,b1
    
    """

    @staticmethod
    def get_b0_b1(X, Y):

        x_mean = X.mean()
        y_mean = Y.mean()

        sum_of_products = 0
        sxx = 0
        for x, y in zip(X, Y):
            sum_of_products = sum_of_products + (x - x_mean) * (y - y_mean)
            residual = x - x_mean
            sxx = sxx + math.pow(residual, 2)

        b1 = sum_of_products / sxx
        b0 = y_mean - (b1 * x_mean)
        return b0, b1

    """
    The training function.
    
    ---Parameters---
    X: independent variable values
    Y: dependent variable values
    
    ---Returns---
    """

    def fit_and_transform(self, X, Y):
        b0, b1 = self.get_b0_b1(X, Y)
        self.slope = b1  #
        self.intercept = b0

        print(f"Coefficient 0 B0: {self.intercept}, Coefficient 1 B1: {self.slope}")
        print(f"Loss J(X,Y,B): {self.get_sum_of_squared_mean_residuals(Y, self.get_predictions(X))}")

    """
    Get the model predictions.
    y_pred = x*self.slope + self.intercept
    
    ---Parameters--
    x: array of independent values
    
    ---Returns---
    array of predictions
    """

    def get_predictions(self, x):
        y_pred = self.slope * x
        y_pred = y_pred + self.intercept
        return y_pred
