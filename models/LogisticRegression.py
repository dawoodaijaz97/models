import numpy as np
import math
from sklearn.metrics import accuracy_score


class LogisticRegression:
    """
    Logistic Regression class constructor function

    ---Parameters---

    n_iter: Maximum number of iteration to train the model
    learning_rate: Maximum Change in the model components Q at each step


    """

    def __init__(self, n_iter, learning_rate, thresh, fit_intercept=True):
        self.n_iter = n_iter  # number of iterations
        self.learning_rate = learning_rate  # learning rate per iteration
        self.thresh = thresh  # threshold for hypothesis
        self.is_single_dimension = False  # is  the X vector single dimension
        self.coefficients = None  # value of model parameter Q1,Q2, ...
        self.theta_vals = None  # value of model parameters through out training
        self.fit_intercept = fit_intercept  # fit intercept or not
        self.coeff_ = None  # best value of coefficients
        self.min_cost_ = 10000000  # min value of cost
        print("Initializing Logistic Regression")

    """
    Implement the hypothesis function h(x) = g(QX) = P(y=1|x;Q)
    Gives the probability of y=1 given X=x and parameters Q

    ----Parameters----
    x: the dependent variable
    Q: the logistic parameters

    """

    def __hypothesis(self, Xi, Q):
        z = 0
        if self.is_single_dimension:
            return self.__sigmoid(Xi * Q[0])

        for Xij, Qj in zip(Xi, Q):
            z = z + (Xij * Qj)
        return self.__sigmoid(z)

    """

    Implement the sigmoid function. The Sigmoid
    function is used as it restricts the range of P
    between 0 and 1

    ---Parameters---
    z: The sum of products of coefficients and independents

    ---Returns---
    Returns the sigmoid value for a given value z

    """

    def __sigmoid(self, z):
        h = 1 / (1 + math.exp(-z))
        if h == 1:
            h = 0.9999
        elif h == 0:
            h = 0.00001
        return h

    """
    Calculate the cost for the entire dataset for given model parameters Q.
    Cost of each data point will be calculated and summed up
    The cost function is a piecewise function
    for y = 0: cost = -log(h(x))
    for y = 1: cost = -log(1-h(x))

    ---Parameters---
    X: Independent features
    Y: Dependent features(target)
    Q: Model Parameter

    ---Returns---
    Returns the total cost for the dataset.
    """

    def __cost_function(self, X, Y):
        try:
            total_cost = 0
            for x, y in zip(X, Y):
                h = self.__hypothesis(x, self.coefficients)
                if y == 1:
                    total_cost = total_cost + -math.log(h)
                else:
                    total_cost = total_cost + -math.log((1 - h))
            const = 1 / X.shape[0]
            total_cost = const * total_cost
            return total_cost
        except ValueError:
            print(f"Except occured at {h},{x},{y},{self.coefficients}")

    """
    Calculate the derivative for a J(Qi) by looping over the dataset.
    J'(Qi) = 1/m[Sum{(hxi - yi)*xij}]
    """

    def __cost_function_derivative(self, X, Y, j):

        error = 0
        for xi, yi in zip(X, Y):
            if self.is_single_dimension:
                xij = xi
            else:
                xij = xi[j]
            hxi = self.__hypothesis(xi, self.coefficients)  # Calculate hypothesis
            error = error + (hxi - yi) * xij
        const = 1 / X.shape[0]
        derivative = error * const
        return derivative[0]

    """
    Get the model predictions for X
    """

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            hypothesis = self.__hypothesis(X[i], self.coeff_)
            if hypothesis >= self.thresh:
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions

    """
    The fit_and_transform function is used to train the logistic  regression mode.
    The function perform the training iterations and at each iteration get the derivative of 
    J(Qi) and updates the value by learning rate. Qj = Qj - (learning_rate * __cost_function_derivative(X, Y,j))
    The function also maintains the theta values,and associated cost for each iteration.
    
    ---Parameters---
    
    X: training independent features
    Y: training dependent feature (or the targets)
    
    ---Returns---
    Null
    
    """
    def fit_and_transform(self, X, Y):
        print(X.shape)
        print(Y.shape)
        if self.fit_intercept: # if fit_intercept check is true. Add additional columns X[l] in X, X[l] = 1,
            intercept = np.ones(X.shape[0]) # fill new columns with ones
            print(intercept.shape)
            X = np.stack((X, intercept), 1) # append column to training dataset.
            self.is_single_dimension = False # make is_single_dimension false as the data set will not be single dimension when intercept column is added

        if X.shape.__len__() == 1:
            self.is_single_dimension = True # if X columns =1, set is_single_dimension to True.
            self.coefficients = [0] * 1 # set coefficient array
            self.theta_vals = [[0] * 2 for _ in range((self.n_iter + 1))]
        else:
            self.coefficients = [0] * X.shape[1] # if not single dimension. Set coefficient array
            self.theta_vals = [[0] * (X.shape[1] + 1) for _ in range((self.n_iter + 1))]
        cost = 0
        for i in range(self.n_iter): # training iterations.
            cost = self.__cost_function(X, Y)
            self.theta_vals[i][-1] = cost
            for j in range(self.coefficients.__len__()): # iterate over each parameter
                derivative = self.__cost_function_derivative(X, Y, j) # calculate derivative for parameter Qj
                self.theta_vals[i][j] = self.coefficients[j]
                self.coefficients[j] = self.coefficients[j] - (self.learning_rate * derivative)

            cost = self.__cost_function(X, Y)
            if cost < self.min_cost_:
                self.coeff_ = self.coefficients
                self.min_cost_ = cost
            predictions = self.predict(X)
            accuracy = accuracy_score(Y, predictions)
            print(f"Iteration# {i}, Error {cost}, Accuracy {accuracy}, {self.coefficients}")

        if not self.is_single_dimension:
            for j in range(X.shape[1]):
                self.theta_vals[self.n_iter][j] = self.coefficients[j]
                self.theta_vals[self.n_iter][-1] = cost
        else:
            self.theta_vals[self.n_iter][0] = self.coefficients[0]
            self.theta_vals[self.n_iter][1] = cost

    def get_hypothesis(self, X):
        probabilities = []
        for i in range(X.shape[0]):
            hypothesis = self.__hypothesis(X[i], self.coeff_)
            probabilities.append(hypothesis)
        return probabilities
