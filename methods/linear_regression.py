import numpy as np
import sys

class LinearRegression(object):
    """
        Linear regressor object. 
        Note: This class will implement BOTH linear regression and ridge regression.
        Recall that linear regression is just ridge regression with lambda=0.
        Feel free to add more functions to this class if you need.
        But make sure that __init__, set_arguments, fit and predict work correctly.
    """

    def __init__(self, *args, **kwargs):
        """
            Initialize the task_kind (see dummy_methods.py)
            and call set_arguments function of this class.
        """
        ##
        ###
        #### YOUR CODE HERE!
        self.task_kind = 'regression'
        self.set_arguments(*args, **kwargs)
        ###
        ##

    def set_arguments(self, *args, **kwargs):
        """
            args and kwargs are super easy to use! See dummy_methods.py
            In case of ridge regression, you need to define lambda regularizer(lmda).

            You can either pass these as args or kwargs.
        """
        ##
        ###
        #### YOUR CODE HERE!
        if "lmda" in kwargs:
            self.lmda = kwargs["lmda"]
        elif len(args) > 0:
            self.lmda = args[0]
        else:
            self.lmda = 0   # default to linreg
        ###
        ##
    
    def get_w_analytical(self, X_train, Y_train):
        """
            Computes weights for the regression via closed-form solution
        """        
        if self.lmda == 0:
            w = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ Y_train
        else:
            leftmat = np.linalg.inv(X_train.T @ X_train + self.lmda * np.identity(Y_train.shape[1]))
            w = leftmat @ X_train.T @ Y_train

        print(f"X: {X_train.shape}, Y: {Y_train.shape}, w: {w.shape};; {(X_train.T @ X_train).shape}")

        return w

    def append_bias_term(self, X_train):
        """
            Adds a bias term to the end of the data matrix
        """
        ones_column = np.ones([X_train.shape[0], 1])
        X_train_bias = np.concatenate([X_train, ones_column], axis=1)
        return X_train_bias

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): regression target of shape (N,regression_target_size)
            Returns:
                pred_regression_targets (np.array): predicted target of shape (N,regression_target_size)
        """
        ##
        ###
        #### YOUR CODE HERE!
        training_data_bias = self.append_bias_term(training_data)
        self.w = self.get_w_analytical(training_data_bias, training_labels)
        pred_regression_targets = training_data_bias @ self.w
        ###
        ##
        return pred_regression_targets

    def predict(self, test_data):
        """
            Runs prediction on the test data.
            
            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                pred_regression_targets (np.array): predicted targets of shape (N,regression_target_size)
        """   
        ##
        ###
        #### YOUR CODE HERE!
        test_data_bias = self.append_bias_term(test_data)
        pred_regression_targets = test_data_bias @ self.w
        ###
        ##
        return pred_regression_targets
