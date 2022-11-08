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
        if "lambda" in kwargs:
            self.lmda = kwargs["lambda"]
        elif len(args) > 0:
            self.lmda = args[0]
        else:
            self.lmda = 0   # default to linreg
        ###
        ##
    

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
        self.D, self.C = training_data.shape[1], training_labels.shape[1]
        print(self.D, self.C)
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
        ###
        ##

        return pred_regression_targets