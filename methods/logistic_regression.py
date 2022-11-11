import numpy as np
import sys
sys.path.append('..')
from utils import label_to_onehot


class LogisticRegression(object):
    """
        LogisticRegression classifier object.
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
        self.task_kind = 'classification'
        self.set_arguments(*args, **kwargs)
        ###
        ##

    def set_arguments(self, *args, **kwargs):
        """
            args and kwargs are super easy to use! See dummy_methods.py
            The LogisticRegression class should have variables defining the learning rate (lr)
            and the number of max iterations (max_iters)
            You can either pass these as args or kwargs.
        """
        
        ##
        ###
        #### YOUR CODE HERE! 
        if "learning_rate" in kwargs:
            self.lr = kwargs["learning_rate"]
        elif len(args) > 0:
            self.lr = args[0]
        else:
            self.lr, self.max_epochs = 0.1, 100

        if "max_epochs" in kwargs:
            self.max_epochs = kwargs["learning_rate"]
        elif len(args) > 1:
            self.max_epochs = args[1]
        ###
        ##
       

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): regression target of shape (N,regression_target_size)
            Returns:
                pred_labels (np.array): target of shape (N,regression_target_size)
        """
        
        
        ##
        ###
        #### YOUR CODE HERE! 
        ###
        ##

        return pred_labels

    def predict(self, test_data):
        """
            Runs prediction on the test data.
            
            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """   
        ##
        ###
        #### YOUR CODE HERE! 
        ###
        ##

        return pred_labels
