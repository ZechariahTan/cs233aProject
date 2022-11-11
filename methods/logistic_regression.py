import numpy as np
import sys
sys.path.append('..')
from utils import label_to_onehot
np.random.seed(0)

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
        self.task_kind = 'classification'
        self.set_arguments(*args, **kwargs)

    def set_arguments(self, *args, **kwargs):
        """
            args and kwargs are super easy to use! See dummy_methods.py
            The LogisticRegression class should have variables defining the learning rate (lr)
            and the number of max iterations (max_iters)
            You can either pass these as args or kwargs.
        """
        if "lr" in kwargs:
            self.lr = kwargs["lr"]
        elif len(args) >0 :
            self.lr = args[0]
        else:
            self.lr = 0.01
        
        if "max_iters" in kwargs:
            self.max_iters = kwargs["max_iters"]
        elif len(args) >0 :
            self.max_iters = args[0]
        else:
            self.max_iters = 10


    def f_softmax(data,w):
        
        nominator = np.exp(np.dot(np.transpose(w),data))
        res = upper/np.sum()
        return res

    def grad_logistic()

    def fit(self, training_labels, training_data, max_iters, k=4):
        """
            Trains the model, returns predicted labels for training data.
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): regression target of shape (N,regression_target_size)
            Returns:
                pred_labels (np.array): target of shape (N,regression_target_size)
        """
        weights = np.random.normal(0, 0.1, [training_data.shape[1], k])
        for it in range(max_iters):
        
        


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
