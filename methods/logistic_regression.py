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

        """ Softmax function
        Args:
            data (np.array): Input data of shape (N, D)
            w (np.array): Weights of shape (D, C) where C is # of classes
            
        Returns:
            res (np.array): Probabilites of shape (N, C), where each value is in 
                range [0, 1] and each row sums to 1.
        """
        
        # nominator = np.exp(np.dot(np.transpose(w),data))
        # res = upper/np.sum()
        x_rows, x_cols = data.shape
        w_rows, w_cols = w.shape

        result = []
        for row in x_rows:
            sum = 0
            for class_w in w_cols:
                sum += np.exp(np.dot(row,class_w))

            x_i_probability = np.zeros((1, w_cols))
            for class_w in w_cols:
                x_i_probability[1,class_w] = (np.exp(np.dot(row,class_w)))/sum
            result.append(x_i_probability)

        y = np.concatenate( result, axis=0 )
        print(y.shape)

        return 0

    def grad_logistic():

        return 0

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
