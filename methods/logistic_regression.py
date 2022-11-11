import numpy as np
import sys
sys.path.append('..')
from utils import label_to_onehot
from utils import onehot_to_label
from metrics import accuracy_fn
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
        elif len(args) > 0 :
            self.lr = args[0]
        else:
            self.lr = 0.01
        
        if "max_iters" in kwargs:
            self.max_iters = kwargs["max_iters"]
        elif len(args) > 1 :
            self.max_iters = args[1]
        else:
            self.max_iters = 10


    def f_softmax(self, data, w):

        """ Softmax function
        Args:
            data (np.array): Input data of shape (N, D)
            w (np.array): Weights of shape (D, C) where C is # of classes
            
        Returns:
            res (np.array): Probabilites of shape (N, C), where each value is in 
                range [0, 1] and each row sums to 1.
        """
        
        x_rows, x_cols = data.shape
        w_rows, w_cols = w.shape

        result = []
        for row in range(x_rows):
            sum = 0
            for class_w in range(w_cols):
                sum += np.exp(np.dot(data[row],w[:, class_w]))

            x_i_probability = np.zeros((1, w_cols))
            for class_w in range(w_cols):
                x_i_probability[0,class_w] = (np.exp(np.dot(data[row],w[:, class_w])))/sum
            result.append(x_i_probability)
            print(row)

        y = np.concatenate( result, axis=0)

        return y

    def loss_logistic_multi(self, data, labels, softmax_matrix):
        """ Loss function for multi class logistic regression
        
        Args:
            data (np.array): Input data of shape (N, D)
            labels (np.array): Labels of shape  (N, C) (in one-hot representation)
            w (np.array): Weights of shape (D, C)
            
        Returns:
            float: Loss value 
        """

        num_samples, dimensions = data.shape
        num_samples, num_classes = labels.shape

        loss = 0
        for sample in range(num_samples):
            for c in range(num_classes):
                loss += int(labels[sample][c]) * np.log(softmax_matrix[sample][int(labels[sample][c])])
        
        loss = (-1)*loss
        return loss
    
    def grad_logistic_multi(self, data, labels, softmax_matrix):
        """ Gradient function for multi class logistic regression
        Args:
            data (np.array): Input data of shape (N, D)
            labels (np.array): Labels of shape  (N, )
            w (np.array): Weights of shape (D, C)
        Returns:
            grad_w (np.array): Gradients of shape (D, C)
        """

        grad = data.T @ (softmax_matrix - labels)
        return grad

    def logistic_regression_classify_multi(self, data, w):
        """ Classification function for multi class logistic regression. 
        Args:
            data (np.array): Dataset of shape (N, D).
            w (np.array): Weights of logistic regression model of shape (D, C)
        Returns:
            np. array: Label assignments of data of shape (N, ).
        """
        predictions = self.f_softmax(data, w)
        predictions = np.argmax(predictions, axis=1)
        return predictions

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): regression target of shape (N,regression_target_size)
            Returns:
                pred_labels (np.array): target of shape (N,regression_target_size)
        """
        weights = np.random.normal(0, 0.1, [training_data.shape[1], 1])
        one_hot_labels = label_to_onehot(training_labels)
        softmax_matrix = self.f_softmax(training_data, weights)

        print_period = 1
       
        for it in range(self.max_iters):
            gradient = self.grad_logistic_multi(training_data, one_hot_labels, softmax_matrix)
            weights = weights - self.lr*onehot_to_label(gradient)
            predictions = self.logistic_regression_classify_multi(training_data, weights)
            if accuracy_fn(training_labels, predictions) == 1:
                break
            #logging and plotting
            if print_period and it % print_period == 0:
                print('loss at iteration', it, ":", self.loss_logistic_multi(training_data, one_hot_labels, softmax_matrix))
                self.weights = weights
        return weights

    def predict(self, test_data):
        """
            Runs prediction on the test data.
            
            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """   
        predictions = self.logistic_regression_classify_multi(test_data, self.weights)

        return predictions
