import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LogisticRegression():
    def __init__(self,penalty = "l2",gamma = 0,fit_intercept = True):
        err_msg = "penalty must be 'l1' or 'l2', but got: {}".format(penalty)
        assert penalty in ["l2","l1"],err_msg

   
    def initialize_params(self,dims):
        """ Initialize model parameters
        Attributes:
            dims: the number of attribute
        """
        W = np.zeros((dims + 1,1))
        return W

    def sigmoid(self,x):
        """
        The logistic sigmoid function
        """
        z = 1 / (1 + np.exp(-x))
        return z


    def logistic(self,X,y,W,flag,lamada):
        """ The logistic model
        Attributes:
            W: parameters which also contains b

            X: A matrix contains all the samples and their attributes,
            the rows are the number of samples and the columns are the
            number of attributes

            y: The response variable

            flag: choose the method of regularization

            lamada: the parameters for regularization
        """
        num_sample = X.shape[0]
        X = X.T
        X.loc['b'] = 1 # add a row which is all 1

        a = self.sigmoid(np.dot(W.T,X))
        new_a = a.flatten()
        # The cost function (on book)
        
        part1 = np.dot(y,np.log(new_a))
        part2 = np.dot((1-y),np.log(1-new_a))
        
        # Different cost function with regularzation
        if flag == 0:
            
            cost = -1*(part1 + part2)/num_sample
            
            grad = np.zeros(W.shape)
            temp = (new_a - y).ravel()
            for j in range(len(W.ravel())):
                term = np.multiply(temp,X.iloc[j])
                grad[j] = np.sum(term) / num_sample
            dW = grad
        
        
        elif flag == 1: # l1 regularzation
            cost = 0.5*(-1*(part1 + part2)/num_sample) + lamada*np.sum(abs(W))/(2*num_sample)
            
            grad = np.zeros(W.shape)
            temp = (new_a - y).ravel()
            for j in range(len(W.ravel())):
                term = np.multiply(temp,X.iloc[j])
                grad[j] = (np.sum(term) + np.sign(W.flatten()[j])*lamada) / (2*num_sample)
            dW = grad 


        elif flag == 2: # l2 regularzation
            cost = 0.5*(-1*(part1 + part2)/num_sample) + lamada*((np.linalg.norm(W))**2)/(2*num_sample)

            grad = np.zeros(W.shape)
            temp = (new_a - y).ravel()
            for j in range(len(W.ravel())):
                term = np.multiply(temp,X.iloc[j])
                grad[j] = (np.sum(term) + W.flatten()[j]*lamada) / (2*num_sample)
            dW = grad 


        

        return a,cost,dW

    
    def Gradient_descent_train(self,X,y,learning_rate,epochs,flag,lamada,tol = 1e-7,max_iter = 1e7):
        """ Gradient descent model training
        Attributes:
            X,y are the same as in function 'logistic'
            learning_rate: the step size of the Iteration 
            epochs: the iteration times

            lamada and flag: the same as in the logistic function

            tol: a value to stop the iteration

            max_iter: the max iteration and if the iteration tiems
            beyond it, we will stop automatically.(It will work when
            the epoch is to large)

        """
        W = self.initialize_params(X.shape[1])
        cost_list = []

        for i in range(epochs):
            if i > max_iter:
                break
            a,cost,dW = self.logistic(X,y,W,flag,lamada)
            dW_norm = np.linalg.norm(dW) # l2 norm
            if(dW_norm <= tol):  # to find the stopping time
                break
            W = W - learning_rate*dW

            if i % 100 == 0:
                cost_list.append(cost)

            if (i % 100 == 0) & (i != 0):
                pass
                #print('epoch %d cost %f' % (i,cost))

        params = {'W':W} # We get the training parameters

        grads = {'dW':dW}
        
        return cost_list,params,grads

    def predict_test(self,X,params):
        """ With the model we run it on the test set
        Attributes:
            X defined the same as above.
            params: the parameter list we get from Gradient descent model train
        """
        X = X.T
        X.loc['b'] = 1
        y_predict = self.sigmoid(np.dot(params['W'].T,X))
        
        for i in range(X.shape[1]):
            if y_predict[0,i] > 0.5:
                y_predict[0,i] = 1
            else:
                y_predict[0,i] = 0

        return y_predict.T

    def accuracy_metric(self,y_test,y_pred):
        correct_count = 0
        for i in range(len(y_test)):
            if y_test.tolist()[i] == y_pred.flatten()[i]:
                correct_count += 1

        accuracy_score = correct_count/len(y_test)
        return accuracy_score