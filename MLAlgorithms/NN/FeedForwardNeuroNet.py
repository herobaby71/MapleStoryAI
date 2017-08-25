import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from sklearn.model_selection import train_test_split
from random import shuffle
from ActivationFunctions import ActivationFunctions

class MultiLayerNeuroNet:
    """ A simple implementation of a 3 layer feedforward neuronet for multiclassification
        Inputs: X, y labels, number of neurons per layer, activation function, lambda included
        Optimization use second order optimizer L-BFGS so no need for learning rate
    """
    def __init__(self,X = np.array([]),y = np.array([]),input_layer_size = 1, hidden_layer_size = 1, output_layer_size = 1, lamb = 0, Thetas = None ,activation_function1 = "sigmoid",activation_function2 = 'sigmoid'):
        self.X = X
        self.y = y
        if(X is not None):
            self.X_train,self.X_cv,self.y_train,self.y_cv = train_test_split(X,y,test_size = .15,  random_state=42)
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
        self.activFunc1 = ActivationFunctions(activation_function1)
        self.activFunc2 = ActivationFunctions(activation_function2)
        self.lamb = lamb #regularization
        
        #Xavier Initialization w/ bias
        Theta0 = np.sqrt(2/(input_layer_size+hidden_layer_size))*np.random.randn(hidden_layer_size,input_layer_size+1)
        Theta1 = np.sqrt(2/(hidden_layer_size+output_layer_size))*np.random.randn(output_layer_size,hidden_layer_size+1)
       
        #randomly initialization w/ bias
##        Theta0 = np.random.random((hidden_layer_size,input_layer_size+1))*2-1
##        Theta1 = np.random.random((output_layer_size,hidden_layer_size+1))*2-1
        if(Thetas is None):    
            self.Thetas = np.append(Theta0.ravel(), Theta1.ravel())
        else:
            self.Thetas = Thetas
    def unrollThetaParams(self, Thetas = None):
        if(Thetas is None):
            Thetas = self.Thetas
        Theta0 = np.reshape(Thetas[0:self.hidden_layer_size*(self.input_layer_size+1)],
                            (self.hidden_layer_size,self.input_layer_size+1))
        Theta1 = np.reshape(Thetas[self.hidden_layer_size*(self.input_layer_size+1):],
                            (self.output_layer_size,self.hidden_layer_size+1))
        return (Theta0, Theta1)
    def forwardProp(self, X, Thetas = None):
        if(Thetas is None):
            Thetas = self.Thetas
        m = X.shape[0]
        T0,T1 = self.unrollThetaParams(Thetas)
        
        a0 = np.append(np.ones([m, 1]), X, axis = 1)
        z1 = np.dot(a0, T0.T)
        temp = self.activFunc1.getVal(z1)
        a1 = np.append(np.ones([m, 1]),temp, axis = 1)
        z2 = np.dot(a1, T1.T)
        a2 = self.activFunc2.getVal(z2)
        return a2
    
    def predict(self,X):
        a2 =self.forwardProp(X)
        return np.argmax(a2, axis=1)
    
    def accuracy(self, X, y):
        pred = self.predict(X)
        target = np.argmax(y, axis = 1)
        return np.mean(list(map(int,np.equal(pred,target))))

    #inverted dropout:
        #at train time: <p)/p
    #for dropout regularization, u might want to start with small prob and slowly increase it
        #drop with prob of p
        #at test time, scale down by /p
    #dropoutProb for regularization 
    def CostFunction(self, Thetas):
        X = self.X_train
        y = self.y_train
        m = X.shape[0]
        lamb = self.lamb
        
        #unroll the set of weights
        T0,T1 = self.unrollThetaParams(Thetas)
        
        #Forward prop
        a0 = np.append(np.ones([m,1]), X, axis=1)
        z1 = np.dot(a0,T0.T)
        a1 = np.append(np.ones([m,1]), self.activFunc1.getVal(z1), axis = 1)
        z2 = np.dot(a1,T1.T)
        a2 = self.activFunc2.getVal(z2)
        #Compute Cost/Loss function
        J = 0
        if(self.activFunc2.getName() == "sigmoid"):
            J = (-1/m)*np.sum((y*np.log(a2)) + (1-y)*(np.log(1-a2)))
        elif(self.activFunc2.getName() == "softmax"):
            #J = (-1/m)*np.sum(np.log(a2[np.arange(m), y]))
            J = np.log(y*a2)
            J[np.isneginf(J)] = 0
            J = (-1/m)*np.sum(J)
        J+= (lamb/(2*m))*(np.sum(np.power(T1[:,1:],2)) + np.sum(np.power(T0[:,1:], 2)))
        print("Cost",J)
        
        #Back Propagation to find the gradient
        Delta0,Delta1 = np.zeros(T0.shape), np.zeros(T1.shape)
        
        delta2 = a2-y
        delta1 = np.dot(delta2, T1[:,1:]) * self.activFunc1.getVal(z1, True)

        Grad1 = (1/m)* np.dot(delta2.T,a1)
        Grad1[:,1:] = Grad1[:,1:] + (lamb/m)*T1[:,1:]
        Grad0 = (1/m)*  np.dot(delta1.T,a0)
        Grad0[:,1:] = Grad0[:,1:] + (lamb/m)*T0[:,1:]
        Grad = np.append(Grad0.ravel(),Grad1.ravel())
        
        return (J,Grad)
    def train(self, iterations = 100):
        """ given the regularization parameter lambda, train the model using advance optimization BFGS"""
        arguments = ()
        print("training...")
        results = optimize.minimize(self.CostFunction,x0 = self.Thetas, args = arguments, options = {'disp':True, 'maxiter': iterations}, method = "L-BFGS-B", jac = True)
        self.Thetas = results['x']
        FinalCost, _ = self.CostFunction(self.Thetas)
        print("successfully trained the model")        
        print("Final Cost for this model is:", FinalCost)
    def saveWeights(self):
        np.save("Thetas1.npy",self.Thetas)


