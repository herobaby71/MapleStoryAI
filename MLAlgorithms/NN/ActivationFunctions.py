import numpy as np
from scipy.special import expit
class ActivationFunctions:
    def __init__(self,name):
        self.name = name
        self.cache = None
        
    def getName(self):
        return self.name
    
    def getVal(self, z, gradient = False):
        return getattr(self, self.name)(z, gradient)

    #sigmoid
    def sigmoid(self, z, gradient = False):
        if(gradient):
            return self.sigmoid(z)*(1-self.sigmoid(z))
        #return 1/(1+np.exp(-z))
        self.cache = z
        return expit(z)

    #for derivative, if x > 0, return 1, O/W .01 () or 0
    def ReLU(self, z, gradient = False):
        if(gradient):
            dz = np.array(z, copy = True)
            dz[self.cache <=0] = 0
            return dz
        self.cache = z
        return np.maximum(0,z)
    

    #Leaky ReLU, to fix the dying ReLU problems
    def LReLU(self, z, cache = None, gradient = False):
        pass

    #for the output layer to compute probability
    def softmax(self, z, gradient = False):
        if(gradient):
            return self.softmax(z)*(1-self.softmax(z))
        #subtract the maximum value to create negative value, which works better for exponents
        probs = np.exp(z - np.max(z, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        return probs

