import numpy as np

class ActivationFunction:
    def __init__(self, types='Sigmoid'):
        # Sets default
        self.func = self.sigmoid
        self.dfunc = self.dsigmoid

        match types:
            case 'Sigmoid':
                self.func = self.sigmoid
                self.dfunc = self.dsigmoid
            case 'Linear':
                self.func = self.linear
                self.dfunc = self.dlinear
            case 'Softmax':
                self.func = self.softmax
                self.dfunc = self.dsoftmax
            case 'Relu':
                self.func = self.relu
                self.dfunc = self.drelu

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def dsigmoid(self, x):
        sig = self.sigmoid(x)
        return sig * (1-sig)
    
    def linear(self, x):
        return x
    
    def dlinear(self, x):
        return 1
    
    def softmax(self, x):
        expX = np.exp(x - np.max(x, axis=1, keepdims=True))
        return expX / np.sum(expX, axis=1, keepdims=True)
    
    def dsoftmax(self, x):
        raise Exception("TODO: Check this")
        s = self.softmax(x)
        dS = np.empty((x.shape[0], x.shape[1], x.shape[1]))
        for n in range(len(s)):
            diagS = np.diag(s[n])
            dS[n] = diagS - np.outer(s[n], s[n])
        return dS
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def drelu(self, x):
        return np.where(x > 0, 1, 0)
