'''
Neural network: learning classes
Leonardo Citraro
'''
from __future__ import division # impose float division
import numpy as np
import scipy as sp
import time
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import optimize  

class Learner(object):
    
    def __init__( self, W, size_layers, epochs, activation, activation_prime, out_activation, 
                 out_activation_prime, Lambda,  dropout_frac, denoise_frac, weights, random_state ):
        self.nn_size                = len(size_layers)      # NN length
        self.W                      = W                     # NN weights
        self.size_layers            = size_layers           # size for each layer
        self.epochs                 = epochs                # number of iteration
        self.activation             = activation            # hidden activation function
        self.activation_prime       = activation_prime      # derivative hidden activation
        self.out_activation         = out_activation        # output activation function
        self.out_activation_prime   = out_activation_prime  # derivative output activation
        self.dropout_frac           = dropout_frac          # dropout percentage
        self.denoise_frac           = denoise_frac          # denoising percentage
        self.Lambda                 = Lambda                # L2 regularization term
        self.weights                = weights               # sample weights
        self.random_state           = random_state          # random seed
        
        # init gradient vector        
        self.dEdW   = [np.empty(0, dtype='float32')]*(self.nn_size-1)
        
    def forward(self, X):          
        # neurons output value
        self.Z = [np.empty(0, dtype='float32')]*self.nn_size 
        
        bias = np.ones((X.shape[0], 1)) # add bias term
        self.Z[0] = np.append(bias, X, axis=1)
        
        # recompute activations
        for i in range(0, self.nn_size-2):           
            A = self.activation( np.dot(self.Z[i], self.W[i]) )   
            
            if self.dropout_frac > 0: # apply dropout to the activation
                self.dropout_mask[i] = (np.random.rand(A.shape[0], A.shape[1])>self.dropout_frac);
                A = A*self.dropout_mask[i];
            
            bias = np.ones((A.shape[0], 1)) # add bias term
            self.Z[i+1] = np.append(bias, A, axis=1)
            
        self.Z[-1] = self.out_activation(np.dot(self.Z[-2], self.W[-1]))
        
        return self.Z
    
    def compute_dEdW( self, X, y, weights=None):
        
        delta  = [np.empty(0, dtype='float32')]*(self.nn_size-1)
        
        Z = self.forward(X);  

        # delta for the output layer
        if weights is None:
            delta[-1] = ( Z[-1] - y ) * self.out_activation_prime( Z[-1] )     
        else:
            delta[-1] = weights * ( Z[-1] - y ) * self.out_activation_prime( Z[-1] )
        
        # compute hidden deltas
        for i in xrange(self.nn_size-2, 0, -1):             
            delta[i-1] = np.dot(delta[i], self.W[i].transpose()[:,1:]) * self.activation_prime( Z[i][:,1:] ) 

        # compute gradient for each weight and layers
        for i in xrange(0,self.nn_size-1):
            self.dEdW[i] = np.dot( Z[i].transpose(), delta[i] ) / delta[i].shape[0] + self.Lambda * self.W[i];

    def set_W(self, W):
        self.W = W
        
    def get_W(self):
        return self.W

class Backpropagation(Learner):
    '''
        Classic backpropagation algorithm
    '''
    
    def __init__(   self, W, size_layers, batchsize, epochs, activation, activation_prime, 
                    out_activation, out_activation_prime, Lambda, dropout_frac, denoise_frac, 
                    learning_rate, weights, random_state):
                        
        self.learning_rate = learning_rate
        self.batchsize = batchsize
        
        Learner.__init__(   self, W, size_layers, epochs, activation, activation_prime, 
                            out_activation, out_activation_prime, Lambda, dropout_frac,
                            denoise_frac, weights, random_state)
    
    def run( self, X, y, callback):
        
        # calculate number of batches
        self.numbatches = X.shape[0]/int(self.batchsize)            
        if self.numbatches.is_integer()==0:
            # print possible/plausible batchsizes
            d_range = np.arange(1, X.shape[0])
            d = d_range[(X.shape[0]%d_range)==0]
            raise Exception('numbatches is not integer! Suited batch sizes: '+np.array_str(d))
        else:
            self.numbatches = int(self.numbatches)    
        
        # through the epochs
        for e in xrange(self.epochs):            
            # shuffle dataset
            np.random.seed(seed=self.random_state)
            kk = np.random.permutation(X.shape[0])
            for l in range(self.numbatches): 
                # subdivide the dataset in batches
                batch_x = X[kk[l*self.batchsize:(l+1)*self.batchsize], :]
                if self.denoise_frac > 0: # force to zero some inputs
                    batch_x = batch_x*(np.random.rand(batch_x.shape[0], batch_x.shape[1])>self.denoise_frac)
                batch_y = y[kk[l*self.batchsize:(l+1)*self.batchsize]]
                if self.weights is not None:
                    batch_w = self.weights[kk[l*self.batchsize:(l+1)*self.batchsize]]
                    self.compute_dEdW(batch_x, batch_y, batch_w)
                    # apply gradient
                    for j in xrange(self.nn_size-1):
                        self.W[j] = self.W[j] - np.multiply(self.learning_rate, self.dEdW[j])
                else:
                    self.compute_dEdW(batch_x, batch_y)
                    # apply gradient
                    for j in xrange(self.nn_size-1):
                        self.W[j] = self.W[j] - np.multiply(self.learning_rate, self.dEdW[j])          
            callback(e)                                             			
        

class iRPROPminus(Learner):
    
    eta_plus  = 1.2   
    eta_minus = 0.5 
    Delta_max = 50.0 
    Delta_min = 0.0
    Delta_zero = 0.0125  
    
    def __init__(   self, W, size_layers, epochs, activation, activation_prime, out_activation, 
                    out_activation_prime, Lambda, dropout_frac, denoise_frac, weights, 
                    random_state):
        
        Learner.__init__(   self, W, size_layers, epochs, activation, activation_prime, 
                            out_activation, out_activation_prime, Lambda, dropout_frac,
                            denoise_frac, weights, random_state)
        
        # init "current" "old" gradient and Deltas
        self.dEdW_pre   = [np.zeros((size_layers[i]+1, size_layers[i+1]), 
                                    dtype='float32') 
                                     for i in xrange(self.nn_size-1)]
        self.Delta      = [np.ones((size_layers[i]+1, size_layers[i+1]), 
                                   dtype='float32')*self.Delta_zero 
                                    for i in xrange(self.nn_size-1)]
        self.Delta_pre  = [np.ones((size_layers[i]+1, size_layers[i+1]), 
                                   dtype='float32')*self.Delta_zero 
                                   for i in xrange(self.nn_size-1)]
    
    def run( self, X, y, callback):

        # through the epochs
        for e in xrange(self.epochs):
            # compute gradients
            self.compute_dEdW(X, y, self.weights)
            # through the layers
            for i in xrange(self.nn_size-1):
                
                # select specific weights
                sel_pos = (self.dEdW_pre[i] * self.dEdW[i])>0.0
                sel_neg = (self.dEdW_pre[i] * self.dEdW[i])<0.0
                
                # if dEdW[i]*dEdW_pre[i] > 0
                self.Delta[i][sel_pos] = np.minimum(self.Delta_pre[i][sel_pos]*self.eta_plus, self.Delta_max)
                self.Delta[i][sel_neg] = np.maximum(self.Delta_pre[i][sel_neg]*self.eta_minus, self.Delta_min)
                self.dEdW[i][sel_neg] = 0.0
                
                sel_pos = self.dEdW[i]>0.0
                sel_neg = self.dEdW[i]<0.0
                
                # if dEdW[i]*dEdW_pre[i] < 0
                self.W[i][sel_pos] = self.W[i][sel_pos] - self.Delta[i][sel_pos]
                self.W[i][sel_neg] = self.W[i][sel_neg] + self.Delta[i][sel_neg]
                
                # update 
                self.dEdW_pre[i] = self.dEdW[i]
                self.Delta_pre[i] = self.Delta[i]
            callback(e)
            
class iRPROPplus(Learner):
    
    eta_plus  = 1.2   
    eta_minus = 0.5 
    Delta_max = 50.0 
    Delta_min = 0.0
    Delta_zero = 0.0125  
    
    def __init__(   self, W, size_layers, epochs, activation, activation_prime, out_activation, 
                    out_activation_prime, Lambda, dropout_frac, denoise_frac, weights, 
                    random_state):
        
        Learner.__init__(   self, W, size_layers, epochs, activation, activation_prime, 
                            out_activation, out_activation_prime, Lambda, dropout_frac,
                            denoise_frac, weights, random_state)
        
        self.dEdW_pre   = [np.zeros((size_layers[i]+1, size_layers[i+1]), 
                                    dtype='float32') 
                                     for i in xrange(self.nn_size-1)]
        self.Delta      = [np.ones((size_layers[i]+1, size_layers[i+1]), 
                                   dtype='float32')*self.Delta_zero 
                                    for i in xrange(self.nn_size-1)]
        self.Delta_pre  = [np.ones((size_layers[i]+1, size_layers[i+1]), 
                                   dtype='float32')*self.Delta_zero 
                                   for i in xrange(self.nn_size-1)]
        self.Delta_w     = [np.ones((size_layers[i]+1, size_layers[i+1]), 
                                   dtype='float32')*self.Delta_zero 
                                    for i in xrange(self.nn_size-1)]
        self.Delta_w_pre = [np.ones((size_layers[i]+1, size_layers[i+1]), 
                                   dtype='float32')*self.Delta_zero 
                                   for i in xrange(self.nn_size-1)]
        self.E = 0
        self.E_pre = 0
    
    def run( self, X, y, callback):

        # through the epochs
        for e in xrange(self.epochs):
            # compute gradients
            self.compute_dEdW(X, y, self.weights)
            # through the layers
            for i in xrange(self.nn_size-1):
                
                # select weights
                sel_pos = (self.dEdW_pre[i] * self.dEdW[i])>0.0
                sel_neg = (self.dEdW_pre[i] * self.dEdW[i])<0.0
                sel_equ = (self.dEdW_pre[i] * self.dEdW[i])==0.0
                
                # if dEdW[i]*dEdW_pre[i] > 0
                self.Delta[i][sel_pos] = np.minimum(self.Delta_pre[i][sel_pos]*self.eta_plus, self.Delta_max)
                self.Delta_w[i][sel_pos] = -np.sign(self.dEdW[i][sel_pos]) * self.Delta[i][sel_pos]               
                self.W[i][sel_pos] = self.W[i][sel_pos] + self.Delta_w[i][sel_pos]
                
                # if dEdW[i]*dEdW_pre[i] < 0
                self.Delta[i][sel_neg] = np.maximum(self.Delta_pre[i][sel_neg]*self.eta_minus, self.Delta_min)
                if self.E>self.E_pre: 
                    self.W[i][sel_neg] = self.W[i][sel_neg] - self.Delta_w_pre[i][sel_neg]
                self.dEdW[i][sel_neg] = 0.0

                # if dEdW[i]*dEdW_pre[i] == 0
                self.Delta_w[i][sel_equ] = -np.sign(self.dEdW[i][sel_equ]) * self.Delta[i][sel_equ]
                self.W[i][sel_equ] = self.W[i][sel_equ] + self.Delta_w[i][sel_equ]
                                
                # update
                self.dEdW_pre[i] = self.dEdW[i]
                self.Delta_pre[i] = self.Delta[i]
                self.Delta_w_pre[i] = self.Delta_w[i]
                self.E_pre = self.E
            self.E = callback(e)

class scipy_minimize(Learner):   
    
    def __init__(   self, W, size_layers, epochs, activation, activation_prime, out_activation, 
                    out_activation_prime, Lambda, dropout_frac, denoise_frac, weights, 
                    random_state, Loss_train, method):  
        self.Loss_train = Loss_train
        self.method = method
        self.e = 0
        
        Learner.__init__(   self, W, size_layers, epochs, activation, activation_prime, 
                            out_activation, out_activation_prime, Lambda, dropout_frac,
                            denoise_frac, weights, random_state)
    
    def run(self, X, y, callback ):
        self.callback = callback
        optimize.minimize(  self.objective, 
                            x0 = self.get_params(), 
                            jac=True, 
                            method=self.method, 
                            args=(X, y), 
                            options = {'maxiter': int(self.epochs), 'disp' : True}, 
                            callback=self.callback_optimization)

    def get_params(self):
        params = np.array([])
        for w in self.W:
            params = np.hstack((params, w.ravel()))
        return params
    
    def set_params(self, params):
        W = []
        for w in self.W:
            W.append(params[0:w.size].reshape(w.shape))
            params = params[w.size::]
        self.W = W 
        
    def objective(self, params, X, y):
        self.set_params(params)
        
        self.compute_dEdW(X, y)
        
        # get cost
        y_hat = self.Z[-1] # forward called by compute_dEdW     
        cost = self.Loss_train.compute_score(y, y_hat)
        
        # get gradient        
        grad = np.array([])
        for dEdW in self.dEdW:
            grad = np.hstack((grad, dEdW.ravel()))
            
        return cost, grad     
    
    def callback_optimization(self, params):
        self.set_params(params)
        self.e += 1
        self.callback(self.e)
            
            
            
            
            
            
            
            
            