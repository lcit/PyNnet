'''
Neural network class
Leonardo Citraro
'''
from __future__ import division # impose float division
import numpy as np
import scipy as sp
import time
import matplotlib.pyplot as plt
import evaluation
import learning

class nnet(object):
    
    def __init__(   self, size_layers, learning_rate=0.1, batchsize=1, Lambda=0, epochs=100, 
                    activation='sigmoid',out_activation='sigmoid', loss='quadratic', evals=None, 
                    optimization ='sgd', plot_en=False, verbose=True, random_state=None, 
                    denoise_frac=0, dropout_frac=0): 
        
        self.nn_size        = len(size_layers)  # length of the network
        self.size_layers    = size_layers       # size for each layer
        self.learning_rate  = learning_rate     # learning rate (only for Backpropagation)
        self.batchsize      = batchsize         # batch size (only for Backpropagation)
        self.W              = []                # NN weights
        self.Lambda         = Lambda            # L2 regularization term
        self.epochs         = epochs            # number of epochs
        self.loss_train     = []                # training cost for each iteration
        self.loss_val       = []                # validation cost for each iteration
        self.loss           = loss              # type of loss function
        self.evals          = evals             # evaluation scores i.e. ['accuracy', 'log-loss']
        self.optimization   = optimization      # type of learner to use
        self.plot_en        = plot_en           # plot graph
        self.verbose        = verbose           # print infos
        self.random_state   = random_state      # random seed
        self.denoise_frac   = denoise_frac      # denoise percentage
        self.dropout_frac   = dropout_frac      # dropout percentage
        self.activation     = activation        # type of hidden activation function
        self.out_activation = out_activation    # type of output activation function
        
        # check logloss conditions
        if loss == 'log-loss' and out_activation == 'linear':
            raise Exception('The log-loss cannot be used with a linear activation function!')
        
        # set up activation functions and derivatives
        if out_activation == 'sigmoid':
            self._out_activation        = self._sigmoid
            self._out_activation_prime  = self._sigmoid_prime
        if out_activation == 'linear':
            self._out_activation        = self._linear
            self._out_activation_prime  = self._linear_prime
        if out_activation == 'softmax':
            self._out_activation        = self._softmax
            self._out_activation_prime  = self._softmax_prime
            if loss != 'log-likelihood':
                print '++Loss function changed with log-likelihood due to softmax activation.'
                self.loss = 'log-likelihood'  

        # set up activation functions and derivatives
        if out_activation == 'sigmoid':
            self._activation        = self._sigmoid
            self._activation_prime  = self._sigmoid_prime
        if out_activation == 'linear':
            self._activation        = self._linear
            self._activation_prime  = self._linear_prime
        if out_activation == 'softmax':
            self._activation        = self._softmax
            self._activation_prime  = self._softmax_prime
        
        # init NN weights
        np.random.seed(seed=self.random_state)
        for i in range(self.nn_size-1): 
            self.W.append((np.random.rand(size_layers[i]+1,size_layers[i+1]) - 0.5)*\
                            2.0 * 4.0 * np.sqrt(6.0 / (size_layers[i]+size_layers[i+1])))
        
        # dynamic plot variables
        if isinstance(evals, list):                    
            self.lines_train = [np.empty(1)]*len(evals)
            self.lines_val = [np.empty(1)]*len(evals)
              
    def predict(self, X):
        return self.Learner.forward(X)[-1] > 0.5   
    
    def predict_raw(self, X):
        return self.Learner.forward(X)[-1]
        
    def _sigmoid(self, z):
        return sp.special.expit(z)
    
    def _sigmoid_prime(self,z):
        return np.multiply(z, (1.0-z))
        
    def _linear(self, z):
        return z
    
    def _linear_prime(self, z):
        return 1
        
    def _softmax(self, z):
        # -max avoids overflow of the exp
        e = np.exp(z - z.max(1).reshape(-1,1))
        out = e / e.sum(1).reshape(-1,1)
        return out
    
    def _softmax_prime(self, z):
        return 1
     
    def fit(    self, X, y, X_val=np.array([]), y_val=np.array([]), weights=None, 
                eval_weights_train=None, eval_weights_val=None  ):

        self.X = X
        self.y = y
        self.X_val = X_val
        self.y_val = y_val 
        self.J_train = []
        self.J_val = []
        self.score_train = []
        self.score_val = []  
        
        # init evaluators for the losses
        self.Loss_train = evaluation.Evaluator(self.loss)
        self.Loss_val = evaluation.Evaluator(self.loss)
        
        # init evaluators for the evals
        self.Eval_train = evaluation.Evaluator(self.evals, eval_weights_train)
        self.Eval_val = evaluation.Evaluator(self.evals, eval_weights_val)
        
        if self.optimization == 'Backpropagation':
            self.Learner = learning.Backpropagation(     
                        W                       = self.W, 
                        size_layers             = self.size_layers,
                        epochs                  = self.epochs,
                        batchsize               = self.batchsize,
                        activation              = self._activation, 
                        activation_prime        = self._activation_prime, 
                        out_activation          = self._out_activation, 
                        out_activation_prime    = self._out_activation_prime, 
                        Lambda                  = self.Lambda, 
                        dropout_frac            = self.dropout_frac,
                        denoise_frac            = self.denoise_frac,
                        weights                 = weights,
                        learning_rate           = self.learning_rate,
                        random_state            = self.random_state
                                                    )
        if self.optimization == 'iRPROP-':
            self.Learner = learning.iRPROPminus(     
                        W                       = self.W, 
                        size_layers             = self.size_layers,
                        epochs                  = self.epochs,
                        activation              = self._activation, 
                        activation_prime        = self._activation_prime, 
                        out_activation          = self._out_activation, 
                        out_activation_prime    = self._out_activation_prime, 
                        Lambda                  = self.Lambda, 
                        dropout_frac            = self.dropout_frac,
                        denoise_frac            = self.denoise_frac,
                        weights                 = weights,
                        random_state            = self.random_state
                                                    )
        if self.optimization == 'iRPROP+':
            self.Learner = learning.iRPROPplus(     
                        W                       = self.W, 
                        size_layers             = self.size_layers,
                        epochs                  = self.epochs,
                        activation              = self._activation, 
                        activation_prime        = self._activation_prime, 
                        out_activation          = self._out_activation, 
                        out_activation_prime    = self._out_activation_prime, 
                        Lambda                  = self.Lambda, 
                        dropout_frac            = self.dropout_frac,
                        denoise_frac            = self.denoise_frac,
                        weights                 = weights,
                        random_state            = self.random_state
                                                    )
        if 'scipy_minimize_' in self.optimization :
            self.Learner = learning.scipy_minimize(     
                        W                       = self.W, 
                        size_layers             = self.size_layers,
                        epochs                  = self.epochs,
                        activation              = self._activation, 
                        activation_prime        = self._activation_prime, 
                        out_activation          = self._out_activation, 
                        out_activation_prime    = self._out_activation_prime, 
                        Lambda                  = self.Lambda, 
                        dropout_frac            = self.dropout_frac,
                        denoise_frac            = self.denoise_frac,
                        weights                 = weights,
                        random_state            = self.random_state,
                        Loss_train              = self.Loss_train,
                        method                  = self.optimization.split("_")[-1]
                                                    )
        
        self._dynamic_plot(init=True)         
         
        # run optimization
        self.Learner.run(X, y, self._compute_costs_scores)   
        
    def _compute_costs_scores(self, iteration):#params=None):
        ''' Callback function for printing, plotting and computing scores
        '''
        if self.X_val.size!=0: 
            y_hat_val = self.predict_raw(self.X_val)
            self.J_val.append( self.Loss_val.compute_score( self.y_val, y_hat_val ) )
        y_hat_train = self.predict_raw(self.X) 
        self.J_train.append( self.Loss_train.compute_score( self.y, y_hat_train ) )
        
        if self.evals is not None:
            if self.X_val.size!=0:            
                self.score_val.append(  self.Eval_val.compute_score( self.y_val, y_hat_val ) )
            self.score_train.append( self.Eval_val.compute_score( self.y, y_hat_train )  ) 
        
        if self.verbose == True:
            str = 'Epoch %d: '%(iteration) 
            if self.X_val.size!=0:   
                # check if evals is a list
                if isinstance(self.evals, list): 
                    for k in xrange(len(self.evals)):
                        str += '%s train: %0.5f || %s val: %0.5f ||' %(self.evals[k], self.score_train[-1][k], self.evals[k], self.score_val[-1][k]) 
                else:
                    str += '%s train: %0.5f || %s val: %0.5f ||'%(self.evals, self.score_train[-1], self.evals, self.score_val[-1])
            else:
                # check if evals is a list
                if isinstance(self.evals, list): 
                    for k in xrange(len(self.evals)):
                        str += '%s train: %0.5f ||'%(self.evals[k], self.score_train[-1][k])
                else:
                    str += '%s train: %0.5f ||'%(self.evals, self.score_train[-1])
            print('-'*(len(str)+5))
            print(str)

                    
        if self.plot_en:
            self._dynamic_plot()
            
        if isinstance(self.J_train[-1], list):
            return self.J_train[-1][0]
        else:
            return self.J_train
        
    def _dynamic_plot(self, init=False):
        if self.plot_en: 
            if isinstance(self.evals, list):
                if init==True:
                    self.figure, self.axarr = plt.subplots(len(self.evals))
                    if len(self.evals) > 1:
                        for k in xrange(len(self.evals)):
                            self.lines_train[k], = self.axarr[k].plot([],[], label='Train')
                            if self.X_val.size!=0:
                                self.lines_val[k], = self.axarr[k].plot([],[], label='Val')
                            self.axarr[k].set_autoscaley_on(True)
                            self.axarr[k].set_xlim(0, self.epochs)
                            self.axarr[k].grid() 
                            self.axarr[k].set_title(self.evals[k])
                    else:
                        self.lines_train, = self.axarr.plot([],[], label='Train')
                        if self.X_val.size!=0:
                            self.lines_val, = self.axarr.plot([],[], label='Val')
                        self.axarr.set_autoscaley_on(True)
                        self.axarr.set_xlim(0, self.epochs)
                        self.axarr.grid() 
                        self.axarr.set_title(self.evals[0])
                else:
                    if len(self.evals) > 1:
                        for k in xrange(len(self.evals)):
                            score_train = [self.score_train[i][k] for i in xrange(len(self.score_train))]
                            self.lines_train[k].set_xdata(range(len(score_train)))        
                            self.lines_train[k].set_ydata(score_train)
                            if self.X_val.size!=0:
                                score_val = [self.score_val[i][k] for i in xrange(len(self.score_val))]
                                self.lines_val[k].set_xdata(range(len(score_val)))
                                self.lines_val[k].set_ydata(score_val)
                            self.axarr[k].relim()
                            self.axarr[k].autoscale_view()
                            self.axarr[k].legend(loc=4)
                            self.figure.canvas.draw()
                            self.figure.canvas.flush_events()                    
                    else:
                        score_train = [self.score_train[i][0] for i in xrange(len(self.score_train))]
                        self.lines_train.set_xdata(range(len(score_train)))        
                        self.lines_train.set_ydata(score_train)
                        if self.X_val.size!=0:
                            score_val = [self.score_val[i][0] for i in xrange(len(self.score_val))]
                            self.lines_val.set_xdata(range(len(score_val)))
                            self.lines_val.set_ydata(score_val)
                        self.axarr.relim()
                        self.axarr.autoscale_view()
                        self.axarr.legend(loc=4)
                        self.figure.canvas.draw()
                        self.figure.canvas.flush_events()
            else:
                if init==True:
                    self.figure, self.axarr = plt.subplots(1)
                    self.lines_train, = self.axarr.plot([],[], label='Train')
                    if self.X_val.size!=0: 
                        self.lines_val, = self.axarr.plot([],[], label='Val')
                    self.axarr.set_autoscaley_on(True)
                    self.axarr.set_xlim(0, self.epochs)
                    self.axarr.grid() 
                    self.axarr.set_title(self.evals)
                else:
                    self.lines_train.set_xdata(range(len(self.score_train)))        
                    self.lines_train.set_ydata(self.score_train)
                    if self.X_val.size!=0: 
                        self.lines_val.set_xdata(range(len(self.score_val)))
                        self.lines_val.set_ydata(self.score_val)
                    self.axarr.relim()
                    self.axarr.autoscale_view()
                    self.axarr.legend(loc=4)
                    self.figure.canvas.draw()
                    self.figure.canvas.flush_events()
                
        


    def score_sweep_th_max(self, data, labels, th, iscorer):
        ''' get max score by varing the threshold th.
            iscorer() is the function that compute the scores
        '''
        (_,score_max,_) = self.score_sweep_th(data, labels, th, iscorer)          
        return score_max
        
    def score_sweep_th(self, data, labels, th, iscorer):
        ''' compute scores using the function iscorer()
            where th is a vector of thresholds.           
        '''
        y_hat = self.predict(data)[:,-1]
        scores = [];
        for thi in th:
           prediction = float_prediction > thi        	
           scores.append(iscorer(labels[:,-1], prediction))
           
        score_max = np.max(scores)
        optimal_th_idx = np.argmax(scores)
        th_optimal = th[optimal_th_idx]
        return (score_max, th_optimal, scores)
