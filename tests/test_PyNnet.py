from __future__ import division # impose float division
import csv as csv 
import numpy as np
import os
import sys
root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root)
import PyNnet
import time
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split 

#==============================================================================
# Read dataset (csv)
#==============================================================================
# read training dataset
current_directory = os.getcwd()
csv_training = csv.reader(open(current_directory + '\\training.csv', 'rb')) 
training = []
for row in csv_training:
    training.append(row)        
training = np.array(training, dtype=float)

# read test dataset
csv_test = csv.reader(open(current_directory + '\\test.csv', 'rb')) 
test = []
for row in csv_test:
    test.append(row)        
test = np.array(test, dtype=float)

print '+Read csv done!'


#==============================================================================
# Get a bigger training datset
#==============================================================================
training_c = training[:,[0,1]]
labels_c = training[:,[2]]
for i in xrange(20):
    training_c = np.concatenate((training_c, training[:,[0,1]]), axis=0)
    labels_c = np.concatenate((labels_c, training[:,[2]]), axis=0)
training_cc = training_c
for i in xrange(10):
    training_cc = np.concatenate((training_cc, training_c), axis=1)

#==============================================================================
# Normalization and split
#==============================================================================
scaler          = preprocessing.StandardScaler().fit(training_cc)
training_n      = scaler.transform(training_cc)
X_tr_n, X_cv_n, y_tr, y_cv = train_test_split(training_cc, labels_c, test_size=0.5, random_state=0)

print '+Normalization and split done!'

#==============================================================================
# Train Neural network
#==============================================================================

start_time = time.time()

NN = PyNnet.nnet(    size_layers    = [X_tr_n.shape[1],20,20,1], 
                    learning_rate   = 0.01,
                    batchsize       = 105,
                    Lambda          = 0, 
                    epochs          = 500, 
                    out_activation  = 'sigmoid', 
                    loss            = 'log-loss', 
                    #loss            = 'quadratic-loss',
                    #loss            = 'quadratic-loss',
                    evals           = ['accuracy', 'log-loss'], 
                    #optimization    = 'Backpropagation',
                    #optimization    = 'iRPROP-',
                    optimization    = 'iRPROP+',
                    #optimization    = 'scipy_minimize_BFGS',
                    #optimization    = 'scipy_minimize_Newton-CG',
                    #optimization    = 'scipy_minimize_CG',
                    plot_en         = True, 
                    verbose         = True,
                    random_state    = None, 
                    denoise_frac    = 0, 
                    dropout_frac    = 0)
NN.fit( X_tr_n, 
        y_tr,
        X_cv_n,
        y_cv)
         
delay = time.time()-start_time 

print '+Training done, time elapsed: %f [s]' %(delay)

