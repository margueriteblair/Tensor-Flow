#In this notebook we'll be walking through 4 fundamental machine learning algorithms.
#Tensor Flow (TF) is an incredibly large library, and it's not always necessary to memorize syntax
#The algorithms we'll look at include:
#Linear regression, classification, clustering, hidden markov models

#Linear regression is one of the most basic forms of machine learning

from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pandas as pd #allows for representing data 
import numpy as np
import sklearn
from six.moves import urllib
# import TensorFlow.compat.v2.feature_column as fc
import tensorflow as tf
# import tensorflow.compat.v2.feature_column as fc
# import tensorflow as tf
#We'll be using linear regression to make a line of best fit and then find patterns in the data to predict future outcomes
#If we create linear correspondence, we can draw a line of best fit to classify the data
#linear regression is used when we have data points that correleate linearly
#but we can also have multidimensional values
#the line of best fit will have a y=mx+b type fit

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') #this is the training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') #this is the testing data
y_train = dftrain.pop('survived') 
y_eval = dfeval.pop('survived')

#pd.read_csv() method will return to us a new pandas dataframe object, think of a df similarly to a table
#we've decided to pop off the survived variable and store it in a new var
#label is our output information
#within the dataset, 0 stands for didn't survive, 1 survived
 

