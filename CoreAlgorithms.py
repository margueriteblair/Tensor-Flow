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
import tensorflow as tf
import tensorflow.compat.v2.feature_column as fc
#We'll be using linear regression to make a line of best fit and then find patterns in the data to predict future outcomes
#If we create linear correspondence, we can draw a line of best fit to classify the data
#linear regression is used when we have data points that correleate linearly
#but we can also have multidimensional values
#the line of best fit will have a y=mx+b type fit

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') #this is the training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') #this is the testing data
y_train = dftrain.pop('survived') 
y_eval = dfeval.pop('survived')
print(dftrain.head()) #it works!
print(dftrain.loc[0], y_train.loc[0])
print(dftrain.describe()) #this will give us high level information about the df
print(dftrain.shape)
print(y_train.head())
#pd.read_csv() method will return to us a new pandas dataframe object, think of a df similarly to a table
#we've decided to pop off the survived variable and store it in a new var
#label is our output information
#within the dataset, 0 stands for didn't survive, 1 survived
 
#now we'll be creating visuals
dftrain.age.hist(bins=20)
plt.show()
dftrain.sex.value_counts().plot(kind='barh') #shows sex breakdown 
plt.show()
dftrain['class'].value_counts().plot(kind='barh')
plt.show()
pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='bar').set_xlabel("% survived")
plt.show()

#we have above loaded in two data sets, with different shapes
print(dfeval.shape)
print(dftrain.shape)

#categorical data is something that's not numeric, that groups entries into a category
#we essentially need to encode categorical data into integer data, like male = 1, female = 0
#the model that we're making will need to just know that male and female are different
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
#numerical data is already in the format we want it in
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = [] #features of the opposite of labels; features are our input variables
#feature columns are actually what we're going to feed to our linear regression model to make predictions

for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique() #this gets a list of all unique values from a given feature column
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
    #this creates a column in the form of a numpy array
for feature_name in NUMERIC_COLUMNS:
    #with a numerical column there could be an infinite amount of values
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

# print(feature_columns)

#we train the model by feeding it information
#for this specific model data is going to be streamed into the model in small batches of 32; batches are typically in multiples of 32
#we won't be feeding our entire dataset to our model all at once, but in small batches of entries
#we will feed these batches to our model multiple times according to the number of epochs
#an epoch is simply one stream of our entire dataset, the number of epochs we define is the amount of times our model will see the entire dataset
#ex: if we have 10 epochs, then our model will see the same dataset 10 times
#because we need to feed our data in batches and multiple times, we need to create something called an input function
#the input function will define how our dataset will be converted into batches every epoch
#the tf model that we are going to use requires that the data we pass it comes in as a tf.data.Dataset object
#this means that we must make an input funtion that can convert our pandas df into an object of type tf.data.Dataset

#this input function will dictate how we're breaking our data into epochs and batches to feed our data to the model
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        #takes our data and encodes it into a dataset (ds) object pandas -> tf
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df)) #this will create a tf.data.Dataset object with data
        if shuffle:
            ds = ds.shuffle(1000) #you randomize the order of data
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds #this will return a batch of the dataset
    return input_function #return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

#this is us creating the model
#estimators are basic implementations
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
linear_est.train(train_input_fn) #training
result = linear_est.evaluate(eval_input_fn) #get the model metrics by testing on testing data
clear_output() #clears console output
print("Accuracy", result['accuracy'])

#tensor flow models are meant to be for large data
result = list(linear_est.predict(eval_input_fn)) #we need to pass in an input function to make a prediction
#linear_est is actually the name of our model
print(dfeval.loc[0])
print(y_eval.loc[0])
print(result[0]['probabilities']) #we get a dictionary from this that represents our predictions
#this shows the probability of survival
#test