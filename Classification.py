from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import pandas as pd

#the datase we're using is separating flowers into 3 different species
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
)

test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"
)

#we're loading training and testing into CSV column names
#we don't need to convert categorical data into numerical data
train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
print(train.head())

train_y = train.pop('Species')
test_y = test.pop('Species')
train.head()
print(train.shape)

def input_fn(features, labels, training=True, batch_size=256):
    #converts the inputs to a Dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)


