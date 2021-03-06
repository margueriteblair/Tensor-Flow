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

#feature columns
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

print(my_feature_columns)

#now we're ready to choose a model; there's lots of preexisting models for classification, such a s DNNClassifier, LinearClassifier, etc
#we're going to choose the DNNClassifier b/c there might not be a linear correlation in our dataset
#estimator stores a bunch of premade models
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    #we're making two hidden layers of 30 nodes and 10 nodes respectively, this is arbitrary
    hidden_units=[30,10],
    #the model must choose between 3 classes, we know there are 3 classes for the flowers
    n_classes=3
)

classifier.train(
    #a lambda is an unnamed one liner function
    #in this instance, we need a function that returns a function
    input_fn=lambda: input_fn(train, train_y, training=True),
    #steps is similar to an epoch, we're just saying we'll go through the dataset until we go through 5000
    steps=5000)
    #the lower the loss number the better
    #.39 isn't great :(

eval_result = classifier.evaluate(input_fn=lambda: input_fn(test, test_y, training=False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


def input_fn(features, batch_size=246):
    #convert the inputs to a dataset without labels
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

print("Please type numeric values as prompted:")
for feature in features:
    valid = True
    while valid:
        val = input(feature + ": ")
        if not val.isdigit(): valid = False

    predict[feature] = [float(val)]

predictions=classifier.predict(input=lambda: input_fn(predict))
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print("Prediction is {} ({:.1f}%".format(SPECIES[class_id], 100*probability))

