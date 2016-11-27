import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
import numpy as np

# Load traffic signs data.

training_file = "train.p"
testing_file = "test.p"
nb_classes = 43

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']
mean = np.mean(X_train)
X_train = X_train - mean
X_test = X_test - mean

# Split data into training and validation sets.
train_set, cv_set, train_labels, cv_labels = train_test_split(X_train, y_train)

# Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int64, (None))

resized = tf.image.resize_images(x, (227, 227))

# pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)

# `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# Add the final layer for traffic sign classification.
# fc8
shape = (fc7.get_shape().as_list()[-1], nb_classes)
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))

logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
probs = tf.nn.softmax(logits)

# Define loss, training, accuracy operations.
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer().minimize(cost, var_list=[fc8W, fc8b])

preds = tf.arg_max(logits, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, y), tf.float32))

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# Train and evaluate the feature extraction model.
def batch(x, y, size=3, batches=3):
    for i in range(batches):
        idx = np.random.randint(0,x.shape[0], size)
        yield x[idx], y[idx]

for i, b in enumerate(batch(train_set, train_labels, size=128, batches=101)):
    x_batch, y_batch = b
    _, batch_acc, batch_cost = sess.run([optimizer, accuracy, cost ], 
                                    feed_dict={x: x_batch, y: y_batch})
    print(i)
    if i % 10 == 0:
        cv_acc, cv_cost = sess.run([accuracy, cost], feed_dict={x: cv_set[:128], y: cv_labels[:128]})
        print(i, "accuracy:", batch_acc, cv_acc, "cost:", batch_cost, cv_cost)
