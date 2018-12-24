import matplotlib as mpl
import numpy as np
import tensorflow as tf
mpl.use('Agg')
from matplotlib import pyplot as plt
HIDDEN_SIZE =30
NUM_LAYERS=2
TIMESTEPS=10
TRAINING_STEPS =10000
BATCH_SIZE =32
TRAINING_EXAMPLES=10000
TESTING_EXAMPLES=1000
SAMPLE_GAP=0.01
def generate_data(seq):
    X=[]
    Y=[]
    for i in range(len(seq)-TIMESTEPS):
        X.append([seq[i:i+TIMESTEPS]])
        y.append([seq[i+TIMESTEPS]])
    return np.array(X,dtype=np.float32), np.array(y,dtype=np.float32)
def lstm_model(X, y, is_training):
    cell = tf.nn.rnn_cell.MultiRNNCell([
        tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        for _ in range(NUM_LAYERS)
    ])
    outputs, _ =tf.nn.dynamic_rnn(cell, X, dtypes=tf.float32)
    output =ouputs[:,-1,:]
    predictions =tf.contrib.layers.fully_connected(
        output , 1, activation_fn=None
    )
    if not is_training:
        retrun predictions, None, None
    loss =tf.losses.mean_squared_error(labels=y, predictions=predictions)
    train_op =tf.contrib.layers.optimize_loss(
        loss, tf.train.get_global_step(),
        optimizer="Adagrad", learning_rate=0.1
    )
    return predictions, loss, train_op

def train( sess,train_X, train_y):
    ds= tf.data.Dataset.from_tensor_slices((train_X, train_y))
    ds =ds.repeat().shuffle(1000).batch(BATCH_SIZE)
    X, y= ds.make_one_shot_iterator().get_next()

    with tf.variable_scope("model"):
        predictions, loss, train_op =lstm_model(X, y, True)
        sess.run(tf.global_variables_initializer())
        for i in range(TRAINING_STEPS):


