import numpy as np
import tensorflow as tf
X=[1,2]
state=[0.0,0.0]
w_cell_state=np.asarray([[0.1,0.2],[0.3,0.4]])
w_cell_input=np.asarray([0.5,0.6])
b_cell=np.asarray([0.1,-0.1])

w_output=np.asarray([[1.0],[2.0]])
b_output=0.1

for i in range(len(X)):
    before_activation=np.dot(state,w_cell_state) + X[i]*w_cell_input + b_cell

    state=np.tanh(before_activation)
    final_output=np.dot(state,w_output)+b_output
    print(final_output)


w_cell_state=tf.constant([[0.1,0.2],[0.3,0.4]])
w_cell_input=tf.constant([[0.5,0.6]])
b_cell=tf.constant([[0.1,-0.1]])
w_output=tf.constant([[1.0],[2.0]])
state=tf.placeholder(tf.float32,shape=(1,2))

for i in range(len(X)):
    if i != 0:
        state=states
    before_activation= tf.matmul(state,w_cell_state)  + X[i]*w_cell_input + b_cell
    states =tf.tanh(before_activation)
    final_output= tf.matmul(states,w_output)+b_output
    with tf.Session() as sess:
        init=tf.global_variables_initializer()
        sess.run(init)
        print(sess.run(final_output,feed_dict={state:[[0.0,0.0]]}))