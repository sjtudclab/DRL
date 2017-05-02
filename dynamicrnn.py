import tensorflow as tf
import numpy as np

n_steps=2
n_inputs=3
n_neurons=5
x= tf.placeholder(tf.float32,[None,n_steps,n_inputs])
seq_length= tf.placeholder(tf.int32,[None])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs,states = tf.nn.dynamic_rnn(basic_cell,x,dtype=tf.float32,sequence_length=seq_length)
a=tf.reduce_max(x,2)

x_batch = np.array([
    [[0,1,2],[9,8,7]],
    [[3,4,5],[0,0,0]]
])
seq_len_batch = np.array([2,1])

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    outputsval,output1=sess.run([outputs,states],feed_dict={x:x_batch,seq_length:seq_len_batch})
    print(outputsval)
