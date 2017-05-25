import numpy as np
import random
import tensorflow as tf

initializer = tf.contrib.layers.variance_scaling_initializer()
dimension = 1
activation = tf.nn.elu
regularizer = tf.contrib.layers.l2_regularizer(0.001)
learning_rate = 0.01
hiddenSize = 100
inputSize = 1000

weights1 = tf.Variable(initializer([dimension, hiddenSize]), dtype=tf.float32, name='weights1')
weights2 = tf.transpose(weights1, name='weights2')
biases1 = tf.Variable(tf.zeros(hiddenSize), name='biases1')
biases2 = tf.Variable(tf.zeros(inputSize), name='biases2')

X  = tf.placeholder(tf.float32, shape=[None, dimension])

hidden = activation(tf.matmul(X, weights1)+biases1)
outputs = tf.matmul(hidden, weights2) + biases2

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
reg_loss = regularizer(weights1) + regularizer(weights2)
loss = reconstruction_loss + reg_loss

training_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)


#----------------------training network-------------------------------
epochs = 5
batchSize = 10
inputSize = 1000
hiddenSize = 100
trainData = np.array([float(i) for i in range(1000)]).reshape(-1, 1).tolist()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        ranInt = random.randint(0, len(trainData)-batchSize)
       
        epochData = trainData[ranInt:ranInt+batchSize]
        print(epochData)
        [_] = sess.run([training_op], feed_dict={X:epochData})
        a=weights1.eval()
        print(a)