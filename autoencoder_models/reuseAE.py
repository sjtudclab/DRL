from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

n_inputs = 28*28 
n_hidden1 = 300
n_hidden2 = 150
n_hidden3 = n_hidden1
n_outputs = 28*28

learning_rate = 0.01
l2_reg = 0.0001

initializer = tf.contrib.layers.variance_scaling_initializer()
#参数初始化，防止梯度消失或者爆炸，truncated_normal(shape,0.0,stddv=sqrt(factor/n))
# He initializer

X = tf.placeholder(tf.float32,shape=[-1,n_inputs],name="inputs")


my_dense_layer = partial(                 
    tf.layers.dense,
    activation = tf.nn.elu,
    kernal_initializer = initializer,
    kernal_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
)

hidden1 = my_dense_layer(X,n_hidden1)
hidden2 = my_dense_layer(n_hidden1,n_hidden2)
hidden3 = my_dense_layer(n_hidden2,n_hidden3)
outputs = my_dense_layer(n_hidden3,n_outputs,activation = None)

mse = tf.reduce_mean(tf.square(outputs-X))

reg_lossed = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add_n([mse]+reg_lossed)

optimizer = tf.train.AdagradDAOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()