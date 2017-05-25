from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data',one_hot=True)

tf.reset_default_graph()

n_inputs = 28*28 
n_hidden1 = 300
n_hidden2 = 150
n_hidden3 = n_hidden1
n_outputs = 10

learning_rate = 0.01
l2_reg = 0.0001

activation = tf.nn.elu
regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
initializer = tf.contrib.layers.variance_scaling_initializer()
#参数初始化，防止梯度消失或者爆炸，truncated_normal(shape,0.0,stddv=sqrt(factor/n))
# He initializer

X = tf.placeholder(tf.float32,shape=[None,n_inputs],name="inputs")
y = tf.placeholder(tf.int32, shape=[None])

weights1_init = initializer([n_inputs,n_hidden1])
weights2_init = initializer([n_hidden1,n_hidden2])
weights3_init = initializer([n_hidden2,n_hidden3])

weights1 = tf.Variable(weights1_init,dtype = tf.float32,name="weights1")
weights2 = tf.Variable(weights2_init,dtype = tf.float32,name="weights2")
weights3 = tf.Variable(weights3_init,dtype = tf.float32,name="weights3")

bias1 = tf.Variable(tf.zeros(n_hidden1),name="bias1")
bias2 = tf.Variable(tf.zeros(n_hidden2),name="bias2")
bias3 = tf.Variable(tf.zeros(n_hidden3),name="bias3")

hidden1 = activation(tf.matmul(X,weights1)+bias1)
hidden2 = activation(tf.matmul(hidden1,weights2)+bias2)
logits = tf.matmul(hidden2,weights3)+bias3

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
reg_loss = regularizer(weights1)+regularizer(weights2)+regularizer(weights3)
loss = cross_entropy + reg_loss
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = open.minimize(loss)

correct = tf.nn.in_top_k(logits,y,1)
accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

init = tf.global_variables_initializer()
pretrain_saver = tf.train.Saver([weights1,weights2,bias1,bias2])
saver = tf.train.Saver()

n_epochs = 4
batch_size = 100
n_labeled_instances = 20000


# without pretrain
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batched = n_labeled_instances//batch_size
        for iteration in range(n_batched):
            x_batch,y_batch  = mnist.train.next_batch(100)
            sess.run(training_op,feed_dict={X:x_batch,y:y_batch})
        accuracy_val = accuracy.eval(feed_dict={X:x_batch,y:y_batch})
        saver.save(sess,"")
        accuracy_val = accuracy.eval(feed_dict={X:mnist.test.images,y:mnist.test.labels})

# pretrain
pretrain = False
if pretrain:
    with tf.Session() as sess:
        init.run()
        # 重新获得所有的参数
        pretrain_saver.restore(sess,"")
        # reuse 部分参数
        reuse_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES,scope="hidden[1,2,3]")
        reuse_vars_dict = dict([(var.name,var.name) for var in reuse_vars])
        original_saver = tf.Saver(reuse_vars_dict)
        for epoch in range(n_epochs):
            n_batched = n_labeled_instances//batch_size
            for iteration in range(n_batched):
                x_batch,y_batch  = mnist.train.next_batch(100)
                sess.run(training_op,feed_dict={X:x_batch,y:y_batch})
            accuracy_val = accuracy.eval(feed_dict={X:x_batch,y:y_batch})
            saver.save(sess,"")
            accuracy_val = accuracy.eval(feed_dict={X:mnist.test.images,y:mnist.test.labels})
