import tensorflow as tf
import numpy as np


xx=tf.placeholder(tf.float32,[None,2,3])
s_step= tf.unstack(tf.transpose(xx,perm=[1,0,2]))
basiccell= tf.contrib.rnn.BasicRNNCell(num_units=3)
output,states = tf.contrib.rnn.static_rnn(basiccell,s_step,dtype=tf.float32)
outputs = tf.transpose(tf.stack(output),perm=[1,0,2])

#print(s_step)
#state = basiccell.zero_state(batch_size=100,dtype=tf.float32) 
#outputs = []
#with tf.variable_scope("testScope"):
#    for time_step in range(2):#batchsize
#        if time_step > 0: tf.get_variable_scope().reuse_variables()
#        (cell_output, state) = basiccell(xx[:,time_step,:], state)
#        #cell_output,states = tf.contrib.rnn.static_rnn(basiccell,,dtype=tf.float32)
#        outputs.append(cell_output)







x= np.array([
    [[0,1,2],[9,8,7]],
    [[3,4,5],[0,0,0]],
    [[9,0,1],[3,2,1]],
])
with tf.Session() as sess:
    # init.run()
    # print(x)
    # a=sess.run(s_step,feed_dict={xx:x})
    # print(a)
    init = tf.global_variables_initializer()
    sess.run(init)
    outputsval,output1=sess.run([outputs,output],feed_dict={xx:x})
    #outputsval=outputs.eval(feed_dict={xx:x})
    print(outputsval)
    print(output1)

