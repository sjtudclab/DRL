#训练数据是1601-1612一年的数据，测试集是1701-1703
#训练数据batchsize为100，连续序列读入
#功能是使用三层autoencoder训练中间一层，没有加入l1范数

import tensorflow as tf 
import numpy as np
from Autoencoder import Autoencoder 
from Agent_v2 import Agent2

import os

os.chdir("/home/swy/code/DRL/autoencoder_models/data")
# save all the file
L=[]
for files in os.walk("/home/swy/code/DRL/autoencoder_models/data"):
    for file in files:
        L.append(file) 
# all the filename
print(L[2])

#np.savetxt('new.csv', L, delimiter = ',')  



x= tf.placeholder(tf.float32,shape=[None, 10],name= "states")
w1 = tf.placeholder(tf.float32,shape=[10,100],name="w1")
b1 = tf.placeholder(tf.float32,shape=[100],name="b1")
critic_feedback = tf.placeholder(tf.float32,shape=[None,None],name= "critic_feedback")
critic_rewards = tf.placeholder(tf.float32,shape=[None,None],name= "critic_rewards")

#Auroencoder trained network
outputs = tf.nn.relu(tf.matmul(x,w1)+b1)

#softmax classifier 
softmax_w = tf.get_variable( "softmax_w", [100, 3], dtype=tf.float32,initializer=tf.truncated_normal_initializer())
softmax_b = tf.get_variable("softmax_b", [3], dtype=tf.float32)
logits = tf.matmul(outputs, softmax_w) + softmax_b
probs = tf.nn.softmax(logits, name="action")
# fetch the maximum probability
action0 = tf.reduce_max(probs, axis=1)
# fetch the index of the maximum probability
argAction = tf.argmax(probs, axis=1)

#loss,train
#policyloss =policyloss  = tf.log(action0)*(critic_rewards-critic_feedback)
#self.policyloss =policyloss  = tf.log(self.action0)*self.critic_rewards
#loss = tf.negative(tf.reduce_mean(policyloss),name="loss")

#actor_train = tf.train.AdamOptimizer(0.01).minimize(loss)

#
autoencoder = Autoencoder(n_input = 10,n_hidden = 100,transfer_function = tf.nn.softplus,optimizer = tf.train.AdamOptimizer(learning_rate = 0.001))

# train the whole file data

batchsize=100
epoch=10
#print(np.floor(len(Agent.dataBase)/batchsize))
for j in range(epoch):
    for k in range(12):
        Agent=Agent2(L[2][k], 10, 100, 2000)
        for i in range(int(np.floor(len(Agent.dataBase)/batchsize))):
            #print(len(Agent.dataBase))
            state = Agent.get_state(i)
            cost = autoencoder.partial_fit(state)
            if i % 10==0:
                print("cost")
                print(cost)

w=autoencoder.getWeights()
b=autoencoder.getBiases()
#print(w1)
#print(b1)

#sess= tf.InteractiveSession()
#sess.run(tf.global_variables_initializer())

cost=[]
for i in range(12,15):
    Agent1=Agent2(L[2][k], 10, 100, 2000)
    for j in range(int(np.floor(len(Agent1.dataBase)/batchsize))):
        state0 = Agent1.get_state(j)
        cost0 = autoencoder.calc_total_cost(state0)
        if j%10 ==0:
            cost.append(cost0)
print(np.mean(cost))
#costres=open('/home/swy/code/DRL/autoencoder_models/data/res.txt',w)
#costres.write(cost)
#costres.close()


#for i in range(100):
#    print('now')
#    state0 = Agent1.get_state(i)
#    cost = autoencoder.calc_total_cost(state0)
    #cost = sess.run([calc_total_cost],feed_dict={x:state0})
    #if i%100 ==0:
#    print(cost)
    
       #print(action)
       #reward=Agent1.get_return(action,state0)
       #print(np.sum(reward))
       



