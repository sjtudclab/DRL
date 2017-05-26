

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from Agent_v1 import Agent1

import matplotlib.pyplot as plt
import tensorflow as tf
import math
import argparse
import sys
import numpy as np

import os




class lmmodel(Agent1):

    def __init__(self, config,sess,FileList):
        self.L=FileList

        self.config = config
        self.sess =sess
        self.inputSize=50  #20features
        self.stepNum=10   #20 price sequence
        self.hiddenSize=128 # fully connected outputs
        self.neuronNum=100
        self.keep_pro = 0.5
        self.buildNetwork()
        self.saver = tf.train.Saver(tf.global_variables())
    
    #input states sequence, generate the action vector by policy Network
    def choose_action(self, state):  
        """Choose an action."""
        return self.sess.run(self.argAction, feed_dict={self.states: state})

    # build the policy Network and value Network
    def buildNetwork(self):
        self.states = tf.placeholder(tf.float32,shape=[None,self.stepNum, self.inputSize],name= "states")
        self.critic_rewards = tf.placeholder(tf.float32,shape=[None],name= "critic_rewards")
        self.new_lr = tf.placeholder(tf.float32,shape=[],name="learning_rate")
        self.lr = tf.Variable(0.1,trainable=False)
        
        # PolicyNetwork
        with tf.variable_scope("Policy") :
 
            L0= tf.contrib.layers.fully_connected(
                inputs=self.states,
                num_outputs=self.hiddenSize, #hidden
                activation_fn=tf.nn.relu,
                weights_initializer=tf.truncated_normal_initializer(stddev=1.0),
                biases_initializer=tf.zeros_initializer()
            )
            L1= tf.contrib.layers.fully_connected(
                inputs=L0,
                num_outputs=self.hiddenSize, #hidden
                activation_fn=tf.nn.relu,
                weights_initializer=tf.truncated_normal_initializer(stddev=1.0),
                biases_initializer=tf.zeros_initializer()
            )

            #construct a lstmcell ,the size is neuronNum, 暂时不加上dropout
            lstmcell = tf.contrib.rnn.BasicLSTMCell(self.neuronNum, forget_bias=1.0, state_is_tuple=True)
            cell =tf.contrib.rnn.DropoutWrapper(lstmcell, output_keep_prob=0.5)
            #cell = tf.contrib.rnn.MultiRNNCell([cell_drop for _ in range(2)], state_is_tuple=True)

            #系统下一时刻的状态仅由当前时刻的状态产生
            #nowinput = tf.reshape(L1,[-1,2,self.hiddenSize])
            nowinput = L1
            outputnew,statenew = tf.nn.dynamic_rnn(cell,nowinput,dtype=tf.float32)
            outputs = tf.reshape(outputnew,[-1,self.neuronNum])
            
            #outputs= tf.contrib.layers.fully_connected(
            #    inputs=outputs0,
            #    num_outputs=self.hiddenSize, #hidden
            #    activation_fn=tf.nn.relu,
            #    weights_initializer=tf.truncated_normal_initializer(stddev=1.0),
            #    biases_initializer=tf.zeros_initializer()
            #)
            
            # last layer is a fully connected network + softmax 
            softmax_w = tf.get_variable( "softmax_w", [self.neuronNum, 3], dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=1.0))
            softmax_b = tf.get_variable("softmax_b", [3], dtype=tf.float32)
            logits = tf.matmul(outputs, softmax_w) + softmax_b
            self.probs = tf.nn.softmax(logits, name="action")
            # fetch the maximum probability
            self.action0 = tf.reduce_max(self.probs, axis=1)
            # fetch the index of the maximum probability
            self.argAction = tf.argmax(self.probs, axis=1)

            #loss,train
            self.lr_update = tf.assign(self.lr,self.new_lr) #更新学习率
            self.policyloss =policyloss  = tf.log(self.action0)*self.critic_rewards
            loss = tf.negative(tf.reduce_mean(policyloss),name="loss")
            
            #tf.summary.scalar('actor_loss',tf.abs(loss))
            #self.actor_train = tf.train.AdamOptimizer(self.lr).minimize(loss)

            tvars = tf.trainable_variables() #得到可以训练的参数
            self.agrads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),5)  #防止梯度爆炸
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.actor_train = optimizer.apply_gradients(zip(self.agrads, tvars))

    def assign_lr(self,session,lr_value):
        session.run(self.lr_update,feed_dict={self.new_lr:lr_value})


    def discount_rewards(self,x, gamma):
        """
        Given vector x, computes a vector y such that
        y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
        """
        result = [0 for i in range(len(x))]
        element = 0
        for i in range(len(x)-1, -1, -1):  #-2
            element = x[i] + gamma * element
            result[i] = element

        return result

    #在策略网络的损失函数中，采用一步截断优势函数，即A=rt+gamma*V(st+1)-V(st)
    def policy_rew(self,r,v,gamma):
        R = [0 for i in range(len(r))]
        element = 0
        for i in range(len(r)-1):
            element = r[i] + gamma * v[i+1]
            R[i] = element
        #R[len(r)-1]=r[len(r)-1]默认最后一个状态R为0
        return R

    #在值函数网络中，target=rt+gamma*rt+1
    def value_rew(self,r,gamma):
        R = [0 for i in range(len(r))]
        element = 0
        for i in range(len(r)-1):
            element = r[i] + gamma * r[i+1]
            R[i] = element
        return R


    def learn(self):
        #self.merged = tf.summary.merge_all()
        #self.writer = tf.summary.FileWriter("/home/swy/code/DRL/tbencoder", self.sess.graph) 
        # 10 days =  240*10 = 2400
        batchsize = 120  #指的是lstm中batchsize
        timestep = 10    #lstm中timestep
       
        max_epoch=2
        learningrate = 0.5

        trainfalse =True
        if trainfalse:
                    
                total=[] 
                sum = 0
                win = 0
                predition = []
        

                #lr_decay = 0.5**max(j+1 - max_epoch,0.0)
                #self.assign_lr(self.sess,learningrate*lr_decay)
                
                #for k in range(2):
                stockList = ['IF1601.CFE.csv', 'IF1602.CFE.csv', 'IF1603.CFE.csv','IF1604.CFE.csv', 'IF1605.CFE.csv', 'IF1606.CFE.csv', 'IF1607.CFE.csv', 'IF1608.CFE.csv', 'IF1609.CFE.csv', 'IF1610.CFE.csv', 'IF1611.CFE.csv', 'IF1612.CFE.csv','IF1701.CFE.csv', 'IF1702.CFE.csv', 'IF1703.CFE.csv']
                super(lmmodel,self).__init__(stockList[0], 50)
                epoch= (len(self.state)//batchsize-1)//timestep #遍历整个数据集
                for j in range(epoch):  
    
                    trajectory = self.get_trajectory(j, timestep, batchsize)
                    action = trajectory["action"]
                    state = trajectory["state"]
                    returns = trajectory["reward"]
                    #print(action)
                    
                    #夏普比率 
                    rmean = np.mean(returns)
                    rvariance = np.std(returns)
                    #print(rmean/rvariance) 

                    #if np.sum(returns)>0:
                    #    win = win +1
                    #sum = sum +1
                    total.append(np.sum(returns))
                    print(np.sum(returns))
                

                    lr,actorResults,loss = self.sess.run([self.lr ,self.actor_train,self.policyloss],feed_dict={
                        self.states: state,
                        self.critic_rewards:returns
                    })
                    print("loss")
                    print(np.shape(loss))
                           

                plt.figure()
                x_values = range(len(total))
                y_values = total
                plt.plot(x_values, y_values)
                plt.savefig('swy.png')
                    
        #self.writer.close()



class config(object):
    learning_rate= 1.0
    num_layers =2
    num_steps= 20
    hidden_size = 28
    batch_size=100
    number=1000

def get_config():
    return config()


def main():
    os.chdir("/home/swy/code/DRL/Info1/Info/data")
    L=[]
    for files in os.walk("/home/swy/code/DRL/Info1/Info/data"):
        for file in files:
            L.append(file) 


    #if tf.gfile.Exists('/home/swy/code/DRL/tbencoder'):
    #    tf.gfile.DeleteRecursively('/home/swy/code/DRL/tbencoder')
    #tf.gfile.MakeDirs('/home/swy/code/DRL/tbencoder')

    config=get_config()
    sess= tf.InteractiveSession()
    trainable=False
    if trainable:
        out = lmmodel(config=config,sess=sess,FileList=L)
        sess.run(tf.global_variables_initializer())
        out.learn()
        save_path = out.saver.save(sess, '/home/swy/code/DRL/cpencoder/model0525_2100.ckpt')
    else:
        out = lmmodel(config=config,sess=sess,FileList=L)
        load_path = out.saver.restore(sess,'/home/swy/code/DRL/cpencoder/model0525_2100.ckpt')
        out.learn()
            


if __name__ == '__main__':
    main()














    





    

