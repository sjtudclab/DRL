

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from Agent import Agent

import matplotlib.pyplot as plt
import tensorflow as tf
import math
import argparse
import sys
import numpy as np
import os

from AutoEncoder.autoencoder import AutoEncoder


class lmmodel(Agent):

    def __init__(self,sess,FileList):
        super(lmmodel, self).__init__('/home/swy/code/DRL/0607/Info1/Info/data/IF1601.CFE.csv', 60)

        self.inputSize=10  #2features
        self.stepNum=60   #20 price sequence
        self.hiddenSize=50 # fully connected outputs
        self.neuronNum=20

        self.L=FileList
        self.sess =sess

        self.buildNetwork()

        self.saver = tf.train.Saver(tf.global_variables())

    #input states sequence, generate the action vector by policy Network
    def choose_action(self, state):  
        """Choose an action."""
        state = np.reshape(state, [-1, self.inputSize])
        context = {}
       # context.update(self.paraDict)
        context.update({self.stateTrain:state})
        return self.sess.run(self.argAction, feed_dict=context)

    def enterParameter(self,session,fullyConnected):
        self.weights_dict = fullyConnected[0]
        self.biases_dict = fullyConnected[1]
      #  self.paraDict={self.weights1:self.weights_dict['weights1'], self.weights2: self.weights_dict['weights2'], self.weights3: self.weights_dict['weights3'], self.biases1: self.biases_dict['biases1'],
      #              self.biases2: self.biases_dict['biases2'],self.biases3: self.biases_dict['biases3']}
        session.run(self.update_w1,feed_dict={self.weights1:self.weights_dict['weights1']})      
        session.run(self.update_w2,feed_dict={self.weights2:self.weights_dict['weights2']})   
        session.run(self.update_w3,feed_dict={self.weights3:self.weights_dict['weights3']})   
        session.run(self.update_b1,feed_dict={self.biases1:self.biases_dict['biases1']})   
        session.run(self.update_b2,feed_dict={self.biases2:self.biases_dict['biases2']})   
        session.run(self.update_b3,feed_dict={self.biases3:self.biases_dict['biases3']})       

    # build the policy Network and value Network
    def buildNetwork(self):
        self.stateTrain = tf.placeholder(tf.float32,shape=[None, self.inputSize],name= "stateTrain")
        self.critic_rewards = tf.placeholder(tf.float32,shape=[None],name= "critic_rewards")

        self.new_lr = tf.placeholder(tf.float32,shape=[],name="learning_rate")
        self.lr = tf.Variable(0.1, trainable=False)

        # PolicyNetwork

        with tf.variable_scope("Policy") :

            self.weights1 = tf.placeholder(tf.float32, shape = [self.inputSize, self.hiddenSize], name = "weights1")
            self.biases1 = tf.placeholder(tf.float32, shape = [self.hiddenSize], name = "biases1")
            self.weights2 = tf.placeholder(tf.float32, shape=[self.hiddenSize, self.hiddenSize], name="weights2")
            self.biases2 = tf.placeholder(tf.float32, shape=[self.hiddenSize], name="biases2")
            self.weights3 = tf.placeholder(tf.float32, shape=[self.hiddenSize, self.hiddenSize], name="weights3")
            self.biases3 = tf.placeholder(tf.float32, shape=[self.hiddenSize], name="biases3")
            #变量保存
            self.w1= tf.get_variable( "w1", [self.inputSize, self.hiddenSize], dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=1.0/math.sqrt(float(self.hiddenSize))),trainable=False)
            self.b1 = tf.get_variable("b1", [self.hiddenSize], dtype=tf.float32,initializer=tf.zeros_initializer(),trainable=False)
            self.w2= tf.get_variable( "w2", [self.hiddenSize, self.hiddenSize], dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=1.0/math.sqrt(float(self.hiddenSize))),trainable=False)
            self.b2 = tf.get_variable("b2", [self.hiddenSize], dtype=tf.float32,initializer=tf.zeros_initializer(),trainable=False)
            self.w3= tf.get_variable( "w3", [self.hiddenSize, self.hiddenSize], dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=1.0/math.sqrt(float(self.hiddenSize))),trainable=True)
            self.b3 = tf.get_variable("b3", [self.hiddenSize], dtype=tf.float32,initializer=tf.zeros_initializer(),trainable=True)
            
            self.update_w1 = tf.assign(self.w1,self.weights1)
            self.update_w2 = tf.assign(self.w2,self.weights2)
            self.update_w3 = tf.assign(self.w3,self.weights3)
            self.update_b1 = tf.assign(self.b1,self.biases1)
            self.update_b2 = tf.assign(self.b2,self.biases2)
            self.update_b3 = tf.assign(self.b3,self.biases3)

            activation = tf.nn.relu

            L0 = activation(tf.matmul(self.stateTrain, self.w1)+self.b1)
            L1 = activation(tf.matmul(L0, self.w2)+self.b2)
            L2 = activation(tf.matmul(L1, self.w2)+self.b2)

            L2 = tf.reshape(L2, [-1, self.stepNum, self.hiddenSize])
            #construct a lstmcell ,the size is neuronNum
            cell = tf.contrib.rnn.BasicLSTMCell(self.neuronNum, forget_bias=1.0, state_is_tuple=True)
            #cell_drop =tf.contrib.rnn.DropoutWrapper(lstmcell, output_keep_prob=0.5)
            #lstmcell = tf.contrib.rnn.BasicLSTMCell(self.neuronNum, forget_bias=1.0, state_is_tuple=True,activation=tf.nn.relu)
            #cell_drop=tf.contrib.rnn.DropoutWrapper(lstmcell, output_keep_prob=0.5)
            #construct 5 layers of LSTM
            #cell = tf.contrib.rnn.MultiRNNCell([cell_drop for _ in range(2)], state_is_tuple=True)


            #系统下一时刻的状态仅由当前时刻的状态产生
            outputnew, statenew = tf.nn.dynamic_rnn(cell, L2, dtype=tf.float32)

            outputs = self.outputs = outputnew[:,self.stepNum-1,:] # 取最后一个step的结果

            
            softmax_w = tf.get_variable( "softmax_w", [self.neuronNum, 3], dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=1.0/math.sqrt(float(self.neuronNum))))
            softmax_b = tf.get_variable("softmax_b", [3], dtype=tf.float32)
            logits = tf.matmul(outputs, softmax_w) + softmax_b
            self.probs = tf.nn.softmax(logits, name="action")

            # fetch the maximum probability # fetch the index of the maximum probability
            self.action0 = tf.reduce_max(self.probs, axis=1)
            self.argAction = tf.argmax(self.probs, axis=1)

            #loss,train
            #self.entropy = -tf.reduce_sum(self.probs * tf.log(self.probs), 1, name="entropy")
            self.lr_update = tf.assign(self.lr,self.new_lr)
            #self.policyloss = policyloss  = tf.log(self.action0)*self.critic_rewards + 0.01 * self.entropy
            self.policyloss = policyloss = tf.log(self.action0)*self.critic_rewards
            self.loss = loss = tf.negative(tf.reduce_mean(policyloss),name="loss")
            #loss = tf.negative(policyloss,name="loss")
            tf.summary.scalar('actor_loss',tf.abs(loss))
            #self.actor_train = tf.train.AdamOptimizer(self.lr).minimize(loss)
            tvars = tf.trainable_variables() #得到可以训练的参数
            self.agrads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),5)  #防止梯度爆炸
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.actor_train = optimizer.apply_gradients(zip(self.agrads, tvars))
    
    #给learning rate 赋值
    def assign_lr(self,session,lr_value):
        session.run(self.lr_update,feed_dict={self.new_lr:lr_value})

    # 计算折扣因子
    def discount_rewards(self,rewards,discount_rate):
        discounted_rewards = np.empty(len(rewards))
        cumulative_rewards = 0
        for step in reversed(range(len(rewards))):
            cumulative_rewards = rewards[step] + cumulative_rewards*discount_rate
            discounted_rewards[step] = cumulative_rewards
        return discounted_rewards
    # normalize rewards
    def discount_and_normalize_rewards(self,all_rewards,discount_rate):
        all_discounted_rewards = [self.discount_rewards(rewards,discount_rate) for rewards in all_rewards]

        flat_rewards = np.concatenate(all_discounted_rewards)
        reward_mean = flat_rewards.mean()
        reward_std = flat_rewards.std()
        if reward_mean == 0 and reward_std == 0:
            return 0
        else:
            return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]


    def learn(self):
        
        
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("/home/swy/code/DRL/0607/tb", self.sess.graph) 
        # 5 days
        batchsize=240
        timestep = self.stepNum
        epoch=1
        max_epoch=2
        learningrate = 0.1

            
        for j in range(epoch):
            total=[]
            act_sum = []
            price_sum = []
            sum = 0
            win = 0
            lr_decay = 0.5**max(j+1 - max_epoch,0.0)

            for i in range(0,len(self.state)-batchsize,240):

                #index = np.random.randint(0, len(self.state)-batchsize-timestep)
                trajectory = self.get_trajectory(i, batchsize)
                state = trajectory["state"]
                rew =trajectory["reward"]
                price = trajectory["price"]
                price_sum.append(price)
                returns = self.discount_and_normalize_rewards([rew],0.95)
                if returns == 0:
                    print("over")
                    continue
                else:
                    returns = np.reshape(returns,[-1])
                    
                    action = trajectory["action"]
                    act_sum.append(action)

                    #统计收益大于0的周数
                    if np.sum(returns)>0:
                        win = win +1
                    sum = sum +1
                    total.append(np.sum(rew))
                    state = np.reshape(state, [-1, self.inputSize])
                   

                    context = {}
                    #context.update(self.paraDict)
                    context.update({self.stateTrain: state})
                    context.update({self.critic_rewards:returns})
                    probs, loss,action0 = self.sess.run([self.probs, self.policyloss,self.argAction],feed_dict=context)
                    #print (probs)
                    #print (loss)
                    summary,actorResults= self.sess.run([self.merged,self.actor_train],feed_dict=context)
                    self.writer.add_summary(summary,i)


            print(win/sum)
            plt.figure()
            x_values = range(len(total))
            y_values = total
            plt.plot(x_values, y_values)
            plt.savefig(str(j)+'.png')

            plt.figure()   
            price_sum=np.reshape(price_sum,[-1])   
            x_values = range(len(price_sum))
            y_values = price_sum
            plt.plot(x_values, y_values)
            plt.savefig("price:"+str(j)+'.png')

            plt.figure()      
            act_sum=np.reshape(act_sum,[-1])   
            x_values = range(len(act_sum))
            y_values = act_sum
            plt.plot(x_values, y_values)
            plt.savefig("act:"+str(j)+'.png')


            self.writer.close()


def main():
    os.chdir("/home/swy/code/DRL/0607/Info1/Info/data")
    L=[]
    for files in os.walk("/home/swy/code/DRL/0607/Info1/Info/data"):
        for file in files:
            L.append(file) 

    if tf.gfile.Exists('/home/swy/code/DRL/0607/tb'):
            tf.gfile.DeleteRecursively('/home/swy/code/DRL/0607/tb')
    tf.gfile.MakeDirs('/home/swy/code/DRL/0607/tb')

    sess= tf.InteractiveSession()
    trainable= True
    if trainable:
        out = lmmodel(sess=sess,FileList=L)
        config = {}
        config['hiddenSize'] = [10, 50, 50,50]
        AETrain = AutoEncoder(config=config)
        AETrain.getTrainData(out.getData())
        AETrain.learn()
        out.enterParameter(sess,AETrain.getParameter())
        sess.run(tf.global_variables_initializer())
        out.learn()
        save_path = out.saver.save(sess, '/home/swy/code/DRL/0607/model0606_d1.ckpt')
    else:
        out = lmmodel(sess=sess,FileList=L)
        load_path = out.saver.restore(sess,'/home/swy/code/DRL/0607/model0605_d2.ckpt')
        out.learn()
        save_path = out.saver.save(sess, '/home/swy/code/DRL/0607/model0605_d3.ckpt')



if __name__ == '__main__':
    main()
    #tf.app.run()














    





    

