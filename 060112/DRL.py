

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




class lmmodel(Agent):

    def __init__(self,sess,FileList):
        #super(lmmodel,self).__init__('data/IF1602.CFE.csv', 10, 240, 2000)

        self.L=FileList
        self.sess =sess
        self.inputSize=6  #2features
        self.stepNum=10   #20 price sequence
        self.hiddenSize=128 # fully connected outputs
        self.neuronNum=100
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
        self.actions_taken = tf.placeholder(tf.int32,shape=[None,2],name= "actions_taken")
        self.new_lr = tf.placeholder(tf.float32,shape=[],name="learning_rate")
        self.lr = tf.Variable(0.01,trainable=False)

        # PolicyNetwork
        with tf.variable_scope("Policy") :

            L0= tf.contrib.layers.fully_connected(
                inputs=self.states,
                num_outputs=self.hiddenSize, #hidden
                activation_fn=tf.nn.relu,
                weights_initializer=tf.truncated_normal_initializer(stddev=1.0),
                biases_initializer=tf.zeros_initializer()
            )
            self.L1 = L1= tf.contrib.layers.fully_connected(
                inputs=L0,
                num_outputs=self.hiddenSize, #hidden
                activation_fn=tf.nn.relu,
                weights_initializer=tf.truncated_normal_initializer(stddev=1.0),
                biases_initializer=tf.zeros_initializer()
            )

            #construct a lstmcell ,the size is neuronNum
            cell = tf.contrib.rnn.BasicLSTMCell(self.neuronNum, forget_bias=1.0, state_is_tuple=True)
            #cell =tf.contrib.rnn.DropoutWrapper(lstmcell, output_keep_prob=0.5)
            #cell = tf.contrib.rnn.MultiRNNCell([cell_drop for _ in range(2)], state_is_tuple=True)
           
            
            #系统下一时刻的状态仅由当前时刻的状态产生
            outputnew,statenew = tf.nn.dynamic_rnn(cell,L1,dtype=tf.float32)
            outputs = outputnew[:,self.stepNum-1,:] # 取最后一个step的结果

            softmax_w = tf.get_variable( "softmax_w", [self.neuronNum, 3], dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=1.0))
            softmax_b = tf.get_variable("softmax_b", [3], dtype=tf.float32,initializer=tf.zeros_initializer())
            logits = tf.matmul(outputs, softmax_w) + softmax_b
            self.probs = tf.nn.softmax(logits, name="action")
            # fetch the maximum probability
            #self.action0 = tf.reduce_max(self.probs, axis=1)
            # fetch the index of the maximum probability
            #self.argAction = tf.argmax(self.probs, axis=1)

            self.argAction = tf.reshape(tf.multinomial(tf.log(self.probs),num_samples=1),[-1])
            self.action0 = tf.gather_nd(self.probs,self.actions_taken)

            critic_rew =tf.reshape(self.critic_rewards,[-1,10])
            action = tf.reshape(self.action0,[-1,10])
            self.policyloss = tf.reduce_sum(critic_rew*action,1)
            loss = tf.negative(tf.reduce_mean(cost),name = "loss")
            #print(critic_rew*action)

            #loss,train
            #self.entropy = -tf.reduce_sum(self.probs * tf.log(self.probs), 1, name="entropy")
            #self.lr_update = tf.assign(self.lr,self.new_lr)
            #self.policyloss = policyloss  = tf.log(self.action0)*self.critic_rewards + 0.01 * self.entropy
            #self.policyloss = policyloss  = tf.log(tf.clip_by_value(self.action0,1e-10,1.0))*self.critic_rewards 
            #self.policyloss = policyloss  = tf.log(self.action0)*self.critic_rewards 
            #loss = tf.negative(tf.reduce_sum(policyloss),name="loss")
            #loss = tf.negative(policyloss,name="loss")

            #tf.summary.scalar('actor_loss',tf.abs(loss))
            self.actor_train = tf.train.AdamOptimizer(self.lr).minimize(loss)
            
            #optimizer = tf.train.AdadeltaOptimizer(self.lr)
            #grads_and_vars = optimizer.compute_gradients(loss)
            #capped_gvs = [(tf.clip_by_value(grad,-1,1),var) for grad,var in grads_and_vars]
            #self.actor_train = optimizer.apply_gradients(capped_gvs)
            
            #self.tvars = tvars = tf.trainable_variables() #得到可以训练的参数
            #self.agrads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),5)  #防止梯度爆炸
            #self.agrads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),5)  #防止梯度爆炸
            #optimizer = tf.train.AdamOptimizer(self.lr)
            #self.actor_train = optimizer.apply_gradients(zip(self.agrads, tvars))

    
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
        #all_discounted_rewards =self.discount_rewards(all_rewards,discount_rate) 
        #sum = 0
        #for rew in all_discounted_rewards:
        #    if rew == 0:
        #        sum=sum+1
        #if sum == len(all_discounted_rewards):
        #    return all_discounted_rewards
        #else:
        #    reward_mean = np.mean(all_discounted_rewards)
        #    reward_std = np.std(all_discounted_rewards)
        #    return [(all_discounted_rewards - reward_mean)/reward_std] 


    def learn(self):
        #self.merged = tf.summary.merge_all()
        #self.writer = tf.summary.FileWriter("/home/swy/code/DRL/tbencoder", self.sess.graph) 
        # 5 days
        batchsize=240
        timestep = 10
        epoch=3
        max_epoch=2
        learningrate = 0.1

        trainfalse =True
        if trainfalse:
            
            for j in range(epoch):  
                total=[] 
                sum = 0
                win = 0
                predition = []

                lr_decay = 0.5**max(j+1 - max_epoch,0.0)
                #self.assign_lr(self.sess,learningrate*lr_decay)
                
                for k in range(1):
                    stockList = ['IF1601.CFE.csv', 'IF1602.CFE.csv', 'IF1603.CFE.csv','IF1604.CFE.csv', 'IF1605.CFE.csv', 'IF1606.CFE.csv', 'IF1607.CFE.csv', 'IF1608.CFE.csv', 'IF1609.CFE.csv', 'IF1610.CFE.csv', 'IF1611.CFE.csv', 'IF1612.CFE.csv','IF1701.CFE.csv', 'IF1702.CFE.csv', 'IF1703.CFE.csv']
                    super(lmmodel,self).__init__(stockList[k])
            
                    #每次滑动5000，训练窗口大小为15000,TEST 为顺延的5000，batchsize大小设置为5000
                    #for i in range(0,len(self.state)-batchsize,batchsize):
                    for i in range(0,len(self.state)-batchsize-timestep,240+timestep):

                                trajectory = self.get_trajectory(i,timestep,batchsize)
                                state = trajectory["state"]
                                action = trajectory["action"]
                                actions = []
                                for m in range(len(action)):
                                    newact = [m,action[m]]
                                    actions.append(newact)
                                #returns = self.discount_rewards(trajectory["reward"],0.95)
                                
                                #rewards = returns = trajectory["reward"]
                                returns =trajectory["reward"]
                                reward = np.reshape(returns,(-1,10))
                                rewards = self.discount_and_normalize_rewards(reward,0.95)
                                if rewards == 0 :
                                    print("over")
                                    continue
                                else: 
                                    rewards = np.reshape(rewards,(-1))
                                    action = trajectory["action"]
                                    # print(action)

                                    #统计收益大于0的周数
                                    if np.sum(returns)>0:
                                        win = win +1
                                    sum = sum +1
                                    total.append(np.sum(returns))
                                    #print(np.sum(returns))
                                    
                                    probs, loss,L1= self.sess.run([self.probs,self.policyloss,self.L1],feed_dict={
                                        self.states: state,
                                        self.critic_rewards:rewards,
                                        self.actions_taken:actions
                                    })
                                    #print(np.shape(L1))
                                    #print("L1")
                                    #print(L1[1,2,:])
                                    print("probs")
                                    print(probs)
                                    #print("grads")
                                    #print(agrads)
                                    #print("tvars")
                                    #for ll in range(8):


                                    #    print(np.shape(tvars[ll]))

                                    action,actorResults= self.sess.run([self.argAction,self.actor_train],feed_dict={
                                        self.states: state,
                                        self.critic_rewards:rewards,
                                        self.actions_taken:actions
                                    })
                                    print("action")
                                    print(action)
                                    #print(np.shape(tvars))
                                    
                                   # print("tvar")
                                   # print(tvar)
                                #print(np.sum(loss))

 
                            
                print("total")
                print(np.sum(total))
                print(win/sum)
                plt.figure()
                x_values = range(len(total))
                y_values = total
                plt.plot(x_values, y_values)
                plt.savefig(str(j)+'.png')
                    
        #self.writer.close()


def main():
    os.chdir("/home/swy/code/DRL/Info1/Info/data")
    L=[]
    for files in os.walk("/home/swy/code/DRL/Info1/Info/data"):
        for file in files:
            L.append(file) 


    #if tf.gfile.Exists('/home/swy/code/DRL/tbencoder'):
    #    tf.gfile.DeleteRecursively('/home/swy/code/DRL/tbencoder')
    #tf.gfile.MakeDirs('/home/swy/code/DRL/tbencoder')

    sess= tf.InteractiveSession()
    trainable= True
    if trainable:
        out = lmmodel(sess=sess,FileList=L)
        sess.run(tf.global_variables_initializer())
        out.learn()
        save_path = out.saver.save(sess, '/home/swy/code/DRL/cpencoder/model0601.ckpt')
    else:
        out = lmmodel(sess=sess,FileList=L)
        load_path = out.saver.restore(sess,'/home/swy/code/DRL/cpencoder/model0601.ckpt')
        out.learn()


if __name__ == '__main__':
    main()
    #tf.app.run()














    





    

