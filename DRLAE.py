
#功能是使用三层autoencoder训练中间一层，没有加入l1范数
#autoencoder训练中间一层的参数仅仅作为第一层全连接的初始化参数，在进行bp时候会更新全部参数
#训练数据是1601-1612一年的数据
#训练数据batchsize为100，连续序列读入



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from Agent_v2 import Agent2
from Autoencoder import Autoencoder 
#import argparse
#import sys

import tensorflow as tf
import math
import argparse
import sys
import numpy as np

#from tensorflow.examples.tutorials.mnist import input_data

import os




class lmmodel(Agent2):

    def __init__(self, config,sess,W1,B1,FileList):
        #super(lmmodel,self).__init__('data/IF1602.CFE.csv', 10, 240, 2000)

        self.w=W1
        self.b=B1
        self.L=FileList

        self.config = config
        self.sess =sess
        #self.sess = tf.InteractiveSession()
        #self.trajecNum=100  #
        #self.batchSize=20   #120 batchSize
        self.inputSize=10  #20features
        self.stepNum=100   #20 price sequence
        self.hiddenSize=128 # fully connected outputs
        self.neuronNum=100
        #self.actionsize=3
        self.buildNetwork()
        self.saver = tf.train.Saver(tf.global_variables())
        #init = tf.global_variables_initializer()
        #self.sess.run(init)
    
    #input states sequence, generate the action vector by policy Network
    def choose_action(self, state):  
        """Choose an action."""
        return self.sess.run(self.argAction, feed_dict={self.states: state})

    # build the policy Network and value Network
    def buildNetwork(self):
        self.states = tf.placeholder(tf.float32,shape=[self.stepNum, self.inputSize],name= "states")
        self.actions_taken = tf.placeholder(tf.float32,shape=[None],name= "actions_taken")
        self.critic_feedback = tf.placeholder(tf.float32,shape=[None],name= "critic_feedback")
        self.critic_rewards = tf.placeholder(tf.float32,shape=[None],name= "critic_rewards")
        #self.w1 = tf.Variable(self.w,dtype=tf.float32,name="w1")
        #self.b1 = tf.Variable(self.b,dtype=tf.float32,name="b1")
        #self.w1 = tf.placeholder(tf.float32,shape=[10,100],name="w1")  # autoencoder pretrain w1
        #self.b1 = tf.placeholder(tf.float32,shape=[100],name="b1")    # autoencoder pretrain b1
       

        #def lstm_cell(size):
        #    return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)

        # PolicyNetwork
        with tf.variable_scope("Policy") :
 
            #construct one layer fully_connected Network
            L1=tf.nn.relu(tf.matmul(self.states,self.w)+self.b)
            #L1= tf.contrib.layers.fully_connected(
            #    inputs=self.states,
            #    num_outputs=self.hiddenSize, #hidden
            #    activation_fn=tf.nn.relu,
                #weights_initializer=tf.truncated_normal_initializer(stddev=1.0),
                #biases_initializer=tf.zeros_initializer()
            #    weights_initializer=self.w1,
            #    biases_initializer=self.b1
                #biases_initializer=tf.zeros_initializer()
            #)

            #construct a lstmcell ,the size is neuronNum
            lstmcell = tf.contrib.rnn.BasicLSTMCell(self.neuronNum, forget_bias=1.0, state_is_tuple=True)
            cell=tf.contrib.rnn.DropoutWrapper(lstmcell, output_keep_prob=0.5)
            #lstmcell = tf.contrib.rnn.BasicLSTMCell(self.neuronNum, forget_bias=1.0, state_is_tuple=True,activation=tf.nn.relu)
            #cell_drop=tf.contrib.rnn.DropoutWrapper(lstmcell, output_keep_prob=0.5)
            #construct 5 layers of LSTM
            #cell = tf.contrib.rnn.MultiRNNCell([cell_drop for _ in range(2)], state_is_tuple=True)
           
            # initialize the lstmcell
            state = cell.zero_state(self.stepNum, tf.float32)
            # the feature ft has the length of inputSize
            with tf.variable_scope("actorScope"):
                for i in range(self.inputSize):
                    te=tf.reshape(L1[:,i],[-1,1])
                    (outputs, state) = cell(te, state)
                    #outputs.append(tf.reshape(output,[-1]))
                    tf.get_variable_scope().reuse_variables()


            #state = cell.zero_state(1, tf.float32)
            #s_step= tf.unstack(L1) 

            #outputs=[]
            #with tf.variable_scope("actorScope"):
            #    for i in s_step:                 
            #        ii=tf.reshape(i,[1,-1])

            #        (output, state) = cell(ii, state)

            #        outputs.append(tf.reshape(output,[-1]))
            #        tf.get_variable_scope().reuse_variables()

            #print("outputs")
            #print(outputs)
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
            #self.policyloss =policyloss  = tf.log(self.action0)*(self.critic_rewards-self.critic_feedback)
            self.policyloss =policyloss  = tf.log(self.action0)*self.critic_rewards
            loss = tf.negative(tf.reduce_mean(policyloss),name="loss")
            tf.summary.scalar('actor_loss',loss)
            #self.actor_train = tf.train.AdamOptimizer(0.01).minimize(loss)


            self.atvars=tvars = tf.trainable_variables()
            #print(tvars)
            #self.gg=tf.gradients(loss, tvars)
            self.agrads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),5)
            print(self.agrads)
            optimizer = tf.train.AdamOptimizer(0.01)
            self.actor_train = optimizer.apply_gradients(zip(self.agrads, tvars))




        # Critic Network
        with tf.variable_scope("critic") as scopeB:

            self.critic_target = tf.placeholder(tf.float32,name= "critic_target")
            
            #construct a layer of fully connected network
            critic_L1=tf.nn.relu(tf.matmul(self.states,self.w)+self.b)
            #critic_L1= tf.contrib.layers.fully_connected(
            #    inputs=self.states,
            #    num_outputs= self.hiddenSize, #hidden
            #    activation_fn= tf.nn.relu,
                #weights_initializer = tf.truncated_normal_initializer(stddev=1.0),
                #biases_initializer = tf.zeros_initializer()
            #    weights_initializer=self.w,
            #    biases_initializer=self.b
                #biases_initializer = tf.zeros_initializer()
            #)
            #construct 5 layers of lstm
            cell=tf.contrib.rnn.BasicLSTMCell(self.neuronNum, forget_bias=1.0, state_is_tuple=True)
            #lstmcell=tf.contrib.rnn.BasicLSTMCell(self.neuronNum, forget_bias=1.0, state_is_tuple=True,activation=tf.nn.relu)
            #cell_drop=tf.contrib.rnn.DropoutWrapper(lstmcell, output_keep_prob=0.5)
            #cell = tf.contrib.rnn.MultiRNNCell([cell_drop for _ in range(2)], state_is_tuple=True)


            state = cell.zero_state(self.stepNum, tf.float32)

            # a feature has a length of inputSize
            with tf.variable_scope("criticScope"):
                for i in range(self.inputSize):
                    cellinput=tf.reshape(critic_L1[:,i],[-1,1])
                    (output, state) = cell(cellinput, state)
                    #outputs.append(tf.reshape(output,[-1]))
                    tf.get_variable_scope().reuse_variables()



            #state = cell.zero_state(1, tf.float32) 
            #ss_step= tf.unstack(critic_L1) 
            #outputs=[]
            #with tf.variable_scope("criticScope"):
            #    for i in ss_step:                 
            #        ii=tf.reshape(i,[1,-1])
            #        (output, state) = cell(ii, state)
            #        outputs.append(tf.reshape(output,[-1]))
            #        tf.get_variable_scope().reuse_variables() 
            #output=outputs 

            #print("critic")
            #print(np.shape(outputs))
         
            #output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, 10])

           # weights = tf.Variable(tf.truncated_normal([28, 10],stddev=1.0 / math.sqrt(float(28))),name='weights')
           # biases = tf.Variable(tf.zeros([10]),name='biases')
           # logits = tf.matmul(cell_output, weights) + biases
            self.critic_value = tf.contrib.layers.fully_connected(
                inputs=output,
                num_outputs= 1, #hidden
                activation_fn= None,
                weights_initializer = tf.truncated_normal_initializer(stddev=1.0),
                biases_initializer = tf.zeros_initializer()
            )

            #loss,train
            self.critic_loss=critic_loss = tf.reduce_mean(tf.square(self.critic_target - self.critic_value) , name ="loss" )
            tf.summary.scalar('critic_loss',self.critic_loss)
            #self.critic_train = tf.train.AdamOptimizer(0.01).minimize(critic_loss) #global_step

            self.ctvar=tvar = tf.trainable_variables()
            #self.gr=tf.gradients(critic_loss, tvar)
            self.cgrads, _ = tf.clip_by_global_norm(tf.gradients(critic_loss, tvar),5)
            optimizer = tf.train.AdamOptimizer(0.01)
            self.critic_train = optimizer.apply_gradients(zip(self.cgrads, tvar))


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


    def learn(self):
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("/home/swy/code/DRL/tbencoder", self.sess.graph) 
        #trajectories = self.get_trajectories()
        #i=0
        #for trajectory in trajectories:
        # loop 10000 times, each time get a trajectory
        batchsize=100
        epoch=10
        for j in range(epoch):
            for k in range(12):
                super(lmmodel,self).__init__(self.L[2][k], 10, 100, 2000)
                #print("haha")

                for i in range(int(np.floor(len(self.dataBase)/batchsize))):
                    trajectory = self.get_trajectory(i)
                    action = trajectory["action"]
                    state = trajectory["state"]
                    #returns = self.discount_rewards(trajectory["reward"],0.99)
                    returns = trajectory["reward"]
                    tf.summary.scalar('return',np.sum(trajectory["reward"]))

                    qw_new = self.sess.run(self.critic_value,feed_dict={self.states:state})
                    qw_new = qw_new.reshape(-1)


                    action3,loss,action2=self.sess.run([self.action0,self.policyloss,self.argAction],feed_dict={
                        self.critic_target:returns,
                        self.states: state,
                        self.actions_taken: action,
                        self.critic_feedback:qw_new,
                        self.critic_rewards:returns})
                   
                    if i%100==0:
                        #print("num:%d",i)
                        print(np.sum(trajectory["reward"]))
                        #print("loss")
                        print(action3)
                        print(loss)
                        print(action2)
                        #print(trajectory["reward"])
                        #print(action)
              
                
                    summary,criticResults, actorResults = self.sess.run([self.merged,self.critic_train,self.actor_train],feed_dict={
                        self.critic_target:returns,
                        self.states: state,
                        self.actions_taken: action,
                        self.critic_feedback:qw_new,
                        self.critic_rewards:returns
                    })

                    
                    #print("grads")
                    #print(gg)
                    self.writer.add_summary(summary,(k+1)*(i+1)*(j+1))


        #for i in range(10000):
        #    trajectory = self.get_trajectory()
        #    action = trajectory["action"]
        #    state = trajectory["state"]
        #    returns = self.discount_rewards(trajectory["reward"],0.99)

        #    qw_new = self.sess.run(self.critic_value,feed_dict={self.states:state})
        #    qw_new = qw_new.reshape(-1)

        #    if i%100==0:
        #        print("num:%d",i)
        #        print(np.sum(trajectory["reward"]))
        #        print(trajectory["reward"])
        #        print(action)
           
        #    summary,criticResults, actorResults = self.sess.run([self.merged,self.critic_train,self.actor_train],feed_dict={
        #        self.critic_target:returns,
        #        self.states: state,
        #        self.actions_taken: action,
        #        self.critic_feedback:qw_new,
        #        self.critic_rewards:returns
        #    })
        #    self.writer.add_summary(summary,i)
            #print (criticResults, actorResults)
        self.writer.close()



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
    os.chdir("/home/swy/code/DRL/autoencoder_models/data")
    L=[]
    for files in os.walk("/home/swy/code/DRL/autoencoder_models/data"):
        for file in files:
            L.append(file) 
    #autoencoder pretrain w1, b1
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




    if tf.gfile.Exists('/home/swy/code/DRL/tbencoder'):
        tf.gfile.DeleteRecursively('/home/swy/code/DRL/tbencoder')
    tf.gfile.MakeDirs('/home/swy/code/DRL/tbencoder')

    config=get_config()
    sess= tf.InteractiveSession()

    out = lmmodel(config=config,sess=sess,W1=w,B1=b,FileList=L)
    sess.run(tf.global_variables_initializer())
    out.learn()
    saver = tf.train.Saver(tf.global_variables())
    save_path = out.saver.save(sess, '/home/swy/code/DRL/cpencoder/model.ckpt')


    #out = lmmodel(config=config,sess=sess)
    #load_path = out.saver.restore(sess,'/home/swy/code/DRL/cp/model.ckpt')
    #out.learn()
            #out=sess.run(out.train_step,feed_dict=feed_dict())


if __name__ == '__main__':
    main()
    #tf.app.run()














    





    

