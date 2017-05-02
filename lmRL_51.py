

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from Agent import Agent
from Agent2 import Agent2
#import argparse
#import sys

import tensorflow as tf
import math
import argparse
import sys
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data


class lmmodel(Agent2):

    def __init__(self, config,sess):
        super(lmmodel,self).__init__('data/IF1601.CFE.csv', 20, 20,2)
        self.config = config
        self.sess = sess 
        #self.trajecNum=100  #
        #self.batchSize=20   #120 batchSize
        self.inputSize=20  #20features
        self.stepNum=20   #20 price sequence
        self.hiddenSize=40 # fully connected outputs
        self.neuronNum=10
        self.actionsize=3
        #self.stateSize=[self.stepNum]
        
        self.buildNetwork()
        self.saver = tf.train.Saver(tf.global_variables())


    def choose_action(self, state):
        """Choose an action."""
        return self.sess.run(self.argAction, feed_dict={self.states: state,self.seq_length:[self.stepNum]})

    def buildNetwork(self):
        self.states = tf.placeholder(tf.float32,shape=[None,self.stepNum, self.inputSize],name= "states")
        #self.actions_taken = tf.placeholder(tf.float32,shape=[None,None],name= "actions_taken")
        self.critic_feedback = tf.placeholder(tf.float32,shape=[None,None],name= "critic_feedback")
        self.critic_rewards = tf.placeholder(tf.float32,shape=[None,None],name= "critic_rewards")
        self.seq_length= tf.placeholder(tf.int32,[None])

        def lstm_cell(size):
            return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)

        # ActorNetwork
        with tf.variable_scope("actor") :

            L1= tf.contrib.layers.fully_connected(
                inputs=self.states,
                num_outputs=self.hiddenSize, #hidden
                activation_fn=tf.tanh,
                weights_initializer=tf.random_normal_initializer(),
                biases_initializer=tf.zeros_initializer()
            )

            lstmcell = tf.contrib.rnn.BasicLSTMCell(self.neuronNum, forget_bias=1.0, state_is_tuple=True)
            cell = tf.contrib.rnn.MultiRNNCell([lstmcell for _ in range(5)], state_is_tuple=True)
            #state = cell.zero_state(tf.shape(L1)[0], tf.float32)
            #seq_length=[]
            #seq_length=tf.Variable(tf.zeros([tf.shape(L1)[0]]),shape=[])
            #print(seq_length)
            #seq_length=tf.assign(seq_length,self.stepNum)
            #num=tf.cast(tf.shape(L1)[0],tf.int32)
            #for i in range(num):
            #    seq_length.append(self.stepNum)
            
            outputs,states = tf.nn.dynamic_rnn(cell,L1,dtype=tf.float32,sequence_length=self.seq_length)
            outputs=tf.reshape(outputs,[-1,self.neuronNum])
            #outputs=[]
            #with tf.variable_scope("actorScope"):
            #    for i in range(20):                 
            #        (output, state) = cell(L1[:,i,:], state)
            #        outputs.append(output)
            #        tf.get_variable_scope().reuse_variables()
            #print("outputs")
            #print(outputs)
            #outputs=tf.reshape(outputs,[-1,self.neuronNum])




            softmax_w = tf.get_variable( "softmax_w", [10, 3], dtype=tf.float32,initializer=tf.random_normal_initializer())
            softmax_b = tf.get_variable("softmax_b", [3], dtype=tf.float32)
            logits = tf.matmul(outputs, softmax_w) + softmax_b
            self.probs = tf.nn.softmax(logits, name="action")
            self.action0 = tf.reduce_max(self.probs, axis=1)
            self.argAction = tf.argmax(self.probs, axis=1)
            self.action = tf.reshape(self.action0,[-1,self.stepNum,3] )  #action change 

            #loss,train
            self.policyloss =policyloss  = tf.log(self.action0)*(self.critic_rewards-self.critic_feedback)
            loss = tf.negative(tf.reduce_sum(policyloss),name="loss")
            tf.summary.scalar('actor_loss',loss)

            self.actor_train = tf.train.AdamOptimizer(0.01).minimize(loss)


            #tvars = tf.trainable_variables()
            #grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),5)
            #optimizer = tf.train.GradientDescentOptimizer(0.01)
            #self.actor_train = optimizer.apply_gradients(zip(grads, tvars))




        # Critic Network
        with tf.variable_scope("critic") as scopeB:

            self.critic_target = tf.placeholder(tf.float32,name= "critic_target")

            critic_L1= tf.contrib.layers.fully_connected(
                inputs=self.states,
                num_outputs= self.hiddenSize, #hidden
                activation_fn= tf.tanh,
                weights_initializer = tf.random_normal_initializer(),
                biases_initializer = tf.zeros_initializer()
            )
            print("critic states")
            print(self.states)

            lstmcell=tf.contrib.rnn.BasicLSTMCell(self.neuronNum, forget_bias=1.0, state_is_tuple=True)
            cell = tf.contrib.rnn.MultiRNNCell([lstmcell for _ in range(5)], state_is_tuple=True)
            #seq_length=[]
            #num=tf.shape(critic_L1)[0]
            #for i in range(self.stepNum):
            #    seq_length.append(self.stepNum)
            print("critic size")
           # print(self.stateSize)
            print(critic_L1)
            outputs,states = tf.nn.dynamic_rnn(cell,critic_L1,dtype=tf.float32,sequence_length=self.seq_length)

            output=outputs
            #output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, 10])

           # weights = tf.Variable(tf.truncated_normal([28, 10],stddev=1.0 / math.sqrt(float(28))),name='weights')
           # biases = tf.Variable(tf.zeros([10]),name='biases')
           # logits = tf.matmul(cell_output, weights) + biases
            self.critic_value = tf.contrib.layers.fully_connected(
                inputs=output,
                num_outputs= 1, #hidden
                activation_fn= tf.tanh,
                weights_initializer = tf.random_normal_initializer(),
                biases_initializer = tf.zeros_initializer()
            )
            print("critic")
            print(self.critic_value)
            self.critic_value = tf.reshape(self.critic_value,[-1,self.stepNum])

            #loss,train
            self.critic_loss=critic_loss = tf.reduce_mean(tf.square(self.critic_target - self.critic_value) , name ="loss" )
            tf.summary.scalar('critic_loss',self.critic_loss)
            self.critic_train = tf.train.AdamOptimizer(0.01).minimize(critic_loss) #global_step

            #tvar = tf.trainable_variables()
            #self.gr=tf.gradients(critic_loss, tvar)
            #self.grads, _ = tf.clip_by_global_norm(tf.gradients(critic_loss, tvar),5)
            #optimizer = tf.train.GradientDescentOptimizer(0.01)
            #self.critic_train = optimizer.apply_gradients(zip(self.grads, tvar))
    


    def discount_rewards(self, x, gamma):
        """
        Given vector x, computes a vector y such that
        y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
        """
        result = [0 for i in range(len(x))]
        element = 0
        for i in range(len(x)-2, -1, -1):
            element = x[i] + gamma * element
            result[i] = element

        return result


    def learn(self):

        #self.merged = tf.summary.merge_all()
        #self.writer = tf.summary.FileWriter("/home/swy/DRL/writer", self.sess.graph) 

        trajectories = self.get_trajectories()
        all_state=[]
        all_action=[]
        all_returns=[]
        seq_length=[]
        for trajectory in trajectories:
            all_action.append(trajectory["action"] )
            all_state.append(trajectory["state"] )
            all_returns.append(self.discount_rewards(trajectory["reward"], 0.99))
            seq_length.append(self.stepNum)
        print("trastate")
        print(np.shape(all_action))
        print(np.shape(all_returns))
        print(np.shape(all_state))

        #print(self.stateSize)
        #all_returns=all_returns.reshape((-1,20))
        
        qw_new = self.sess.run(self.critic_value,feed_dict={self.states:all_state,self.seq_length:seq_length})
       # qw_new = qw_new.reshape(-1)
        qw_new = tf.reshape(qw_new,[-1,self.stepNum])
        print(qw_new)

        val,ff= self.sess.run([self.critic_train,self.actor_train],feed_dict={
                self.critic_target:all_returns,
                self.states: all_state,
                self.critic_feedback:qw_new,
                self.critic_rewards:all_returns
            })

      



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


  #  mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data',one_hot=True)
  #  train_input,ys = mnist.train.next_batch(100)
   # if tf.gfile.Exists('/home/swy/DRL/writer'):
   #     tf.gfile.DeleteRecursively('/home/swy/DRL/writer')
   # tf.gfile.MakeDirs('/home/swy/DRL/writer')

    config=get_config()
    sess= tf.InteractiveSession()

    out = lmmodel(config=config,sess=sess)
    sess.run(tf.global_variables_initializer())
    out.learn()
    #saver = tf.train.Saver(tf.global_variables())
    #save_path = out.saver.save(sess, '/home/swy/DRL/saver')


   # out = lmmodel(config=config,sess=sess)
   # load_path = out.saver.restore(sess,'/home/swy/DRL/saver')
   # out.learn()
    


if __name__ == '__main__':
    main()
    #tf.app.run()














    





    

