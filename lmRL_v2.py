

#MNIST with a BasicLSTMCell
#try tf.app.run and def

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from Agent_v2 import Agent2
#import argparse
#import sys

import tensorflow as tf
import math
import argparse
import sys
import numpy as np

#from tensorflow.examples.tutorials.mnist import input_data


class lmmodel(Agent2):

    def __init__(self, config,sess):
        super(lmmodel,self).__init__('data/IF1601.CFE.csv', 20, 20, 5000)
        self.config = config
        self.sess =sess
        #self.sess = tf.InteractiveSession()
        #self.trajecNum=100  #
        #self.batchSize=20   #120 batchSize
        self.inputSize=20  #20features
        self.stepNum=20   #20 price sequence
        self.hiddenSize=40 # fully connected outputs
        self.neuronNum=10
        #self.actionsize=3
        self.buildNetwork()
        self.saver = tf.train.Saver(tf.global_variables())
        #init = tf.global_variables_initializer()
        #self.sess.run(init)

    def choose_action(self, state):
        """Choose an action."""
        return self.sess.run(self.argAction, feed_dict={self.states: state})

    def buildNetwork(self):
        self.states = tf.placeholder(tf.float32,shape=[self.stepNum, self.inputSize],name= "states")
        self.actions_taken = tf.placeholder(tf.float32,shape=[None],name= "actions_taken")
        self.critic_feedback = tf.placeholder(tf.float32,shape=[None],name= "critic_feedback")
        self.critic_rewards = tf.placeholder(tf.float32,shape=[None],name= "critic_rewards")

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
            state = cell.zero_state(1, tf.float32)
            
            s_step= tf.unstack(L1) 
            outputs=[]
            with tf.variable_scope("actorScope"):
                for i in s_step:                 
                    ii=tf.reshape(i,[1,-1])
                    (output, state) = cell(ii, state)
                    outputs.append(tf.reshape(output,[-1]))
                    tf.get_variable_scope().reuse_variables()

            #print("output")
            #print(outputs)
            softmax_w = tf.get_variable( "softmax_w", [10, 3], dtype=tf.float32,initializer=tf.random_normal_initializer())
            softmax_b = tf.get_variable("softmax_b", [3], dtype=tf.float32)
            logits = tf.matmul(outputs, softmax_w) + softmax_b
            self.probs = tf.nn.softmax(logits, name="action")
            self.action0 = tf.reduce_max(self.probs, axis=1)
            self.argAction = tf.argmax(self.probs, axis=1)

            #loss,train
            self.policyloss =policyloss  = tf.log(self.action0)*(self.critic_rewards-self.critic_feedback)
            loss = tf.negative(tf.reduce_mean(policyloss),name="loss")
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

            lstmcell=tf.contrib.rnn.BasicLSTMCell(self.neuronNum, forget_bias=1.0, state_is_tuple=True)
            cell = tf.contrib.rnn.MultiRNNCell([lstmcell for _ in range(5)], state_is_tuple=True)
            state = cell.zero_state(1, tf.float32) 
            #(outputs, state) = cell(critic_L1, state)

            ss_step= tf.unstack(critic_L1) 
            outputs=[]
            with tf.variable_scope("criticScope"):
                for i in ss_step:                 
                    ii=tf.reshape(i,[1,-1])
                    (output, state) = cell(ii, state)
                    outputs.append(tf.reshape(output,[-1]))
                    tf.get_variable_scope().reuse_variables()  

            print("critic")
            print(np.shape(outputs))
            output=outputs
            #output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, 10])

           # weights = tf.Variable(tf.truncated_normal([28, 10],stddev=1.0 / math.sqrt(float(28))),name='weights')
           # biases = tf.Variable(tf.zeros([10]),name='biases')
           # logits = tf.matmul(cell_output, weights) + biases
            self.critic_value = tf.contrib.layers.fully_connected(
                inputs=output,
                num_outputs= 1, #hidden
                activation_fn= None,
                weights_initializer = tf.random_normal_initializer(),
                biases_initializer = tf.zeros_initializer()
            )

            #loss,train
            self.critic_loss=critic_loss = tf.reduce_mean(tf.square(self.critic_target - self.critic_value) , name ="loss" )
            tf.summary.scalar('critic_loss',self.critic_loss)
            self.critic_train = tf.train.AdamOptimizer(0.01).minimize(critic_loss) #global_step

            #tvar = tf.trainable_variables()
            #self.gr=tf.gradients(critic_loss, tvar)
            #self.grads, _ = tf.clip_by_global_norm(tf.gradients(critic_loss, tvar),5)
            #optimizer = tf.train.GradientDescentOptimizer(0.01)
            #self.critic_train = optimizer.apply_gradients(zip(self.grads, tvar))
    #self.saver = tf.train.Saver(tf.global_variables())


    def discount_rewards(self,x, gamma):
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
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("/home/swy/code/DRL/tb", self.sess.graph) 
        trajectories = self.get_trajectories()
        i=0
        for trajectory in trajectories:
            i=i+1
            action = trajectory["action"]
            state = trajectory["state"]
            returns = self.discount_rewards(trajectory["reward"],0.99)
            print("returns")
            print(returns)
            qw_new = self.sess.run(self.critic_value,feed_dict={self.states:state})
            qw_new = qw_new.reshape(-1)

            if i%100==0:
                print("num:%d",i)
                print(np.sum(trajectory["reward"]))
           
            summary,criticResults, actorResults = self.sess.run([self.merged,self.critic_train,self.actor_train],feed_dict={
                self.critic_target:returns,
                self.states: state,
                self.actions_taken: action,
                self.critic_feedback:qw_new,
                self.critic_rewards:returns
            })
            self.writer.add_summary(summary,i)
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

    if tf.gfile.Exists('/home/swy/code/DRL/tb'):
        tf.gfile.DeleteRecursively('/home/swy/code/DRL/tb')
    tf.gfile.MakeDirs('/home/swy/code/DRL/tb')

    config=get_config()
    sess= tf.InteractiveSession()

    out = lmmodel(config=config,sess=sess)
    sess.run(tf.global_variables_initializer())
    out.learn()
    #saver = tf.train.Saver(tf.global_variables())
    save_path = out.saver.save(sess, '/home/swy/code/DRL/cp/model.ckpt')


    #out = lmmodel(config=config,sess=sess)
    #load_path = out.saver.restore(sess,'/home/swy/DRL/saver')
    #out.learn()
            #out=sess.run(out.train_step,feed_dict=feed_dict())


if __name__ == '__main__':
    main()
    #tf.app.run()














    





    

