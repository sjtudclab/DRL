import numpy as np
import tensorflow as tf
"""
 m:ft,feature
 batchSize: a trajectory
 batchLength: trajectories
"""

class Agent3(object):

    def __init__(self, fileName, m, batchSize, trajecNum):
        self.action_space = [-1, 0, 1]
        self.m = m
        self.batchSize = batchSize
        self.trajecNum = trajecNum
        self.state = []
        self.statenew = []

        f = open(fileName, 'r')

        self.dataBase = f.readline()
        self.dataBase = self.dataBase.split(',')
        self.dataBase.pop()
      
        self.diff = []
        for i in range(len(self.dataBase)):
            self.dataBase[i] = float(self.dataBase[i])
        for i in range(1, len(self.dataBase)):
             self.diff.append(self.dataBase[i] - self.dataBase[i-1])

        mean = np.mean(self.diff)
        variance = np.var(self.diff)
        #mean,variance = tf.nn.moments(self.diff,[0])
        #self.diff1=tf.div(tf.sub(self.diff,mean),variance)
        self.diff1 = (self.diff - mean)/variance
        for i in range(0,len(self.diff1)-m+1):
            self.statenew.append(self.diff1[i:i+m])


        for i in range(0,len(self.diff)-m+1):
            self.state.append(self.diff[i:i+m])



         
        #for i in range(self.m-1, len(self.dataBase)):
        #    state_tmp = self.dataBase[i-m+1:i+1] 
        #    self.state.append(state_tmp)

        #self.state = self.state[m-1:]



    def choose_action(self,state):
        pass
       # return np.random.randint(-1,2)




    def get_trajectory(self):
        index = np.random.randint(0, len(self.state)-self.batchSize+1)
        state = self.state[index:index+self.batchSize]
        statenew = self.statenew[index:index+self.batchSize]

        action = self.choose_action(statenew)

        action = action -1
        #print(action)
        rewards = [float(0)]
        rews = [float(0)]
        for i in range(1, self.batchSize):
            rew = action[i-1] * state[i][-1] - 1 * abs(action[i]-action[i-1])
            reward = action[i-1] * statenew[i][-1] - 1 * abs(action[i]-action[i-1])
            rewards.append(reward)
            rews.append(rew)
        #print(rewards)

        return {"reward":rewards,
                "state": state,
                "action": action,
                "rews":rews
                }

    def get_trajectories(self):
        #
        index=10
        trajectories = []
        i=0
        #while i < self.trajecNum and index<=len(self.state)-self.batchSize+1:
        while i < self.trajecNum:
            i += 1
            trajectory = self.get_trajectory()
            index +=1
            trajectories.append(trajectory)
        return trajectories
