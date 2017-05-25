import numpy as np
import matplotlib.pyplot as plt
"""
 m:ft,feature
 batchSize: a trajectory
 batchLength: trajectories
"""

class Agent(object):

    def __init__(self, fileName):
        self.action_space = [-1, 0, 1]
        #self.batchSize = batchSize
        #self.timeStep = timeStep
        
        #跳过第一行
        f_matrix = np.loadtxt(open(fileName,'rb'),delimiter=',',skiprows=1)
        
        #价差
        self.diff = f_matrix[:,7]
       
        self.state=[]
        for i in len(self.diff):
            #每个时刻状态由（close，volume）
            state = []
            state.append(f_matrix[i,4])
            state.append(f_matrix[i,5])
            self.state.append(state)

    def get_trajectory(self,index,timeStep,batchSize): 
        batch = []
        st = index
        for i in range(batchSize):
            one = []
            for j in range(timeStep):
                one.append(self.state[st+j])
            batch.append(one)
            st = st + 1
        #batch size = [batchSize,timeStep,2]
        print("state")
        print(batch)

        action = self.choose_action(batch)
        action = action - 1

        #文章中的定义reward
        #在状态0时刻，不产生reward，但是当产生1，或者-1的时候，会产生手续费
        rewards = []
        diff = self.diff[i+timeStep:i+timeStep+batchSize-1]
        for i in range(action):
            if i==0:

                rew = - 1* abs(action[i])
                
            else:
                rew = action[i-1] * diff[i] - 1* abs(action[i]-action[i-1])

            rewards.append(rew)

        return {"reward":rewards,
                "state": batch,
                "action": action
                }

    def choose_action(self,state):
        pass
       # return np.random.randint(-1,2)


  