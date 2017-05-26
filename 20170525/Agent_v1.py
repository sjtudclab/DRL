import numpy as np
import matplotlib.pyplot as plt
"""
 m:ft,feature
 batchSize: a trajectory
 batchLength: trajectories
"""

class Agent1(object):

    def __init__(self, fileName, m):
        self.action_space = [-1, 0, 1]
        self.m = m

        
        #跳过第一行
        f_matrix = np.loadtxt(open(fileName,'rb'),delimiter=',',skiprows=1)
        self.dataBase = f_matrix[:,4]
      
        self.diff = []
        #按照价格序列
        for i in range(len(self.dataBase)):
            self.dataBase[i] = float(self.dataBase[i])
        for i in range(1, len(self.dataBase)):
             self.diff.append(self.dataBase[i] - self.dataBase[i-1])
             
      
        self.state = []
        for i in range(0,len(self.diff)-m):
            #state由前m个价差表示
            self.state.append(self.diff[i:i+m])

       #state 由h,l,o,c,volume,amt,chg,pct_chg,oi,MA....
        befor = 2
        for i in range(len(self.diff)-(m-befor)):
            #newstate =np.hstack((f_matrix[i+(m-10),1:11],self.diff[i:i+m-10]))
            #newstate = np.hstack((f_matrix[i+(m-befor),5],f_matrix[i+(m-befor),7],f_matrix[i+(m-befor)+1,9],self.diff[i:i+m-befor]))
            newstate = np.hstack((f_matrix[i+(m-befor),4],f_matrix[i+(m-befor),5],self.diff[i:i+m-befor]))




    def choose_action(self,state):
        pass
       # return np.random.randint(-1,2)

    def get_state(self,i):
        #index = np.random.randint(0, len(self.state)-self.batchSize+1)
        index = i*100
        state = self.state[index:index+self.batchSize]
        return state

    def get_reward(self,state,action):
        #rewards=[float(0)]
        rewards = []
        #print(np.shape(state))
        #print(np.shape(action))
        #print(len(action))
        state=np.reshape(state,[-1,self.m])
        action=np.reshape(action,[-1])
        action = action - 1
        #print(np.shape(state))
        #print(np.shape(action))
        for i in range(len(action)):
            if i == 0 :
                reward = -1*abs(action[i])
                #reward = 0
            else:
                reward=action[i-1]*state[i][-1]-1*abs(action[i]-action[i-1])
                #reward=action[i-1]*state[i][-1]
            rewards.append(reward)
        return rewards


    def get_trajectory(self,i,timestep,batchsize):

        index = i*(timestep+batchsize) #batchstate的起点
        batchstate = []
        for j in range(batchsize):
            newstate = self.state[index+j*timestep:index+j*timestep+timestep] #长度为timestep
            batchstate.append(newstate)

        action = self.choose_action(batchstate)
        #将action转换为-1,0,1
        action = action -1
        state = np.reshape(batchstate,(-1,self.m))
        #在状态0时刻，不产生reward，但是当产生1，或者-1的时候，会产生手续费
        rewards = []
        for i in range(len(action)):
            if i==0:

                rew = - 1* abs(action[i])

            else:
                rew = action[i-1] * state[i][-1] - 1* abs(action[i]-action[i-1])

            rewards.append(rew)

        return {"reward":rewards,
                "state": batchstate,
                "action": action
                }

    