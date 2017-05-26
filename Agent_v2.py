import numpy as np
import matplotlib.pyplot as plt
"""
 m:ft,feature
 batchSize: a trajectory
 batchLength: trajectories
"""

class Agent2(object):

    def __init__(self, fileName, m, batchSize, trajecNum):
        self.action_space = [-1, 0, 1]
        self.m = m
        self.batchSize = batchSize
        self.trajecNum = trajecNum

        
        #跳过第一行
        f_matrix = np.loadtxt(open(fileName,'rb'),delimiter=',',skiprows=1)
        self.dataBase = f_matrix[:,4]
        

        #print(self.dataBase)

        #f = open(fileName, 'r')
        #self.dataBase = f.readline()
        #self.dataBase = self.dataBase.split(',')
        #self.dataBase.pop()
        #print(len(self.dataBase))
      
        self.diff = []
        #按照价格序列
        for i in range(len(self.dataBase)):
            self.dataBase[i] = float(self.dataBase[i])
        for i in range(1, len(self.dataBase)):
             self.diff.append(self.dataBase[i] - self.dataBase[i-1])
             
      
        #对数据进行归一化处理
        self.statenew = []
        mean = np.mean(self.diff)
        variance = np.var(self.diff)
        self.diff1 = (self.diff - mean)/variance
        for i in range(0,len(self.diff1)-m+1):
            self.statenew.append(self.diff1[i:i+m])

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
            
            #self.state.append(newstate)
            #if i == 0 :
            #    print(newstate)
            #    print(np.shape(newstate))





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


    #def get_trajectory(self,i,test,train):
    def get_trajectory(self,i):
        #index = np.random.randint(0, len(self.state)-self.batchSize+1)
        #state = self.state[index:index+self.batchSize]
        #state=[]
        #for item in self.state[index:index+self.batchSize]:
        #    state.append(item)
        #for item in self.dataBase[index:index+self.batchSize]:
        #    state.append(item)
      
        #index = i*100
        index = i
        state = self.state[index:index+self.batchSize]
        print("state")
        action0 = self.choose_action(state)
        #print("action")
        print(action0)
        #将action转换为-1,0,1
        action = action0 -1
        print("action")

        #重新定义reward
        #状态s0，采取动作a0，状态转移为s1，产生reward：p1-p0，a0产生的代价是，0到a0.最后一个状态产生reward为0
        #rewards=[]
        #for i in range(self.batchSize-1):
        #    if i == 0:
        #        reward = action[i]*state[i+1][-1]-1*abs(action[i])
                #reward = action[i]*state[i+1][-1]
        #    else:
        #        reward = action[i]*state[i+1][-1]-1*abs(action[i]-action[i-1])
                #reward = action[i]*state[i+1][-1]
        #    rewards.append(reward)
        #rewards.append(float(0))

        #文章中的定义reward
        #在状态0时刻，不产生reward，但是当产生1，或者-1的时候，会产生手续费
        rewards = []
        for i in range(self.batchSize):
            if i==0:
                #test的时候，两个bantch首位相接
                #if test:
                   #print("start")
                   #print(self.start)
                #   rew = self.start * state[i][-1]- 1* abs(action[i]-self.start)
                #if train:
                   #print("begin")
                   #print(self.begin)
                #   rew = self.begin * state[i][-1]- 1* abs(action[i]-self.begin)
                rew = - 1* abs(action[i])
                
                #rew=0
            else:
                rew = action[i-1] * state[i][-1] - 1* abs(action[i]-action[i-1])
                #rew = action[i-1] * state[i][-1] 
            rewards.append(rew)
        
        #print(rewards)

        return {"reward":rewards,
                "state": state,
                "action": action0
                }

    def random_trajectory(self,i):
      
        index = np.random.randint(0, len(self.state)-self.batchSize+1)
        state = self.state[index:index+self.batchSize]

        #action = np.random.randint(-1,2,size=100)
        #将action转换为-1,0,1
        action = self.choose_action(state)
        action = action -1
       

        #重新定义reward
        #状态s0，采取动作a0，状态转移为s1，产生reward：p1-p0，a0产生的代价是，0到a0.最后一个状态产生reward为0
        rewards=[]
        for i in range(self.batchSize):
            if i == 0:
                #reward = action[i]*state[i+1][-1]-1*abs(action[i])
                #reward = state[i+1][-1]
                reward = - 1* abs(action[i])
            else:
                #reward = action[i]*state[i+1][-1]-1*abs(action[i]-action[i-1])
                #reward = state[i+1][-1]
                rew = action[i-1] * state[i][-1] - 1* abs(action[i]-action[i-1])
            rewards.append(reward)
        #rewards.append(float(0))

        return {"reward":rewards,
                "state": state,
                "action": action
                }

    def get_trajectories(self):
        #
        index=10
        trajectories = []
        i=0
        #while i < self.trajecNum and index<=len(self.state)-self.batchSize+1:
        while i < self.trajecNum:
            i += 1
            trajectory = self.get_trajectory(i)
            index +=1
            trajectories.append(trajectory)
        return trajectories
