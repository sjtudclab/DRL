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
        
        #跳过第一行,跳过前面包含nan数据的几行
        f_matrix = np.loadtxt(open(fileName,'rb'),delimiter=',',skiprows=5)
        
        #价差，涨跌
        self.diff = f_matrix[:,7]
        self.price = f_matrix[:,4]
       
        self.state=[]
        for i in range(len(self.diff)):
            #每个时刻状态由（close，volume）选取10个状态
            state =f_matrix[i,1:11]

            self.state.append(state)
        
        #计算每一列的均值和标准差
        self.mean = np.mean(self.state,axis = 0)
        self.std = np.std(self.state,axis= 0)
    
        #self.state = [self.compute_normal(state) for state in self.state]
 
    #特征数据归一化处理
    def compute_normal(self,state):
        for i in range(len(state)):
            state[i]=(state[i]-self.mean[i])/self.std[i]
        return state

    def get_trajectory(self,index,timeStep,batchSize): 
        batch = []
        #st = index
        st = np.random.randint(0,len(self.state)-batchSize-timeStep+1)
        for i in range(batchSize):
            one = []
            for j in range(timeStep):
                one.append(self.state[st+j])
            batch.append(one)
            st = st + 1
  
        #！！！当前时刻状态由前timestep状态预测得到！！！
        action0 = self.choose_action(batch)
        action = action0 - 1

        #文章中的定义reward
        #在状态0时刻，不产生reward，但是当产生1，或者-1的时候，会产生手续费
        rewards = [] 
        diff = self.diff[index+timeStep:index+timeStep+batchSize] #不包含最后一个
        #diff = self.diff[index+timeStep-1:index+timeStep+batchSize-1]

        if index%20== 0:

            plt.figure()
            y=self.price[index+timeStep-1:index+timeStep+batchSize-1]
            x = range(len(action))
            #for m in range(len(action)):
            #    y.append(state[m][-1])
            midindex = []
            midvalue = []
            buyindex = []
            buyvalue = []
            sellindex = []
            sellvalue = []
            #print(action)
            for m in range(len(action)):
                if m == 0 :
                    if action[m] == 0:
                        continue
                    if action[m]==1:
                        buyindex.append(m)
                        buyvalue.append(y[m])
                    else:
                        sellindex.append(m)
                        sellvalue.append(y[m])
                else:
                    if action[m]!=action[m-1]:
                        if action[m]==1:
                            buyindex.append(m)
                            buyvalue.append(y[m])
                        if action[m] == -1:
                            sellindex.append(m)
                            sellvalue.append(y[m])
                        if action[m] == 0:
                            midindex.append(m)
                            midvalue.append(y[m])
            plt.plot(y)
            plt.plot(buyindex,buyvalue,"k^")
            plt.plot(sellindex,sellvalue,"gv")
            plt.plot(midindex,midvalue,"y<")
            #plt.show()
            plt.savefig(str(index)+'.png')





        for i in range(len(action)):
            if i==0:

                rew = - 1* abs(action[i])
                
            else:
                rew = action[i-1] * diff[i] - 1* abs(action[i]-action[i-1])

            rewards.append(rew)

        return {"reward":rewards,
                "state": batch,
                "action": action0  #没有-1，是index
                }

    def choose_action(self,state):
        pass
       # return np.random.randint(-1,2)


  