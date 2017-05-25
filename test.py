from Agent import Agent
import numpy as np

#testAgent = Agent('data/IF1601.CFE.csv', 3, 5, 3)
#trajectories=testAgent.get_trajectories()
#print(trajectories)
#all_state=[]
#for trajectory in trajectories:
#    all_state.append(trajectory["state"] )

#print(all_state)


trainData = np.array([float(i) for i in range(1000)]).reshape(-1, 1).tolist()
print(trainData)

print(trainData[1:10])
print(np.shape(trainData[1:10]))