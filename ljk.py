import numpy as np
import tensorflow as tf

x=[1,2,3,4,5,6,7,8,9]
mean = np.mean(x)
variance = np.var(x)
print((x-mean)/variance)

print(mean)
print(variance)

#x=tf.Variable(tf.random_normal([4]))
#print(x)
#axis= list(range(len(x.get_shape())-1))
#print(axis)
#mean,variance = tf.nn.moments(x,[0])

#print('fff')
#print(mean)

#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    seed,mean=sess.run([x,mean])
#    print(seed)
#    print(mean)
    



#f = open('/home/swy/code/DRL/data/IF1601.CFE.csv', 'r')

#dataBase = f.readline()
#dataBase = dataBase.split(',')
#dataBase.pop()

#m=20
#state=[]
#tmp=[]
#for i in range(len(dataBase)):
#    dataBase[i] = float(dataBase[i])
#for i in range(1, len(dataBase)):
#    tmp.append(dataBase[i] - dataBase[i-1])

#for i in range(0,len(tmp)-m+1):
#   state.append(tmp[i:i+m]) 

#print(np.shape(state))
#print(tmp[len(tmp)-m:len(tmp)-1])
#print(state[0])
#print(state[len(state)-1])
