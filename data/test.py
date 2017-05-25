#import csv
#with open('C:\\Users\\Public.dclab-PC\\Desktop\\ML\\trade\\Info\\data\\IF1601.CFE.csv','rb') as csvfile:
#    reader = csv.reader(csvfile)
#    column = [row[2] for row in reader]
#print(column)

import numpy as np


my_matrix = np.loadtxt(open("C:\\Users\\Public.dclab-PC\\Desktop\\ML\\trade\\Info\\data\\IF1601.CFE.csv","rb"),delimiter=",",skiprows=0)  
print(my_matrix[1][5])
print(np.shape(my_matrix))
