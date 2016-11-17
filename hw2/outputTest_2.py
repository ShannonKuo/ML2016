import sys
import math
import numpy as np

file  = open(sys.argv[1], "r") ##read model
file2 = open(sys.argv[2], "r") ##read test data
file3 = open(sys.argv[3], "w") ##write ans


test = list()
line_Id = 0

avg_0 = list()
avg_1 = list()
sig = list()
doNotUse = list()
P_0 = 0.0
P_1 = 0.0
pi = 3.14156
for line in file.readlines():
    line_Id += 1
    line = line.strip().split(',')
    if line_Id == 1 : 
       N0 = int(line[0])
    if line_Id == 2 : 
       N1 = int(line[0])
    if line_Id == 3 : 
        avg_0 = line
    if line_Id == 4 : 
        avg_1 = line
    if line_Id == 5 :
        for i in range(len(line)):
            doNotUse = line
    if line_Id >= 6 :
        sig.append(line)

feature = [1 for i in range(0, 57, 1)]
for i in range(len(doNotUse)):
    feature[int(doNotUse[i])] = 0

for line in file2.readlines():
    line_Id += 1
    if not line: break
    line = line.strip().split(',')
    tmp = list()
    for i in range(len(line) - 1):
        if feature[i] == 1: 
            tmp.append(float(line[i+1]))
    test.append( tmp )

avg_00 = np.zeros((len(test[0]), 1)) 
avg_11 = np.zeros((len(test[0]), 1)) 
sigma = np.zeros((len(test[0]),len(test[0])))

for i in range(len(test[0])):
    avg_00[i] = float(avg_0[i])
    avg_11[i] = float(avg_1[i])

for i in range(len(test[0])):
    for j in range(len(test[0])):
        sigma[i][j] = float(sig[i][j])


x = [0 for i in range(len(test))]
for i in range(len(test)):
    x[i] = np.zeros((len(test[0]), 1))
for i in range(len(test)):
    for j in range(len(test[0])):
        x[i][j] = float(test[i][j])

ans_test = [0.0 for i in range(len(test))]

for i in range (len(test)): 
    """
    exp = math.exp(-0.5*np.dot( np.dot((x[i] - avg_00).T , np.linalg.inv(sigma)),  (x[i] - avg_00)))     
    f_0 = 1.0/((2*pi) ** (1/2)) / (np.linalg.norm(sigma) ** 0.5) * exp 
    exp = math.exp(-0.5*np.dot( np.dot((x[i] - avg_11).T , np.linalg.inv(sigma)),  (x[i] - avg_11)))     
    f_1 = 1.0/((2*pi) ** (1/2)) / (np.linalg.norm(sigma) ** 0.5) * exp 
    #print f_0," ", P_0," ", f_1," ", P_1
    ans_test[i] = f_0 * P_0 / (f_0 * P_0 + f_1 * P_1)
    if math.isnan(ans_test[i]):
        print np.dot( np.dot((x[i] - avg_11).T , np.linalg.inv(sigma)),  (x[i] - avg_11))
    #print ans_test[i]
    """
    w_T = np.dot((avg_00 - avg_11).T, np.linalg.inv(sigma))
    b = -0.5 * np.dot(np.dot(avg_00.T, np.linalg.inv(sigma)), avg_00) +0.5 * np.dot(np.dot(avg_11.T, np.linalg.inv(sigma)), avg_11) + math.log(N0, N1)
    temp = np.dot(w_T, x[i]) + b
    ans_test[i] = 1/(1+math.exp(-temp)) 
    if math.isnan(ans_test[i]) : 
        print "id_wrong",i," "


    if ans_test[i] > 0.5: ans_test[i] = 0
    else : ans_test[i] = 1 
#######calculate ans
file3.write("id,label"+'\n')
for i in range(len(test)):
    file3.write(str(i+1))
    file3.write(',')
    file3.write(str(ans_test[i]))
    file3.write('\n')

  
