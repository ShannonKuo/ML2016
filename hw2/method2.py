import sys
import math
import numpy as np

def calculateAns(data, spam_data, N0, N1, avg_00, avg_11, sigma, length):
    pi = 3.14159
    ans_data = [0.0 for i in range (length)]
    """
    k = np.matrix(np.eye(np.shape(data[i])[0]) * 1e-10)
    print np.linalg.inv(sigma_1+k)
    for i in range (length): 
        exp0 = math.exp(-0.5*np.dot( np.dot((data[i] - avg_00).T , np.linalg.inv(sigma_0 + k)), (data[i] - avg_00)))     
        f_0 = 1.0/((2*pi) ** (1/2)) / (np.linalg.norm(sigma_0) ** 0.5) * exp0 
        exp1 = math.exp(-0.5*np.dot( np.dot((data[i] - avg_11).T , np.linalg.inv(sigma_1 + k)), (data[i] - avg_11)))     
        f_1 = 1.0/((2*pi) ** (1/2)) / (np.linalg.norm(sigma_1) ** 0.5) * exp1 
        ans_data[i] = f_0 * P_0 / (f_0 * P_0 + f_1 * P_1)
        if math.isnan(ans_data[i]) : 
            print "id",i," "
    """ 
    for i in range(length):
        w_T = np.dot((avg_00 - avg_11).T, np.linalg.inv(sigma))
        b = -0.5 * np.dot(np.dot(avg_00.T, np.linalg.inv(sigma)), avg_00) +0.5 * np.dot(np.dot(avg_11.T, np.linalg.inv(sigma)), avg_11) + math.log(N0, N1)
        temp = np.dot(w_T, data[i]) + b
        ans_data[i] = 1/(1+math.exp(-temp)) 
        if math.isnan(ans_data[i]) : 
            print "id_wrong",i," "

        if ans_data[i] > 0.5: ans_data[i] = 0
        else : ans_data[i] = 1   

    correct_data = 0.0
    for i in  range(length):
        if ans_data[i] == spam_data[i]:
            correct_data += 1.0

    print  correct_data/len(data)


###
file = open(sys.argv[1], 'r')
train = list()
valid = list()
spam_train = list()
spam_valid = list()

feature = [1 for i in range(0, 57, 1)]
doNotUse = [56]
for i in range(len(doNotUse)):
    feature[ doNotUse[i] ] = 0

v = 0
line_Id = 0
validation_start = 3000
epsilon = 1e-20

for line in file.readlines():

    line_Id += 1
    if not line: break
    line = line.strip().split(',')
    tmp = list()
    for i in range (len(line) - 2 ):
        if feature[i] == 1:
            tmp.append(float(line[i+1]))
    if v == 1 and line_Id > (validation_start) : 
        valid.append( tmp )
        spam_valid.append(float(line[len(line)-1]))
    else :
        train.append( tmp )
        spam_train.append(float(line[len(line)-1]))
######################################
P0 = 0.0
P1 = 0.0
C0 = 0
C1 = 0
pi = 3.1415926

avg_0 = np.zeros((len(train[0]),1))
avg_1 = np.zeros((len(train[0]),1))
sig_0 = np.zeros((len(train[0]), len(train[0]))) 
sig_1 = np.zeros((len(train[0]), len(train[0]))) 
for i in range(len(train)):
    if spam_train[i] == 0:
        C0 += 1
    else :
        C1 += 1
P0 = float(C0) / len(train)
P1 = float(C1) / len(train)

x_0 = [0 for i in range(C0)]
x_1 = [0 for i in range(C1)]
x   = [0 for i in range(len(train))]
for i in range(C0):
    x_0[i] = np.zeros((len(train[0]),1))
for i in range(C1):
    x_1[i] = np.zeros((len(train[0]),1))
for i in range(len(train)):
    x[i]   = np.zeros((len(train[0]),1))

cnt0 = 0
cnt1 = 0
for i in range(len(train)):
    if spam_train[i] == 0:
        for j in range(len(train[0])):
             x_0[cnt0][j] = train[i][j]
        cnt0 += 1
    else :
        for j in range(len(train[0])):
             x_1[cnt1][j] = train[i][j]
        cnt1 += 1
    for j in range(len(train[0])):
        x[i][j] = train[i][j]
 
for i in range(C0):
    avg_0 += x_0[i] / C0
for i in range(C1):
    avg_1 += x_1[i] / C1



for i in range(C0):
    sig_0 += np.dot(((x_0[i] - avg_0)), (x_0[i] - avg_0).T) / C0
for i in range(C1):
    sig_1 += np.dot(((x_1[i] - avg_1)), (x_1[i] - avg_1).T) / C1
sigma = sig_0 * C0 / (C0 + C1) + sig_1 * C1 / (C0 + C1)
#######calculate ans
print "train_final_ans : ",
calculateAns(x, spam_train, C0, C1, avg_0, avg_1, sigma, len(train))
if v == 1 : 
    print "valid_final_ans : ",
    calculateAns(x, spam_train, C0, C1, avg_0, avg_1, sigma, len(x))


#######output model
file4 = open(sys.argv[2], "w")
avg_0 = avg_0.astype(np.float)

file4.write(str(C0) + '\n')
file4.write(str(C1) + '\n')
for i in range(len(train[0])):
    file4.write(str(avg_0[i][0]))
    if i != len(train[0]) - 1 : 
        file4.write(",")
file4.write('\n')

for i in range(len(train[0])):
    file4.write(str(avg_1[i][0]))
    if i != len(train[0]) - 1 : 
        file4.write(",")
file4.write('\n')

for i in range(len(doNotUse)):
    file4.write(str(doNotUse[i]))
    if i != len(doNotUse) - 1 : 
        file4.write(",")
file4.write('\n')

for i in range(len(sigma)):
    for j in range(len(sigma[0])):
        file4.write(str(sigma[i][j]))
        if j != len(sigma[0]) - 1:
            file4.write(',')
    file4.write('\n')
