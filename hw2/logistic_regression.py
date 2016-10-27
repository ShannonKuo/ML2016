import sys
import math

def calculateAns(data, w_data, spam_data, b, doNotUse, write, valid):
    ans_data = [0.0 for i in range (len(data))]
    
    for i in range (len(data)):
        for k in range (len(data[0])):
            ans_data[i] += w_data[k] * data[i][k]
        ans_data[i] += b
        ans_data[i] = (1 / (1.0 + math.exp(-ans_data[i] + epsilon)))
        if ans_data[i] > 0.5: ans_data[i] = 1
        else : ans_data[i] = 0

    correct_data = 0.0
    for i in  range(len(data)):
        if ans_data[i] == spam_data[i]:
            correct_data += 1.0

    if write == 1:
        #file5 = open("train_ans.csv", 'a')
        #if valid == 1:
        #    file5.write("valid : ")
        #else :
        #    file5.write("train : ")
        #for i in range(len(doNotUse)):
        #    file5.write(str(doNotUse[i]) + " ")
        #file5.write('\n')
 
        print  correct_data/len(data)
        #file5.write(str(correct_data/len(data)))
        #file5.write('\n')


###
file = open(sys.argv[1], 'r')
train = list()
valid = list()
spam_train = list()
spam_valid = list()

feature = [1 for i in range(0, 57, 1)]
doNotUse = [53, 56 ]
#doNotUse = [43, 32, 25, 17, 1,53, 56 ]
for i in range(len(doNotUse)):
    feature[ doNotUse[i] ] = 0

#iteration = 300000
iteration = 10000
v = 0
line_Id = 0
adagrad = 1
mini_batch = 1
batch = 5
pos = 0
validation_start = 3000
epsilon = 1e-20
b = -1
if adagrad == 1 : n0 = 0.01
else : n0 = 0.000001

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

w_train = [0.1 for i in range(0, len(train[0]))]
wsq_train = [1e-20 for i in range(0, len(train[0]))]
n_train = [n0 for i in range(0, len(train[0]))]
n_b     = n0
sig_train = [0.0 for i in range(0, len(train[0]))]
sig_b = 0.0

######training start
for i in range (0, iteration, 1):
    likelihood = 0

    for j in range(len(train[0])):
        sig_train[j] = 0
    sig_b = 0

    for j in range(pos, pos + len(train) / batch):
        f = 0.0
        for k in range(0, len(train[0])):
            f += w_train[k] * train[j][k]
        f += b
        f = (1 / (1.0 + math.exp(-f)))
        for k in range(0, len(train[0])):
            sig_train[k] += -(spam_train[j] - f) * train[j][k] 
        sig_b += -(spam_train[j] - f) 
        if i  == 0:
            likelihood += -((spam_train[j] * math.log(f + epsilon))
                           + ((1.0-spam_train[j]) * math.log(1.0-f+epsilon)))
    if mini_batch == 1 :
        if pos != len(train) / batch * (batch - 1) : pos += len(train) / batch
        else : pos = 0
        
    if adagrad == 1:
        for j in range(0, len(train[0])):
            wsq_train[j] += sig_train[j] ** 2
        for j in range(0, len(train[0])):
            n_train[j] = n0 / ((wsq_train[j]) ** 0.5 + epsilon)
    for j in range(0, len(train[0])):
        w_train[j] -= n_train[j] * sig_train[j]
    
    b -= n_b * sig_b
    #print "sig_train ",sig_train
    if i % 100 == 0:
        print "doNotUse   : ",doNotUse
        print "iteration  : ",i
        print "likelihood : ",likelihood 
        """
        if v == 1:
           print "valid_ans : ",
           calculateAns(valid, w_train, spam_valid, b, doNotUse, 0, 1)
        else :
           print "train_ans : ",
           calculateAns(train, w_train, spam_train, b, doNotUse, 0, 0)
        """
#######calculate ans
if v == 1 : 
   print "valid_final_ans : ",
   calculateAns(valid, w_train, spam_valid, b, doNotUse,1, 1 )
else :
   print "train_final_ans : ",
   calculateAns(train, w_train, spam_train, b, doNotUse,1, 0 )

#######output model
file4 = open(sys.argv[2], "w")
for i in range(len(train[0])):
    file4.write(str(w_train[i]))
    if i != len(train[0]) - 1 : 
        file4.write(",")
    else : file4.write('\n')

file4.write(str(b) + '\n')
file4.write(str(epsilon) + '\n')
for i in range(len(doNotUse)):
    file4.write(str(doNotUse[i]))
    print doNotUse[i]
    if i != len(doNotUse) - 1 : 
        file4.write(",")
         
