import sys
import math

file  = open(sys.argv[1], "r") ##read model
file2 = open(sys.argv[2], "r") ##read test data
file3 = open(sys.argv[3], "w") ##write ans


test = list()
w_train = list()
line_Id = 0
epsilon = 0
for line in file.readlines():
    line_Id += 1
    line = line.strip().split(',')
    print line
    if line_Id == 1 : 
        w_train = line
    if line_Id == 2 : 
        b = float(line[0])
    if line_Id == 3 :
        epsilon = float(line[0])
    if line_Id == 4 :
        doNotUse = line

print len(w_train)
print doNotUse
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

ans_test = [0.0 for i in range(len(test))]
for i in range (len(test)):
    for k in range (len(test[0])):
        ans_test[i] += float(w_train[k]) * test[i][k]
    ans_test[i] += b
    ans_test[i] = (1 / (1.0 + math.exp(-ans_test[i] + epsilon)))
    if ans_test[i] > 0.5: ans_test[i] = 1
    else : ans_test[i] = 0

file3.write("id,label"+'\n')
for i in range(len(test)):
    file3.write(str(i+1))
    file3.write(',')
    file3.write(str(ans_test[i]))
    file3.write('\n')

  
