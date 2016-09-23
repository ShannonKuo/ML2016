import sys
file = open(sys.argv[2], 'r')
#content = file.read()
result = list()
for line in file.readlines():
    line = line.strip()
    if not line: break
    arr = list()
    arr = line.split()
    result.append(arr)
    #list.append(line)
ans = list()
for i in range(0,len(result)-1 , 1):
    ans.append(float(result[i][int(sys.argv[1])]))
ans.sort()
file2 = open('ans1.txt', 'w')
for i in range(0, len(ans), 1):
    file2.write( str( ans[i] ) )
    if i != (len(ans)-1): 
        file2.write(',')

file.close
file2.close
