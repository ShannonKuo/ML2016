import sys
import math

file = open("train.csv", 'r')
file2 = open("test_X.csv", 'r')
file3 = open(sys.argv[20], 'w')
PM25_poly = int(sys.argv[21])
PM10_poly = int(sys.argv[22])


adagrad = 0
mini_batch = 0
mini = 0
mini_len = 1
iteration = int(sys.argv[19])
x = 0
b = 0
if adagrad == 0 : n = 0.00000001
if adagrad == 1 : n = 100.0


data_AMBTEMP = list()
data_CH4 = list()
data_CO = list()
data_NMHC = list()
data_NO = list()
data_NO2 = list()
data_NOx = list()
data_O3 = list()
data_PM10 = list()
data_PM25 = list()
data_RAINFALL = list()
data_RH = list()
data_SO2 = list()
data_THC = list()
data_WDHR = list()
data_WINDDIREC = list()
data_WINDSPEED = list()
data_WSHR = list()


for line in file.readlines():
    line = line.strip()
    if not line: break
    if line.find("AMB_TEMP") >= 0:
       arr = line[line.find("AMB_TEMP")+9:]
       arr = arr.split(',')
       data_AMBTEMP.append(arr)
    if line.find("CH4") >= 0:
       arr = line[line.find("CH4")+4:]
       arr = arr.split(',')
       data_CH4.append(arr)
    if line.find("CO") >= 0:
       arr = line[line.find("CO")+3:]
       arr = arr.split(',')
       data_CO.append(arr)
    if line.find("NMHC") >= 0:
       arr = line[line.find("NMHC")+5:]
       arr = arr.split(',')
       data_NMHC.append(arr)
    if line.find("NO") >= 0:
       arr = line[line.find("NO")+3:]
       arr = arr.split(',')
       data_NO.append(arr)
    if line.find("NO2") >= 0:
       arr = line[line.find("NO2")+4:]
       arr = arr.split(',')
       data_NO2.append(arr)
    if line.find("NOx") >= 0:
       arr = line[line.find("NOx")+4:]
       arr = arr.split(',')
       data_NOx.append(arr)
    if line.find("O3") >= 0:
       arr = line[line.find("O3")+3:]
       arr = arr.split(',')
       data_O3.append(arr)
    if line.find("PM10") >= 0:
       arr = line[line.find("PM10")+5:]
       arr = arr.split(',')
       data_PM10.append(arr)
    if line.find("PM2.5") >= 0:
       arr = line[line.find("PM2.5")+6:]
       arr = arr.split(',')
       data_PM25.append(arr)
    if line.find("RAINFALL") >= 0:
       arr = line[line.find("RAINFALL")+9:]
       arr = arr.split(',')
       for i in range(0,len(arr),1):
           if arr[i] == "NR": arr[i] = "0"
       data_RAINFALL.append(arr)
    if line.find("RH") >= 0:
       arr = line[line.find("RH")+3:]
       arr = arr.split(',')
       data_RH.append(arr)
    if line.find("SO2") >= 0:
       arr = line[line.find("SO2")+4:]
       arr = arr.split(',')
       data_SO2.append(arr)
    if line.find("THC") >= 0:
       arr = line[line.find("THC")+4:]
       arr = arr.split(',')
       data_THC.append(arr)
    if line.find("WD_HR") >= 0:
       arr = line[line.find("WD_HR")+6:]
       arr = arr.split(',')
       data_WDHR.append(arr)
    if line.find("WIND_DIREC") >= 0:
       arr = line[line.find("WIND_DIREC")+11:]
       arr = arr.split(',')
       data_WINDDIREC.append(arr)
    if line.find("WIND_SPEED") >= 0:
       arr = line[line.find("WIND_SPEED")+11:]
       arr = arr.split(',')
       data_WINDSPEED.append(arr)
    if line.find("WS_HR") >= 0:
       arr = line[line.find("WS_HR")+6:]
       arr = arr.split(',')
       data_WSHR.append(arr)



AMBTEMP_term = int(sys.argv[1])
CH4_term = int(sys.argv[2])
CO_term = int(sys.argv[3])
NMHC_term = int(sys.argv[4])
NO_term = int(sys.argv[5])
NO2_term = int(sys.argv[6])
NOx_term = int(sys.argv[7])
O3_term = int(sys.argv[8])
PM10_term = int(sys.argv[9])
PM25_term = int(sys.argv[10])
RAINFALL_term = int(sys.argv[11])
RH_term = int(sys.argv[12])
SO2_term = int(sys.argv[13])
THC_term = int(sys.argv[14])
WDHR_term = int(sys.argv[15])
WINDDIREC_term = int(sys.argv[16])
WINDSPEED_term = int(sys.argv[17])
WSHR_term = int(sys.argv[18])


#put numbers in a group

data2_AMBTEMP = list()
data2_CH4 = list()
data2_CO = list()
data2_NMHC = list()
data2_NO = list()
data2_NO2 = list()
data2_NOx = list()
data2_O3 = list()
data2_PM10 = list()
data2_PM25 = list()
data2_RAINFALL = list()
data2_RH = list()
data2_SO2 = list()
data2_THC = list()
data2_WDHR = list()
data2_WINDDIREC = list()
data2_WINDSPEED = list()
data2_WSHR = list()

data2_PM10_poly = list()
data2_PM25_poly = list()


for i in range(len(data_PM25)):

    for j in range(PM25_term - AMBTEMP_term, len(data_AMBTEMP[i])-AMBTEMP_term,1):
        temp = list()
        temp = data_AMBTEMP[i][j:j+(AMBTEMP_term)+1]
        data2_AMBTEMP.append(temp)
    for j in range(PM25_term - CH4_term, len(data_CH4[i])-CH4_term,1):
        temp = list()
        temp = data_CH4[i][j:j+(CH4_term)+1]
        data2_CH4.append(temp)
    for j in range(PM25_term - CO_term,len(data_CO[i])-CO_term,1):
        temp = list()
        temp = data_CO[i][j:j+(CO_term)+1]
        data2_CO.append(temp)
    for j in range(PM25_term - NMHC_term,len(data_NMHC[i])-NMHC_term,1):
        temp = list()
        temp = data_NMHC[i][j:j+(NMHC_term)+1]
        data2_NMHC.append(temp)
    for j in range(PM25_term - NO_term,len(data_NO[i])-NO_term,1):
        temp = list()
        temp = data_NO[i][j:j+(NO_term)+1]
        data2_NO.append(temp)
    for j in range(PM25_term - NO2_term,len(data_NO2[i])-NO2_term,1):
        temp = list()
        temp = data_NO2[i][j:j+(NO2_term)+1]
        data2_NO2.append(temp)
    for j in range(PM25_term - NOx_term,len(data_NOx[i])-NOx_term,1):
        temp = list()
        temp = data_NOx[i][j:j+(NOx_term)+1]
        data2_NOx.append(temp)
    for j in range(PM25_term - O3_term,len(data_O3[i])-O3_term,1):
        temp = list()
        temp = data_O3[i][j:j+(O3_term)+1]
        data2_O3.append(temp)
    for j in range(PM25_term - PM10_term,len(data_PM10[i])-PM10_term,1):
        temp = list()
        temp = data_PM10[i][j:j+(PM10_term)+1]
        data2_PM10.append(temp)
    for j in range(PM25_term - PM25_term,len(data_PM25[i])-PM25_term,1):
        temp = list()
        temp = data_PM25[i][j:j+(PM25_term)+1]
        data2_PM25.append(temp)
    for j in range(PM25_term - RAINFALL_term,len(data_RAINFALL[i])-RAINFALL_term,1):
        temp = list()
        temp = data_RAINFALL[i][j:j+(RAINFALL_term)+1]
        data2_RAINFALL.append(temp)
    for j in range(PM25_term - RH_term,len(data_RH[i])-RH_term,1):
        temp = list()
        temp = data_RH[i][j:j+(RH_term)+1]
        data2_RH.append(temp)
    for j in range(PM25_term - SO2_term,len(data_SO2[i])-SO2_term,1):
        temp = list()
        temp = data_SO2[i][j:j+(SO2_term)+1]
        data2_SO2.append(temp)
    for j in range(PM25_term - THC_term,len(data_THC[i])-THC_term,1):
        temp = list()
        temp = data_THC[i][j:j+(THC_term)+1]
        data2_THC.append(temp)
    for j in range(PM25_term - WDHR_term,len(data_WDHR[i])-WDHR_term,1):
        temp = list()
        temp = data_WDHR[i][j:j+(WDHR_term)+1]
        data2_WDHR.append(temp)
    for j in range(PM25_term - WINDDIREC_term,len(data_WINDDIREC[i])-WINDDIREC_term,1):
        temp = list()
        temp = data_WINDDIREC[i][j:j+(WINDDIREC_term)+1]
        data2_WINDDIREC.append(temp)
    for j in range(PM25_term - WINDSPEED_term,len(data_WINDSPEED[i])-WINDSPEED_term,1):
        temp = list()
        temp = data_WINDSPEED[i][j:j+(WINDSPEED_term)+1]
        data2_WINDSPEED.append(temp)
    for j in range(PM25_term - WSHR_term,len(data_WSHR[i])-WSHR_term,1):
        temp = list()
        temp = data_WSHR[i][j:j+(WSHR_term)+1]
        data2_WSHR.append(temp)

    for j in range(PM25_term - PM10_poly,len(data_PM10[i])-PM10_poly,1):
        temp = list()
        temp = data_PM10[i][j:j+(PM10_poly)+1]
        data2_PM10_poly.append(temp)
    for j in range(PM25_term - PM25_poly,len(data_PM25[i])-PM25_poly,1):
        temp = list()
        temp = data_PM25[i][j:j+(PM25_poly)+1]
        data2_PM25_poly.append(temp)


#train data
w_AMBTEMP = [1.0/AMBTEMP_term for i in range(0, AMBTEMP_term ,1)]
w_CH4 = [0.0/CH4_term for i in range(0, CH4_term,1)]
w_CO = [0.5/CO_term for i in range(0, CO_term,1)]
w_NMHC = [0.5/NMHC_term for i in range(0, NMHC_term,1)]
w_NO = [0.5/NO_term for i in range(0, NO_term,1)]
w_NO2 = [0.1/NO2_term for i in range(0, NO2_term,1)]
w_NOx = [0.1/NOx_term for i in range(0, NOx_term,1)]
w_O3 = [0.1/O3_term for i in range(0, O3_term,1)]
w_PM10 = [0.5/PM10_term for i in range(0, PM10_term,1)]
w_PM25 = [0.6/PM25_term for i in range(0, PM25_term,1)]
w_RAINFALL = [0.0/RAINFALL_term for i in range(0, RAINFALL_term,1)]
w_RH = [0.5/RH_term for i in range(0, RH_term,1)]
w_SO2 = [0.1/SO2_term for i in range(0, SO2_term,1)]
w_THC = [0.0/THC_term for i in range(0, THC_term,1)]
w_WDHR = [0.5/WDHR_term for i in range(0, WDHR_term,1)]
w_WINDDIREC = [0.5/WINDDIREC_term for i in range(0, WINDDIREC_term,1)]
w_WINDSPEED = [0.5/WINDSPEED_term for i in range(0, WINDSPEED_term,1)]
w_WSHR = [0.5/WSHR_term for i in range(0, WSHR_term,1)]

w_PM25_poly = [0 for i in range(0, PM25_poly, 1)]
w_PM10_poly = [0 for i in range(0, PM10_poly, 1)]

#learning rate
n_AMBTEMP = [n for i in range(0, AMBTEMP_term ,1)]
n_CH4 = [n for i in range(0, CH4_term,1)]
n_CO = [n for i in range(0, CO_term,1)]
n_NMHC = [n for i in range(0, NMHC_term,1)]
n_NO = [n for i in range(0, NO_term,1)]
n_NO2 = [n for i in range(0, NO2_term,1)]
n_NOx = [n for i in range(0, NOx_term,1)]
n_O3 = [n for i in range(0, O3_term,1)]
n_PM10 = [n for i in range(0, PM10_term,1)]
n_PM25 = [n for i in range(0, PM25_term,1)]
n_RAINFALL = [n for i in range(0, RAINFALL_term,1)]
n_RH = [n for i in range(0, RH_term,1)]
n_SO2 = [n for i in range(0, SO2_term,1)]
n_THC = [n for i in range(0, THC_term,1)]
n_WDHR = [n for i in range(0, WDHR_term,1)]
n_WINDDIREC = [n for i in range(0, WINDDIREC_term,1)]
n_WINDSPEED = [n for i in range(0, WINDSPEED_term,1)]
n_WSHR = [n for i in range(0, WSHR_term,1)]


n_PM25_poly = [n/10000.0 for i in range(0, PM25_poly,1)]
n_PM10_poly = [n/10000.0 for i in range(0, PM10_poly,1)]

n0 = n

diffb = 0
loss = 0
loss_pre = 0
w_s = 0


diff_AMBTEMP = [0 for i in range(0, AMBTEMP_term ,1)]
diff_CH4 = [0 for i in range(0, CH4_term,1)]
diff_CO = [0 for i in range(0, CO_term,1)]
diff_NMHC = [0 for i in range(0, NMHC_term,1)]
diff_NO = [0 for i in range(0, NO_term,1)]
diff_NO2 = [0 for i in range(0, NO2_term,1)]
diff_NOx = [0 for i in range(0, NOx_term,1)]
diff_O3 = [0 for i in range(0, O3_term,1)]
diff_PM10 = [0 for i in range(0, PM10_term,1)]
diff_PM25 = [0 for i in range(0, PM25_term,1)]
diff_RAINFALL = [0 for i in range(0, RAINFALL_term,1)]
diff_RH = [0 for i in range(0, RH_term,1)]
diff_SO2 = [0 for i in range(0, SO2_term,1)]
diff_THC = [0 for i in range(0, THC_term,1)]
diff_WDHR = [0 for i in range(0, WDHR_term,1)]
diff_WINDDIREC = [0 for i in range(0, WINDDIREC_term,1)]
diff_WINDSPEED = [0 for i in range(0, WINDSPEED_term,1)]
diff_WSHR = [0 for i in range(0, WSHR_term,1)]


diff_PM10_poly = [0 for i in range(0, PM10_poly,1)]
diff_PM25_poly = [0 for i in range(0, PM25_poly,1)]


sd_AMBTEMP = [0.001 for i in range(0, AMBTEMP_term ,1)]
sd_CH4 = [0.001 for i in range(0, CH4_term ,1)]
sd_CO = [0.001 for i in range(0, CO_term ,1)]
sd_NMHC = [0.001 for i in range(0, NMHC_term ,1)]
sd_NO = [0.001 for i in range(0, NO_term ,1)]
sd_NO2 = [0.001 for i in range(0, NO2_term ,1)]
sd_NOx = [0.001 for i in range(0, NOx_term ,1)]
sd_O3 = [0.001 for i in range(0, O3_term ,1)]
sd_PM10 = [0.001 for i in range(0, PM10_term ,1)]
sd_PM25 = [0.001 for i in range(0, PM25_term ,1)]
sd_RAINFALL = [0.001 for i in range(0, RAINFALL_term ,1)]
sd_RH = [0.001 for i in range(0, RH_term ,1)]
sd_SO2 = [0.001 for i in range(0, SO2_term ,1)]
sd_THC = [0.001 for i in range(0, THC_term ,1)]
sd_WDHR = [0.001 for i in range(0, WDHR_term ,1)]
sd_WINDDIREC = [0.001 for i in range(0, WINDDIREC_term ,1)]
sd_WINDSPEED = [0.001 for i in range(0, WINDSPEED_term ,1)]
sd_WSHR = [0.001 for i in range(0, WSHR_term ,1)]
sd_b = 0.001


sd_PM10_poly = [0.001 for i in range(0, PM10_poly ,1)]
sd_PM25_poly = [0.001 for i in range(0, PM25_poly ,1)]

for k in range (0, iteration, 1):
   loss = 0
   diffb = 0
   for i in range(0, AMBTEMP_term, 1):
       diff_AMBTEMP[i] = 0
   for i in range(0, CH4_term, 1):
       diff_CH4[i] = 0
   for i in range(0, CO_term, 1):
       diff_CO[i] = 0
   for i in range(0, NMHC_term, 1):
       diff_NMHC[i] = 0
   for i in range(0, NO_term, 1):
       diff_NO[i] = 0
   for i in range(0, NO2_term, 1):
       diff_NO2[i] = 0
   for i in range(0, NOx_term, 1):
       diff_NOx[i] = 0
   for i in range(0, O3_term, 1):
       diff_O3[i] = 0
   for i in range(0, PM10_term, 1):
       diff_PM10[i] = 0
   for i in range(0, PM25_term, 1):
       diff_PM25[i] = 0
   for i in range(0, RAINFALL_term, 1):
       diff_RAINFALL[i] = 0
   for i in range(0, RH_term, 1):
       diff_RH[i] = 0
   for i in range(0, SO2_term, 1):
       diff_SO2[i] = 0
   for i in range(0, THC_term, 1):
       diff_THC[i] = 0
   for i in range(0, WDHR_term, 1):
       diff_WDHR[i] = 0
   for i in range(0, WINDDIREC_term, 1):
       diff_WINDDIREC[i] = 0
   for i in range(0, WINDSPEED_term, 1):
       diff_WINDSPEED[i] = 0
   for i in range(0, WSHR_term, 1):
       diff_WSHR[i] = 0

   for i in range(0, PM10_poly, 1):
       diff_PM10_poly[i] = 0
   for i in range(0, PM25_poly, 1):
       diff_PM25_poly[i] = 0
   


   for i in range(0, len(data2_PM25)/mini_len,1):
       t = 0
       w_s = 0
       for j in range(0, AMBTEMP_term, 1):
           t += float(data2_AMBTEMP[i+mini][j])*float(w_AMBTEMP[j]) 
           #w_s += float(w_AMBTEMP[j]) ** 2
       for j in range(0, CH4_term, 1):
           t += float(data2_CH4[i+mini][j])*float(w_CH4[j]) 
           #w_s += float(w_CH4[j]) ** 2
       for j in range(0, CO_term, 1):
           t += float(data2_CO[i+mini][j])*float(w_CO[j]) 
           #w_s += float(w_CO[j]) ** 2
       for j in range(0, NMHC_term, 1):
           t += float(data2_NMHC[i+mini][j])*float(w_NMHC[j]) 
           #w_s += float(w_NMHC[j]) ** 2
       for j in range(0, NO_term, 1):
           t += float(data2_NO[i+mini][j])*float(w_NO[j]) 
           #w_s += float(w_NO[j]) ** 2
       for j in range(0, NO2_term, 1):
           t += float(data2_NO2[i+mini][j])*float(w_NO2[j]) 
           #w_s += float(w_NO2[j]) ** 2
       for j in range(0, NOx_term, 1):
           t += float(data2_NOx[i+mini][j])*float(w_NOx[j]) 
           #w_s += float(w_NOx[j]) ** 2
       for j in range(0, O3_term, 1):
           t += float(data2_O3[i+mini][j])*float(w_O3[j]) 
           #w_s += float(w_O3[j]) ** 2
       for j in range(0, PM10_term, 1):
           t += float(data2_PM10[i+mini][j])*float(w_PM10[j]) 
           #w_s += float(w_PM10[j]) ** 2
       for j in range(0, PM25_term, 1):
           t += float(data2_PM25[i+mini][j])*float(w_PM25[j]) 
           #w_s += float(w_PM25[j]) ** 2
       for j in range(0, RAINFALL_term, 1):
           t += float(data2_RAINFALL[i+mini][j])*float(w_RAINFALL[j]) 
           #w_s += float(w_RAINFALL[j]) ** 2
       for j in range(0, RH_term, 1):
           t += float(data2_RH[i+mini][j])*float(w_RH[j]) 
           #w_s += float(w_RH[j]) ** 2
       for j in range(0, SO2_term, 1):
           t += float(data2_SO2[i+mini][j])*float(w_SO2[j]) 
           #w_s += float(w_SO2[j]) ** 2
       for j in range(0, THC_term, 1):
           t += float(data2_THC[i+mini][j])*float(w_THC[j]) 
           #w_s += float(w_THC[j]) ** 2
       for j in range(0, WDHR_term, 1):
           t += float(data2_WDHR[i+mini][j])*float(w_WDHR[j]) 
           #w_s += float(w_WDHR[j]) ** 2
       for j in range(0, WINDDIREC_term, 1):
           t += float(data2_WINDDIREC[i+mini][j])*float(w_WINDDIREC[j]) 
           #w_s += float(w_WINDDIREC[j]) ** 2
       for j in range(0, WINDSPEED_term, 1):
           t += float(data2_WINDSPEED[i+mini][j])*float(w_WINDSPEED[j]) 
           #w_s += float(w_WINDSPEED[j]) ** 2
       for j in range(0, WSHR_term, 1):
           t += float(data2_WSHR[i+mini][j])*float(w_WSHR[j]) 
           #w_s += float(w_WSHR[j]) ** 2

       for j in range(0, PM10_poly, 1):
           t += (float(data2_PM10_poly[i+mini][j])**2 )*float(w_PM10_poly[j]) 
           #w_s += float(w_PM10_poly[j]) ** 2
       for j in range(0, PM25_poly, 1):
           t += (float(data2_PM25_poly[i+mini][j])**2 )*float(w_PM25_poly[j]) 
           #w_s += float(w_PM25_poly[j]) ** 2
       


       for j in range(0, AMBTEMP_term, 1):
           diff_AMBTEMP[j] += 2 * (float(data2_PM25[i+mini][PM25_term]) - (t + b)) * (-float(data2_AMBTEMP[i+mini][j])) + x * 2 * w_AMBTEMP[j]
       for j in range(0, CH4_term, 1):
           diff_CH4[j] += 2 * (float(data2_PM25[i+mini][PM25_term]) - (t + b)) * (-float(data2_CH4[i+mini][j])) + x * 2 * w_CH4[j]
       for j in range(0, CO_term, 1):
           diff_CO[j] += 2 * (float(data2_PM25[i+mini][PM25_term]) - (t + b)) * (-float(data2_CO[i+mini][j])) + x * 2 * w_CO[j]
       for j in range(0, NMHC_term, 1):
           diff_NMHC[j] += 2 * (float(data2_PM25[i+mini][PM25_term]) - (t + b)) * (-float(data2_NMHC[i+mini][j])) + x * 2 * w_NMHC[j]
       for j in range(0, NO_term, 1):
           diff_NO[j] += 2 * (float(data2_PM25[i+mini][PM25_term]) - (t + b)) * (-float(data2_NO[i+mini][j])) + x * 2 * w_NO[j]
       for j in range(0, NO2_term, 1):
           diff_NO2[j] += 2 * (float(data2_PM25[i+mini][PM25_term]) - (t + b)) * (-float(data2_NO2[i+mini][j])) + x * 2 * w_NO2[j] 
       for j in range(0, NOx_term, 1):
           diff_NOx[j] += 2 * (float(data2_PM25[i+mini][PM25_term]) - (t + b)) * (-float(data2_NOx[i+mini][j])) + x * 2 * w_NOx[j]
       for j in range(0, O3_term, 1):
           diff_O3[j] += 2 * (float(data2_PM25[i+mini][PM25_term]) - (t + b)) * (-float(data2_O3[i+mini][j])) + x * 2 * w_O3[j]
       for j in range(0, PM10_term, 1):
           diff_PM10[j] += 2 * (float(data2_PM25[i+mini][PM25_term]) - (t + b)) * (-float(data2_PM10[i+mini][j])) + x * 2 * w_PM10[j]
       for j in range(0, PM25_term, 1):
           diff_PM25[j] += (2 * (float(data2_PM25[i+mini][PM25_term]) - (t + b)) * (-float(data2_PM25[i+mini][j]))) + x * 2 * w_PM25[j]
       for j in range(0, RAINFALL_term, 1):
           diff_RAINFALL[j] += (2 * (float(data2_PM25[i+mini][PM25_term]) - (t + b)) * (-float(data2_RAINFALL[i+mini][j]))) + x * 2 * w_RAINFALL[j]
       for j in range(0, RH_term, 1):
           diff_RH[j] += 2 * (float(data2_PM25[i+mini][PM25_term]) - (t + b)) * (-float(data2_RH[i+mini][j])) + x * 2 * w_RH[j]
       for j in range(0, SO2_term, 1):
           diff_SO2[j] += 2 * (float(data2_PM25[i+mini][PM25_term]) - (t + b)) * (-float(data2_SO2[i+mini][j])) + x * 2 * w_SO2[j]
       for j in range(0, THC_term, 1):
           diff_THC[j] += 2 * (float(data2_PM25[i+mini][PM25_term]) - (t + b)) * (-float(data2_THC[i+mini][j])) + x * 2 * w_THC[j]
       for j in range(0, WDHR_term, 1):
           diff_WDHR[j] += 2 * (float(data2_PM25[i+mini][PM25_term]) - (t + b)) * (-float(data2_WDHR[i+mini][j])) + x * 2 * w_WDHR[j]
       for j in range(0, WINDDIREC_term, 1):
           diff_WINDDIREC[j] += 2 * (float(data2_PM25[i+mini][PM25_term]) - (t + b)) * (-float(data2_WINDDIREC[i+mini][j])) + x * 2 * w_WINDDIREC[j]
       for j in range(0, WINDSPEED_term, 1):
           diff_WINDSPEED[j] += 2 * (float(data2_PM25[i+mini][PM25_term]) - (t + b)) * (-float(data2_WINDSPEED[i+mini][j])) + x * 2 * w_WINDSPEED[j]
       for j in range(0, WSHR_term, 1):
           diff_WSHR[j] += 2 * (float(data2_PM25[i+mini][PM25_term]) - (t + b)) * (-float(data2_WSHR[i+mini][j])) + x * 2 * w_WSHR[j]


       for j in range(0, PM10_poly, 1):
           diff_PM10_poly[j] += 2 * (float(data2_PM25[i+mini][PM25_term]) - (t + b)) * (-1)*(float(data2_PM10_poly[i+mini][j])**2) + x * 2 * w_PM10_poly[j]
       for j in range(0, PM25_poly, 1):
           diff_PM25_poly[j] += 2 * (float(data2_PM25[i+mini][PM25_term]) - (t + b)) * (-1)*(float(data2_PM25_poly[i+mini][j])**2) + x * 2 * w_PM25_poly[j]

       diffb += 2 * (float(data2_PM25[i+mini][PM25_term]) - (t + b)) * (-1)
       loss += (float(data2_PM25[i+mini][PM25_term]) - (t + b)) **2 + x * w_s 

   if mini_batch == 1: 
      if mini != len(data2_PM25)*(mini_len-1)/mini_len : mini += len(data2_PM25)/mini_len
      else : mini = 0
   #adagrad
   if adagrad == 1:
      sd_b += diffb ** 2
      n_b = n0 / (sd_b ** 0.5)
      for i in range(0, AMBTEMP_term, 1):
          sd_AMBTEMP[i] += (diff_AMBTEMP[i] ** 2)
          n_AMBTEMP[i] = n0 / (sd_AMBTEMP[i] ** 0.5)
      for i in range(0, CH4_term, 1):
          sd_CH4[i] += (diff_CH4[i] ** 2)
          n_CH4[i] = n0 / (sd_CH4[i] ** 0.5)
      for i in range(0, CO_term, 1):
          sd_CO[i] += (diff_CO[i] ** 2)
          n_CO[i] = n0 / (sd_CO[i] ** 0.5)
      for i in range(0, NMHC_term, 1):
          sd_NMHC[i] += (diff_NMHC[i] ** 2)
          n_NMHC[i] = n0 / (sd_NMHC[i] ** 0.5)
      for i in range(0, NO_term, 1):
          sd_NO[i] += (diff_NO[i] ** 2)
          n_NO[i] = n0 / (sd_NO[i] ** 0.5)
      for i in range(0, NO2_term, 1):
          sd_NO2[i] += (diff_NO2[i] ** 2)
          n_NO2[i] = n0 / (sd_NO2[i] ** 0.5)
      for i in range(0, NOx_term, 1):
          sd_NOx[i] += (diff_NOx[i] ** 2)
          n_NOx[i] = n0 / (sd_NOx[i] ** 0.5)
      for i in range(0, O3_term, 1):
          sd_O3[i] += (diff_O3[i] ** 2)
          n_O3[i] = n0 / (sd_O3[i] ** 0.5)
      for i in range(0, PM10_term, 1):
          sd_PM10[i] += (diff_PM10[i] ** 2)
          n_PM10[i] = n0 / (sd_PM10[i] ** 0.5)
      for i in range(0, PM25_term, 1):
          sd_PM25[i] += (diff_PM25[i] ** 2)
          n_PM25[i] = n0 / (sd_PM25[i] ** 0.5)
      for i in range(0, RAINFALL_term, 1):
          sd_RAINFALL[i] += (diff_RAINFALL[i] ** 2)
          n_RAINFALL[i] = n0 / (sd_RAINFALL[i] ** 0.5)
      for i in range(0, RH_term, 1):
          sd_RH[i] += (diff_RH[i] ** 2)
          n_RH[i] = n0 / (sd_RH[i] ** 0.5)
      for i in range(0, SO2_term, 1):
          sd_SO2[i] += (diff_SO2[i] ** 2)
          n_SO2[i] = n0 / (sd_SO2[i] ** 0.5)
      for i in range(0, THC_term, 1):
          sd_THC[i] += (diff_THC[i] ** 2)
          n_THC[i] = n0 / (sd_THC[i] ** 0.5)
      for i in range(0, WDHR_term, 1):
          sd_WDHR[i] += (diff_WDHR[i] ** 2)
          n_WDHR[i] = n0 / (sd_WDHR[i] ** 0.5)
      for i in range(0, WINDDIREC_term, 1):
          sd_WINDDIREC[i] += (diff_WINDDIREC[i] ** 2)
          n_WINDDIREC[i] = n0 / (sd_WINDDIREC[i] ** 0.5)
      for i in range(0, WINDSPEED_term, 1):
          sd_WINDSPEED[i] += (diff_WINDSPEED[i] ** 2)
          n_WINDSPEED[i] = n0 / (sd_WINDSPEED[i] ** 0.5)
      for i in range(0, WSHR_term, 1):
          sd_WSHR[i] += (diff_WSHR[i] ** 2)
          n_WSHR[i] = n0 / (sd_WSHR[i] ** 0.5)


      for i in range(0, PM10_poly, 1):
          sd_PM10_poly[i] += (diff_PM10_poly[i] ** 2)
          n_PM10_poly[i] = n0 / (sd_PM10_poly_[i] ** 0.5)
      for i in range(0, PM25_poly, 1):
          sd_PM25_poly[i] += (diff_PM25_poly[i] ** 2)
          n_PM25_poly[i] = n0 / (sd_PM25_poly[i] ** 0.5)
   else : n_b = n0




   for j in range(0, AMBTEMP_term, 1):
       w_AMBTEMP[j] -= n_AMBTEMP[j] * diff_AMBTEMP[j]
   for j in range(0, CH4_term, 1):
       w_CH4[j] -= n_CH4[j] * diff_CH4[j]
   for j in range(0, CO_term, 1):
       w_CO[j] -= n_CO[j] * diff_CO[j]
   for j in range(0, NMHC_term, 1):
       w_NMHC[j] -= n_NMHC[j] * diff_NMHC[j]
   for j in range(0, NO_term, 1):
       w_NO[j] -= n_NO[j] * diff_NO[j]
   for j in range(0, NO2_term, 1):
       w_NO2[j] -= n_NO2[j] * diff_NO2[j]
   for j in range(0, NOx_term, 1):
       w_NOx[j] -= n_NOx[j] * diff_NOx[j]
   for j in range(0, O3_term, 1):
       w_O3[j] -= n_O3[j] * diff_O3[j]
   for j in range(0, PM10_term, 1):
       w_PM10[j] -= n_PM10[j] * diff_PM10[j]
   for j in range(0, PM25_term, 1):
       w_PM25[j] -= n_PM25[j] * diff_PM25[j]
   for j in range(0, RAINFALL_term, 1):
       w_RAINFALL[j] -= n_RAINFALL[j] * diff_RAINFALL[j]
   for j in range(0, RH_term, 1):
       w_RH[j] -= n_RH[j] * diff_RH[j]
   for j in range(0, SO2_term, 1):
       w_SO2[j] -= n_SO2[j] * diff_SO2[j]
   for j in range(0, THC_term, 1):
       w_THC[j] -= n_THC[j] * diff_THC[j]
   for j in range(0, WDHR_term, 1):
       w_WDHR[j] -= n_WDHR[j] * diff_WDHR[j]
   for j in range(0, WINDDIREC_term, 1):
       w_WINDDIREC[j] -= n_WINDDIREC[j] * diff_WINDDIREC[j]
   for j in range(0, WINDSPEED_term, 1):
       w_WINDSPEED[j] -= n_WINDSPEED[j] * diff_WINDSPEED[j]
   for j in range(0, WSHR_term, 1):
       w_WSHR[j] -= n_WSHR[j] * diff_WSHR[j]


   for j in range(0, PM10_poly, 1):
       w_PM10_poly[j] -= n_PM10_poly[j] * diff_PM10_poly[j]
   for j in range(0, PM25_poly, 1):
       w_PM25_poly[j] -= n_PM25_poly[j] * diff_PM25_poly[j]
   b -= n_b * diffb
   print k
   loss = (loss/(len(data2_PM25)/mini_len))**0.5 
   if (abs(loss_pre - loss)<0.000000001) : 
      break
   loss_pre = loss

RMSE = 0
for i in range(0, len(data2_PM25), 1):
   t = 0
   for j in range(0, AMBTEMP_term, 1):
       t += float(data2_AMBTEMP[i][j])*float(w_AMBTEMP[j]) 
   for j in range(0, CH4_term, 1):
       t += float(data2_CH4[i][j])*float(w_CH4[j]) 
   for j in range(0, CO_term, 1):
       t += float(data2_CO[i][j])*float(w_CO[j]) 
   for j in range(0, NMHC_term, 1):
       t += float(data2_NMHC[i][j])*float(w_NMHC[j]) 
   for j in range(0, NO_term, 1):
       t += float(data2_NO[i][j])*float(w_NO[j]) 
   for j in range(0, NO2_term, 1):
       t += float(data2_NO2[i][j])*float(w_NO2[j]) 
   for j in range(0, NOx_term, 1):
       t += float(data2_NOx[i][j])*float(w_NOx[j]) 
   for j in range(0, O3_term, 1):
       t += float(data2_O3[i][j])*float(w_O3[j]) 
   for j in range(0, PM10_term, 1):
       t += float(data2_PM10[i][j])*float(w_PM10[j]) 
   for j in range(0, PM25_term, 1):
       t += float(data2_PM25[i][j])*float(w_PM25[j]) 
   for j in range(0, RAINFALL_term, 1):
       t += float(data2_RAINFALL[i][j])*float(w_RAINFALL[j]) 
   for j in range(0, RH_term, 1):
       t += float(data2_RH[i][j])*float(w_RH[j]) 
   for j in range(0, SO2_term, 1):
       t += float(data2_SO2[i][j])*float(w_SO2[j]) 
   for j in range(0, THC_term, 1):
       t += float(data2_THC[i][j])*float(w_THC[j]) 
   for j in range(0, WDHR_term, 1):
       t += float(data2_WDHR[i][j])*float(w_WDHR[j]) 
   for j in range(0, WINDDIREC_term, 1):
       t += float(data2_WINDDIREC[i][j])*float(w_WINDDIREC[j]) 
   for j in range(0, WINDSPEED_term, 1):
       t += float(data2_WINDSPEED[i][j])*float(w_WINDSPEED[j]) 
   for j in range(0, WSHR_term, 1):
       t += float(data2_WSHR[i+mini][j])*float(w_WSHR[j]) 


   for j in range(0, PM10_poly, 1):
       t += (float(data2_PM10_poly[i][j])**2)*float(w_PM10_poly[j]) 
   for j in range(0, PM25_poly, 1):
       t += (float(data2_PM25_poly[i][j])**2)*float(w_PM25_poly[j]) 

   RMSE += (float(data2_PM25[i][PM25_term]) - (t + b)) **2 
RMSE = ((RMSE/len(data2_PM25))**0.5)
print RMSE

#input test_X.csv 

data3_AMBTEMP = list()
data3_CH4 = list()
data3_CO = list()
data3_NMHC = list()
data3_NO = list()
data3_NO2 = list()
data3_NOx = list()
data3_O3 = list()
data3_PM10 = list()
data3_PM25 = list()
data3_RAINFALL = list()
data3_RH = list()
data3_SO2 = list()
data3_THC = list()
data3_WDHR = list()
data3_WINDDIREC = list()
data3_WINDSPEED = list()
data3_WSHR = list()


for line2 in file2.readlines():
    line2 = line2.strip()
    if not line2: break
    if line2.find("AMB_TEMP") >= 0:
       arr = line2[line2.find("AMB_TEMP")+9:]
       arr = arr.split(',')
       data3_AMBTEMP.append(arr)
    if line2.find("CH4") >= 0:
       arr = line2[line2.find("CH4")+4:]
       arr = arr.split(',')
       data3_CH4.append(arr)
    if line2.find("CO") >= 0:
       arr = line2[line2.find("CO")+3:]
       arr = arr.split(',')
       data3_CO.append(arr)
    if line2.find("NMHC") >= 0:
       arr = line2[line2.find("NMHC")+5:]
       arr = arr.split(',')
       data3_NMHC.append(arr)
    if line2.find("NO") >= 0:
       arr = line2[line2.find("NO")+3:]
       arr = arr.split(',')
       data3_NO.append(arr)
    if line2.find("NO2") >= 0:
       arr = line2[line2.find("NO2")+4:]
       arr = arr.split(',')
       data3_NO2.append(arr)
    if line2.find("NOx") >= 0:
       arr = line2[line2.find("NOx")+4:]
       arr = arr.split(',')
       data3_NOx.append(arr)
    if line2.find("O3") >= 0:
       arr = line2[line2.find("O3")+3:]
       arr = arr.split(',')
       data3_O3.append(arr)
    if line2.find("PM10") >= 0:
       arr = line2[line2.find("PM10")+5:]
       arr = arr.split(',')
       data3_PM10.append(arr)
    if line2.find("PM2.5") >= 0:
       arr = line2[line2.find("PM2.5")+6:]
       arr = arr.split(',')
       data3_PM25.append(arr)
    if line2.find("RAINFALL") >= 0:
       arr = line2[line2.find("RAINFALL")+9:]
       arr = arr.split(',')
       for i in range(0,len(arr),1):
           if arr[i] == "NR": arr[i] = "0"
       data3_RAINFALL.append(arr)
    if line2.find("RH") >= 0:
       arr = line2[line2.find("RH")+3:]
       arr = arr.split(',')
       data3_RH.append(arr)
    if line2.find("SO2") >= 0:
       arr = line2[line2.find("SO2")+4:]
       arr = arr.split(',')
       data3_SO2.append(arr)
    if line2.find("THC") >= 0:
       arr = line2[line2.find("THC")+4:]
       arr = arr.split(',')
       data3_THC.append(arr)
    if line2.find("WD_HR") >= 0:
       arr = line2[line2.find("WD_HR")+6:]
       arr = arr.split(',')
       data3_WDHR.append(arr)
    if line2.find("WIND_DIREC") >= 0:
       arr = line2[line2.find("WIND_DIREC")+11:]
       arr = arr.split(',')
       data3_WINDDIREC.append(arr)
    if line2.find("WIND_SPEED") >= 0:
       arr = line2[line2.find("WIND_SPEED")+11:]
       arr = arr.split(',')
       data3_WINDSPEED.append(arr)
    if line2.find("WS_HR") >= 0:
       arr = line2[line2.find("WS_HR")+6:]
       arr = arr.split(',')
       data3_WSHR.append(arr)


file3.write("id,value" + '\n')
ans = list()
ans = [0 for i in range(0,len(data3_PM25),1)]

for i in range(len(data3_PM25)):
   sum = 0

   for j in range(0, AMBTEMP_term, 1):
       sum += float(w_AMBTEMP[j]) * float(data3_AMBTEMP[i][9-AMBTEMP_term+j])
   for j in range(0, CH4_term, 1):
       sum += float(w_CH4[j]) * float(data3_CH4[i][9-CH4_term+j])
   for j in range(0, CO_term, 1):
       sum += float(w_CO[j]) * float(data3_CO[i][9-CO_term+j])
   for j in range(0, NMHC_term, 1):
       sum += float(w_NMHC[j]) * float(data3_NMHC[i][9-NMHC_term+j])
   for j in range(0, NO_term, 1):
       sum += float(w_NO[j]) * float(data3_NO[i][9-NO_term+j])
   for j in range(0, NO2_term, 1):
       sum += float(w_NO2[j]) * float(data3_NO2[i][9-NO2_term+j])
   for j in range(0, NOx_term, 1):
       sum += float(w_NOx[j]) * float(data3_NOx[i][9-NOx_term+j])
   for j in range(0, O3_term, 1):
       sum += float(w_O3[j]) * float(data3_O3[i][9-O3_term+j])
   for j in range(0, PM10_term, 1):
       sum += float(w_PM10[j]) * float(data3_PM10[i][9-PM10_term+j])
   for j in range(0, PM25_term, 1):
       sum += float(w_PM25[j]) * float(data3_PM25[i][9-PM25_term+j])
   for j in range(0, RAINFALL_term, 1):
       sum += float(w_RAINFALL[j]) * float(data3_RAINFALL[i][9-RAINFALL_term+j])
   for j in range(0, RH_term, 1):
       sum += float(w_RH[j]) * float(data3_RH[i][9-RH_term+j])
   for j in range(0, SO2_term, 1):
       sum += float(w_SO2[j]) * float(data3_SO2[i][9-SO2_term+j])
   for j in range(0, THC_term, 1):
       sum += float(w_THC[j]) * float(data3_THC[i][9-THC_term+j])
   for j in range(0, WDHR_term, 1):
       sum += float(w_WDHR[j]) * float(data3_WDHR[i][9-WDHR_term+j])
   for j in range(0, WINDDIREC_term, 1):
       sum += float(w_WINDDIREC[j]) * float(data3_WINDDIREC[i][9-WINDDIREC_term+j])
   for j in range(0, WINDSPEED_term, 1):
       sum += float(w_WINDSPEED[j]) * float(data3_WINDSPEED[i][9-WINDSPEED_term+j])
   for j in range(0, WSHR_term, 1):
       sum += float(w_WSHR[j]) * float(data3_WSHR[i][9-WSHR_term+j])

   for j in range(0, PM10_poly, 1):
       sum += float(w_PM10_poly[j]) * (float(data3_PM10[i][9-PM10_poly+j]) ** 2)
   for j in range(0, PM25_poly, 1):
       sum += float(w_PM25_poly[j]) * (float(data3_PM25[i][9-PM25_poly+j]) ** 2)


   ans[i] = sum + b
      
for i in range(len(data3_PM25)):
    file3.write("id_"+ str(i) +",")
    file3.write(str(ans[i]))
    file3.write('\n')
