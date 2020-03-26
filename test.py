"""# **Testing**
![alt text](https://drive.google.com/uc?id=1165ETzZyE6HStqKvgR0gKrJwgFLK6-CW)

載入 test data，並且以相似於訓練資料預先處理和特徵萃取的方式處理，使 test data 形成 240 個維度為 18 * 9 + 1 的資料。
"""
import sys
import pandas as pd
import numpy as np
from sys import argv

mean_x = np.load('mean_x.npy')
std_x = np.load('std_x.npy')

feature_set = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
feature = len(feature_set)
pm2_5 = feature_set.index(9)
WD_HR = feature_set.index(14)
WIND_DIREC = feature_set.index(15)
double = feature_set.index(10)
degree = 345

data_path_test = argv[1]
submit =  argv[2]

# testdata = pd.read_csv('gdrive/My Drive/hw1-regression/test.csv', header = None, encoding = 'big5')
testdata = pd.read_csv(data_path_test, header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, feature*9]) 

for i in range(240):
    t1 = test_data[18 * i : 18 * (i + 1), :].astype(float)    
    t2 = t1[feature_set,:]
    #t2[0,:] = np.cos((t2[WIND_DIREC,:]-degree)*np.pi/180)**2
    t2[WD_HR, : ] = np.cos((t2[WD_HR, : ]-degree)*np.pi/180)
    t2[double, : ] = np.cos((t2[WD_HR, : ]-degree)*np.pi/180)**2
    t2[WIND_DIREC, : ] = np.cos((t2[WIND_DIREC, : ]-degree)*np.pi/180)
    t2[16, : ] = np.cos((t2[WIND_DIREC, : ]-degree)*np.pi/180)**2
    test_x[i, :] = t2.reshape(1, -1) 
    
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)
test_x

"""# **Prediction**
說明圖同上

![alt text](https://drive.google.com/uc?id=1165ETzZyE6HStqKvgR0gKrJwgFLK6-CW)

有了 weight 和測試資料即可預測 target。
"""

w = np.load('weight.npy')
ans_y = np.dot(test_x, w)
ans_y

"""# **Save Prediction to CSV File**"""

import csv
with open(submit, mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)

