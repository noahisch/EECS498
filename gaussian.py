

import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import sklearn.linear_model as sm
import math
import sklearn.svm as sv
from multiprocessing import Process, Lock

COLORS = ['b+', 'r+', 'g+', 'k+', 'c+', 'c+']
ITER = 0
DOSECOLORS = ['m', 'y', 'c', 'b', 'g', 'r', 'k']

HR = []
HR_sens = []
HR_spec = []
PULSE = []
PULSE_sens = []
PULSE_spec = []
HR_COEF = []
HR_COL = []
PULSE_COEF = []
PULSE_COL = []
RESP_COEF = []
RESP_COL = []
SP02_COEF = []
SP02_COL = []
COUNT = 0


def plotRegr(x_col, y_col, color, num, id_):
    if(len(x_col) < 2):
        return
    if id_ == 1:
        LR = sm.LinearRegression()
        # print len(x_col), len(y_col)
        x_col = np.array(x_col)
        x_col = x_col - np.min(x_col)
        x_col = x_col[:, np.newaxis]
        y_col = np.transpose(y_col)
        # print x_col
        # print y_col
        LR.fit(x_col, y_col)
        global PULSE_COEF
        # PULSE_COEF.append(np.std(y_col))
        # print np.shape(LR.coef_)
        PULSE_COEF.append(LR.coef_)
        global PULSE_COL
        PULSE_COL.append(num)
        return
    if id_ == 2:
        # LR = sm.LinearRegression()
        # # print len(x_col), len(y_col)
        # x_col = np.array(x_col)
        # x_col = x_col - np.min(x_col)
        # x_col = x_col[:, np.newaxis]
        # y_col = np.transpose(y_col)
        # # print x_col
        # # print y_col
        # LR.fit(x_col, y_col)
        global RESP_COEF
        global RESP_COL
        RESP_COEF.append(np.std(y_col))
        # RESP_COEF.append(LR.coef_)
        RESP_COL.append(num)
        return
    if id_ == 3:
        global SP02_COEF
        global SP02_COL
        SP02_COEF.append(np.std(y_col))
        SP02_COL.append(num)
        return
    global COUNT
    COUNT +=1
    LR = sm.LinearRegression()
    # print len(x_col), len(y_col)
    x_col = np.array(x_col)
    x_col = x_col - np.min(x_col)
    x_col = x_col[:, np.newaxis]
    y_col = np.transpose(y_col)
    # print x_col
    # print y_col
    LR.fit(x_col, y_col)
    # plt.scatter(np.average(x_col), LR.coef_, c = color)
    global HR_COL
    global HR_COEF
    HR_COEF.append(LR.coef_)

    HR_COL.append(num)

    


    

def timeToInt(time):

    split1 = time.split('-')
    returnTime= 0
    split2 = split1[2].split(' ')
    returnTime += int(split2[0])*60*24
    split3 = split2[1].split(':')
    returnTime += int(split3[0]) * 60 + int(split3[1])
    return returnTime


def displayDoses(fileName, figure, startTime):
    data = pd.read_csv(fileName, header=None)
    data.columns = ['row', 'id', 'itemid', 'time', 'dose', 'units']
    start = timeToInt(startTime)
    doses = []
    for x in data['time']:
        doses.append(timeToInt(x) - start)

    return (doses, [x for x in data['dose']])

def join(in_data, next_data):
    columns = in_data.columns
    columns_2 = []
    for col in columns:
        columns_2.append(col)
    columns_2.insert(1, 'day')
    next_data.columns = columns_2

    in_data = in_data.values
  #  print in_data
    in_data = np.transpose(in_data)
    next_data = next_data.values
    next_data = np.transpose(next_data)

   # print in_data
    data = [[]]
    for i,row in enumerate(in_data):
        if i == 0:
            continue
      #  print row
        data.append( [float(x) for x in row])

    for i,row in enumerate(next_data):
        if i < 2:
            continue
        data[i-1] = data[i-1] + [x for x in row]
       # print len(data[i-1])
    return data

def normalize(color, num):
    if(num > 300):
        return 0
    if(num > 180):
        return color * .2
    if (num > 150):
        return color * .4
    if (num > 120):
        return color * .6
    if(num > 80):
        return color * .8
    if (num < 15):
        return color * .6
    return color
def readAndCalculate(fileName, minutes, figure, title, color, nrows_, doseUnits):
    data = 0
    columns = []
    if(nrows_ == 0):
        data = pd.read_csv(fileName, sep='\s+', header= 0, skiprows=[1], na_values = ['-'])
        columns = data.columns
        data = data.values
        data_tmp = np.transpose(data)
        data = [[]]
        for i,row in enumerate(data_tmp):
            if i == 0:
                continue
            data.append( [x for x in row])
        #print data
    else:
        in_data = pd.read_csv(fileName, sep='\s+', header=0, skiprows=[1], na_values = ['-'], nrows = nrows_)
        columns = in_data.columns
        next_data = pd.read_csv(fileName, sep='\s+', header=None, skiprows=nrows_+2, na_values = ['-'], index_col =False)
        data = join(in_data, next_data)

    data_x = [ range(len(row)) for row in data]
    for j, row in enumerate(data):
        if(columns[j] != 'HR' and columns[j] != 'PULSE' and columns[j] != 'RESP' and columns[j] != 'SpO2'):
            continue
        for i, val in reversed(list(enumerate(row))):
            if math.isnan(val) or val == 0:
                for l,x in enumerate(data):
                    if(len(data[l]) == 0):
                        continue
                    data[l].pop(i)
                    data_x[l].pop(i)
                # data[j].pop(i)
                # data_x[j].pop(i)




    # # print columns
    # for x in data:
    #     print len(x)
    index = 0
    global ITER
    ITER += 1
    for i,row in enumerate(data):
        # print columns[i]
        if(columns[i] != 'HR' and columns[i] != 'PULSE' and columns[i] != 'RESP' and columns[i] != 'SpO2'):
            # print 'skipped',columns[i]
            continue
        # plt.figure(figure*10*ITER + i)
        # plt.suptitle(columns[i])
        id_ = 0
        if(columns[i] == 'HR'):
            id_ = 0
        elif(columns[i] == 'PULSE'):
            id_ = 1
        elif(columns[i] == 'SpO2'):
            id_ = 3
        else:
            id_ = 2
        if len(row) == 0:
            continue
        a = np.array(data_x[i])
        b = np.array([x for x in doseUnits[0]])
        # print len(a), len(b)
        time = a
        TimeVec = []
        ValueVec = []
        TimeVec.append([])
        ValueVec.append([])
        for x in range(len(doseUnits[0])):
            TimeVec.append([])
            ValueVec.append([])

        for k, x in enumerate(time):
            minTime = -10000 #sentinel
            dose = 0
            for j,y in enumerate(b):
                if(minTime == -10000):
                    minTime = x-y
                    dose = j
                    if(x-y < 0):
                        dose = -1
                elif(minTime < 0 and x-y > 0):
                    minTime = x-y
                    dose = j
                elif minTime < 0:
                    if(minTime < x-y):
                        minTime = x-y
                        dose = -1
                if (x-y < minTime and x-y > 0):
                    minTime = x-y
                    dose = j
            time[k] = minTime
            if(dose > 5):
                dose = 5
            # if(dose > -1 and minTime < 0):
                # print minTime
            TimeVec[dose + 1].append(minTime)
            ValueVec[dose + 1].append(row[k])
        count = 0
        for x in TimeVec:
            count += len(x)
        # print count,id_
        iterator = 0
        # NUM = len(TimeVec) / 6
        # NUM = NUM * 100 + 11
        # print NUM, len(TimeVec)
        # print columns[i]
        # print len(TimeVec),
        # print len(ValueVec)
        # for i, x in enumerate(TimeVec):
        #      print len(TimeVec[i]), len(ValueVec[i])
        # continue
        # plt.subplot(NUM)
        for k, x in enumerate(TimeVec):
            if(iterator >= len(DOSECOLORS)):
                iterator = 0
                # NUM += 1
                # plt.subplot(NUM)
            # print iterator
            # if DOSECOLORS[iterator] == 'k':
                # print 'bang'
            j = 0
            while j < len(TimeVec[k]):
                second = j + 5;
                if np.average(TimeVec[k][j:second]) > 700:
                    j = second
                    continue
                color = normalize(k, np.average(TimeVec[k][j:second]))
                color = int(round(color))
                plotRegr(TimeVec[k][j:second], ValueVec[k][j:second],  DOSECOLORS[iterator], color, id_)
                # plt.scatter(np.average(TimeVec[k][j:second]), np.std(ValueVec[k][j:second]), c = DOSECOLORS[iterator])
                j = second;

            # plt.plot(x, ValueVec[k], DOSECOLORS[iterator])
            iterator += 1

def computeThree(predictions, actual):
    negative = 0
    positive = 0
    neg_bot = 0
    pos_bot = 0
    accuracy = 0
    for i, x in enumerate(predictions):
        if(actual[i] == 0 and predictions[i] == 0):
            negative += 1
        if(actual[i] > 0 and predictions[i] > 0):
            positive += 1
        if(actual[i] == 0):
            neg_bot += 1
        if(actual[i] > 0):
            pos_bot += 1    
        if(predictions[i] == actual[i]):
            accuracy += 1
        elif(predictions[i] > 0 and actual[i] > 0):
            accuracy += 1
    # print "predicting"
    # print float(accuracy)/float(len(predictions))
    acc = float(accuracy) / float(len(predictions))
    # print float(positive)/float(pos_bot), ' ', positive, ' ', pos_bot
    # print float(negative)/float(neg_bot), ' ', negative, ' ', neg_bot
    pos = float(positive)/float(pos_bot)
    neg = float(negative)/float(neg_bot)
    acc = (float(pos) + float(neg)) / float(2)
    # print 'if equal: ', float(pos + neg) / float(2)
    return (acc, pos, neg) 

# def trainPredict(LR, data, actual, weight):
#     LR.fit(data, actual, class_weight=weight)
#     prediction = LR.predict(data)
#     return computeThree(prediction, actual)
def calcDifference(old, new):
    return (np.abs(old[0] - new[0]), np.abs(old[0] - new[0]), np.abs(old[0] - new[0]))

def maximizePos(data, actual):
    maxWeight = np.max(actual)
    minWeight = np.min(actual)
    weight = {}
    for x in actual:
        weight[x] = 1
    acc = 0
    pos = 0
    neg = 0
    MaxLoops = 100
    count = 0
    old = (acc,pos,neg)
    convergeSpeed = .1
    new = (0,0,0)
    tolerance = .02
    while(count < MaxLoops):
        count += 1
        LR = sm.LogisticRegression(class_weight=weight)
        LR.fit(data, actual)
        prediction = LR.predict(data)
        new = computeThree(prediction, actual)
        diff  = calcDifference(old, new)
        if(new[2] < .4):
            weight[minWeight] += convergeSpeed
        elif(new[2] > .5 and old[2] > .4 and old[1] < .5 and new[1] > .5):
            return new
        elif(new[2] > .4 and new[2] < .5 + tolerance):
            return new
        else:
            weight[maxWeight] += convergeSpeed
    # print 'fail'
    return new

def maximizeNeg(data, actual):
    maxWeight = np.max(actual)
    minWeight = np.min(actual)
    weight = {}
    for x in actual:
        weight[x] = 1
    acc = 0
    pos = 0
    neg = 0
    MaxLoops = 100
    count = 0
    old = (acc,pos,neg)
    convergeSpeed = .1
    new = (0,0,0)
    tolerance = .02
    while(count < MaxLoops):
        count += 1
        LR = sm.LogisticRegression(class_weight=weight)
        LR.fit(data, actual)
        prediction = LR.predict(data)
        new = computeThree(prediction, actual)
        # diff  = calcDifference(old, new)
        if(new[1] < .4):
            weight[maxWeight] += convergeSpeed
        elif(new[1] > .5 and old[1] > .4 and old[1] < .5 and new[2] > .5):
            # print '1'
            return new
        elif(new[1] > .4 and new[1] < (.5 + tolerance)):
            # print '2'
            return new
        else:
            weight[minWeight] += convergeSpeed
    # print 'fail'
    # print weight
    return new

def maximizeAcc(data, actual):
    maxWeight = np.max(actual)
    minWeight = np.min(actual)
    weight = {}
    for x in actual:
        weight[x] = 1
    acc = 0
    pos = 0
    neg = 0
    MaxLoops = 100
    count = 0
    old = (acc,pos,neg)
    convergeSpeed = .1
    tolerance = .03
    new = (0,0,0)
    while(count < MaxLoops):
        count += 1
        LR = sm.LogisticRegression(class_weight=weight)
        LR.fit(data, actual)
        prediction = LR.predict(data)
        new = computeThree(prediction, actual)
        if(np.abs(new[1] - new[2]) < tolerance):
            # print np.abs(new[1] - new[2])
            return new
        elif(new[1] < new[2]):
            weight[maxWeight] += convergeSpeed
        else:
            weight[minWeight] += convergeSpeed
    # print 'fail'
    # print weight
    return new
def threadFunc(data, actual, lock, output):
    # print 'starting'
    neg = maximizeNeg(data, actual)
    # print 'neg done'
    pos = maximizePos(data, actual)
    # print 'pos done'
    acc = maximizeAcc(data, actual)
    lock.acquire()
    print '---------------------'
    print output
    print neg
    print pos
    print acc
    print '---------------------'
    lock.release()




def my_accuracy(data, actual, threshold_number):
    accuracy = 0
    negative = 0
    positive = 0
    neg_bot = 0
    pos_bot = 0
    for i, x in enumerate(data):
        prediction = 0
        if(np.abs(data[i]) > threshold_number):
            prediction = 1
        if(actual[i] == 0 and prediction == 0):
            negative += 1
        if(actual[i] > 0 and prediction > 0 ):
            positive += 1
        if(actual[i] == 0):
            # print
            neg_bot += 1
        if(actual[i] > 0):
            pos_bot += 1

        if(prediction == actual[i]):
            accuracy += 1
        elif(prediction > 0 and actual[i] > 0):
            accuracy += 1

    print "My predicting"
    print float(accuracy)/float(len(data))
    print float(positive)/float(pos_bot), ' ', positive, ' ', pos_bot
    print float(negative)/float(neg_bot), ' ', negative, ' ', neg_bot
    pos = float(positive)/float(pos_bot)
    neg = float(negative)/float(neg_bot)
    print 'if equal: ', float(pos + neg) / float(2) 

def Test(fileName, nrows_, LR, threshold_number):
    data = 0
    columns = []
    if(nrows_ == 0):
        data = pd.read_csv(fileName, sep='\s+', header= 0, skiprows=[1], na_values = ['-'])
        columns = data.columns
        data = data.values
        data_tmp = np.transpose(data)
        data = [[]]
        for i,row in enumerate(data_tmp):
            if i == 0:
                continue
            data.append( [x for x in row])
        #print data
    else:
        in_data = pd.read_csv(fileName, sep='\s+', header=0, skiprows=[1], na_values = ['-'], nrows = nrows_)
        columns = in_data.columns
        next_data = pd.read_csv(fileName, sep='\s+', header=None, skiprows=nrows_+2, na_values = ['-'], index_col =False)
        data = join(in_data, next_data)

    data_x = [ range(len(row)) for row in data]
    for j, row in enumerate(data):
        for i, val in reversed(list(enumerate(row))):
            if math.isnan(val) or val == 0:
                data[j].pop(i)
                data_x[j].pop(i)
    for i,row in enumerate(data):
        # print columns[i]
        if(columns[i] != 'HR'):# and columns[i] != 'PULSE' and columns[i] != 'RESP'):
            continue
        if len(row) == 0:
            continue
        a = np.array(data_x[i])
        b = np.array([x for x in doseUnits[0]])

        time = a
        TimeVec = []
        ValueVec = []
        TimeVec.append([])
        ValueVec.append([])

        for x in range(len(doseUnits[0])):
            TimeVec.append([])
            ValueVec.append([])

        for k, x in enumerate(time):
            minTime = -10000 #sentinel
            dose = 0
            for j,y in enumerate(b):
                if(minTime == -10000):
                    minTime = x-y
                    dose = j
                    if(x-y < 0):
                        dose = -1
                elif(minTime < 0 and x-y > 0):
                    minTime = x-y
                    dose = j
                elif minTime < 0:
                    if(minTime < x-y):
                        minTime = x-y
                        dose = -1
                if (x-y < minTime and x-y > 0):
                    minTime = x-y
                    dose = j
            time[k] = minTime
            if(dose > 5):
                dose = 5
            # if(dose > -1 and minTime < 0):
                # print minTime
            TimeVec[dose + 1].append(minTime)
            ValueVec[dose + 1].append(row[k])
        for k, x in enumerate(TimeVec):
            j = 0
            coef = []
            while j < len(TimeVec[k]):
                second = j + 5;
                if np.average(TimeVec[k][j:second]) > 300:
                    j = second
                    continue
                LR_ = sm.LinearRegression()
                # print len(x_col), len(y_col)
                x_col = TimeVec[k][j:second]
                y_col = ValueVec[k][j:second]
                x_col = np.array(x_col)
                x_col = x_col[:, np.newaxis]
                y_col = np.transpose(y_col)
                # print x_col
                # print y_col
                LR_.fit(x_col, y_col)
                coef.append(LR_.coef_)
                j = second;
            if len(coef) < 1:
                continue
            predictions = LR.predict(coef)
            accuracy(predictions, k)
            my_accuracy(coef, k, threshold_number)


if __name__ == '__main__':
    global HR_COEF
    global HR_COL
    global PULSE_COL
    global PULSE_COEF
    global COUNT
    global RESP_COEF
    global RESP_COL
    global SP02_COEF
    global SP02_COL
    ITER = -1
    # doseUnits = displayDoses('13508-doses.csv', 1, '3184-06-20 19:07:21')
    # # print doseUnits
    # readAndCalculate('13508-n.txt', 60, 1, '13508', 'b', 293, doseUnits)

    doseUnits = displayDoses('21673-doses.csv', 1, '3325-02-19 00:33:59')
    readAndCalculate('21673-n.txt', 60, 1, '13508', 'b', 0, doseUnits)

    # doseUnits = displayDoses('8557-doses.csv', 1, '2746-06-10 04:25:32')
    # readAndCalculate('8557-n.txt', 60, 1, '8557', 'b', 1175, doseUnits)

    # doseUnits = displayDoses('11995-doses.csv', 1, '2617-10-03 22:44:28')
    # readAndCalculate('11995-n.txt', 60, 1, '11995', 'b', 76, doseUnits)

    # doseUnits = displayDoses('21805-doses.csv', 1, '3388-07-07 11:49:38')
    # readAndCalculate('21805-n-2.txt', 60, 1, '21805', 'b', 0, doseUnits)
    # global ITER
    # ITER -= 1
    # doseUnits = displayDoses('21805-doses.csv', 1, '3388-07-07 02:16:38')
    # readAndCalculate('21805-n.txt', 60, 1, '21805', 'b', 0, doseUnits)
    # exit()
    # print HR_COEF
    # print HR_COL
    # print len(HR_COEF)
    # print COUNT

    LR_col = [HR_COEF, RESP_COEF, SP02_COEF]
    LR_col = np.transpose(LR_col)

    LR1_col = [HR_COEF, RESP_COEF]
    LR1_col = np.transpose(LR1_col)

    LR2_col = [HR_COEF, SP02_COEF]
    LR2_col = np.transpose(LR2_col)

    LR3_col = [RESP_COEF, SP02_COEF]
    LR3_col = np.transpose(LR3_col)

    RESP_COEF = np.array(RESP_COEF)
    RESP_COEF = RESP_COEF[:, np.newaxis]
    SP02_COEF = np.array(SP02_COEF)
    SP02_COEF = SP02_COEF[:,  np.newaxis]

    lock = Lock()

    Process(target=threadFunc, args=(HR_COEF, HR_COL, lock, 'HR')).start()
    Process(target=threadFunc, args=(RESP_COEF, RESP_COL, lock, 'RESP')).start()
    Process(target=threadFunc, args=(PULSE_COEF, PULSE_COL, lock, 'PULSE')).start()
    Process(target=threadFunc, args=(SP02_COEF, SP02_COL, lock, 'Sp02')).start()
    Process(target=threadFunc, args=(LR_col, HR_COL, lock, 'HR, RESP, Sp02')).start()
    Process(target=threadFunc, args=(LR1_col, HR_COL, lock, 'HR, RESP')).start()
    Process(target=threadFunc, args=(LR2_col, HR_COL, lock, 'HR, Sp02')).start()
    Process(target=threadFunc, args=(LR3_col, HR_COL, lock, 'RESP, Sp02')).start()

    exit()

# # threadFunc(HR_COEF, HR_COL, lock, 'HR')
# # threadFunc(RESP_COEF, RESP_COL, lock, 'RESP')
# # threadFunc(PULSE_COEF, PULSE_COL, , 'PULSE')
# # threadFunc(SP02_COEF, SP02_COL, , 'Sp02')
# # threadFunc(LR_col, HR_COL, , 'HR, RESP, Sp02')
# # threadFunc(LR1_col, HR_COL, , 'HR, RESP')
# # threadFunc(LR2_col, HR_COL, , 'HR, RESP')
# # threadFunc(LR3_col, HR_COL, , 'RESP, Sp02')
# exit()
# weight = {}
# for x in range(5):
#     weight[x] = 1
# weight[0] = 5.1
# weight[4] = 6.6

# LR = sm.LogisticRegression(class_weight=weight)

# weight2 = {}
# for x in range(5):
#     weight2[x] = 1
#     # print x
# weight2[0] = 4.2
# weight2[4] = 4

# LR2 = sm.LogisticRegression(class_weight=weight2)
# weight3 = {}
# for x in range(5):
#     weight3[x] = 1
#     # print x
# weight3[0] = 5.2
# weight3[4] = 6.75
# LR3 = sm.LogisticRegression(class_weight=weight3)

# weight4 = {}
# for x in range(5):
#     weight4[x] = 1
#     # print x
# weight4[0] = 4
# weight4[4] = 5.1
# LR4 = sm.LogisticRegression(class_weight=weight4)



# weight5 = {}
# for x in range(5):
#     weight5[x] = 1
# weight5[0] = 4.6
# weight5[4] = 5
# LR5 = sm.LogisticRegression(class_weight = weight5)
# LR_col = [HR_COEF, RESP_COEF, SP02_COEF]
# LR_col = np.transpose(LR_col)

# weight6 = {}
# for x in range(5):
#     weight6[x] = 1
# # 4.5
# weight6[0] = 4.6
# weight6[4] = 1
# LR6 = sm.LogisticRegression(class_weight = weight6)
# LR1_col = [HR_COEF, RESP_COEF]
# LR1_col = np.transpose(LR1_col)

# weight7 = {}
# for x in range(5):
#     weight7[x] = 1
# weight7[0] = 3.7
# weight7[4] = 4.75
# LR7 = sm.LogisticRegression(class_weight = weight7)
# LR2_col = [HR_COEF, SP02_COEF]
# LR2_col = np.transpose(LR2_col)

# weight8 = {}
# for x in range(5):
#     weight8[x] = 1
# weight8[0] = 4.6
# weight8[4] = 6
# weight8[3] = 1
# weight8[2] = 1
# LR8 = sm.LogisticRegression(class_weight = weight8)
# LR3_col = [RESP_COEF, SP02_COEF]
# LR3_col = np.transpose(LR3_col)

# # print LR_col
# # for i,x in enumerate(HR_COL):
# #     if(HR_COL != PULSE_COL):
# #         print 'no'

# # exit()
# LR.fit(HR_COEF, HR_COL)
# # threadFunc()
# # print 'LENGTH'
# # print len(PULSE_COEF)
# # print len(PULSE_COL)
# # PULSE_COEF = np.array(PULSE_COEF)
# # PULSE_COEF = PULSE_COEF[:, np.newaxis]

# # LR2.fit(PULSE_COEF,PULSE_COL)
# # Test('13508-n.txt', 293, LR)
# LR3.fit(RESP_COEF, RESP_COL)
# LR4.fit(SP02_COEF, SP02_COL)
# LR5.fit(LR_col, HR_COL)
# LR6.fit(LR1_col, HR_COL)
# LR7.fit(LR2_col, HR_COL)
# LR8.fit(LR3_col, HR_COL)

# # print HR_COL

# # plt.plot(HR_COEF, LR.predict(HR_COEF), 'g+')
# # plt.plot(HR_COEF, HR_COL, 'b+')
# threshold_percent = 0.10

# # plt.plot([x for i,x in enumerate(HR_COEF) if HR_COL[i] == 0], [1for i,x in enumerate(HR_COEF) if HR_COL[i] == 0], 'g+')
# plt.plot(HR_COEF, HR_COL, 'b+')
# # print np.shape(PULSE_COEF)
# # print PULSE_COEF
# plt.plot(PULSE_COEF, np.array(PULSE_COL) +.2, 'g+')
# plt.plot(RESP_COEF, np.array(RESP_COL) +.6, 'r+')
# plt.plot(SP02_COEF, np.array(SP02_COL) +.8, 'k+')

# # ABSHR_COEF = [np.abs(x) for i,x in enumerate(HR_COEF) if HR_COL[i] == 0]
# # ABSHR_COEF = sorted(ABSHR_COEF)
# # threshold_number = len(ABSHR_COEF) / 100
# # threshold_number = int(threshold_number * (100 - 100 *threshold_percent))
# # threshold_number = ABSHR_COEF[threshold_number]

# # ABSPU_COEF = [np.abs(x) for i,x in enumerate(PULSE_COEF) if PULSE_COL[i] == 0]
# # ABSPU_COEF = sorted(ABSPU_COEF)
# # threshold_number2 = len(ABSPU_COEF) / 100
# # threshold_number2 = int(threshold_number2 * (100 - 100 *threshold_percent))
# # threshold_number2 = ABSPU_COEF[threshold_number2]

# # ABSRE_COEF = [np.abs(x) for i,x in enumerate(RESP_COEF) if RESP_COL[i] == 0]
# # ABSRE_COEF = sorted(ABSRE_COEF)
# # threshold_number3 = len(ABSRE_COEF) / 100
# # threshold_number3 = int(threshold_number3 * (100 - 100 *threshold_percent))
# # threshold_number3 = ABSRE_COEF[threshold_number3]

# # ABSSP_COEF = [np.abs(x) for i,x in enumerate(SP02_COEF) if SP02_COL[i] == 0]
# # ABSSP_COEF = sorted(ABSSP_COEF)
# # threshold_number4 = len(ABSSP_COEF) / 100
# # threshold_number4 = int(threshold_number4 * (100 - 100 *.14))
# # threshold_number4 = ABSSP_COEF[threshold_number4]
# # print threshold_number
# # exit()
# # Test('13508-n.txt', 293, LR, threshold_number)

# print 'alldataTest'
# predictions = LR.predict(HR_COEF)
# # my_accuracy(HR_COEF, HR_COL, threshold_number)
# accuracy(predictions, HR_COL)

# # predictions2 = LR2.predict(PULSE_COEF)
# # accuracy(predictions2, PULSE_COL)
# # my_accuracy(PULSE_COEF, PULSE_COL, threshold_number2)

# predictions3 = LR3.predict(RESP_COEF)
# accuracy(predictions3, RESP_COL)
# # my_accuracy(RESP_COEF, RESP_COL, threshold_number3)

# predictions4 = LR4.predict(SP02_COEF)
# accuracy(predictions4, SP02_COL)

# predictions5 = LR5.predict(LR_col)
# accuracy(predictions5, SP02_COL)

# predictions6 = LR6.predict(LR1_col)
# accuracy(predictions6, SP02_COL)

# predictions7 = LR7.predict(LR2_col)
# accuracy(predictions7, SP02_COL)

# predictions8 = LR8.predict(LR3_col)
# accuracy(predictions8, SP02_COL)
# # my_accuracy(SP02_COEF, SP02_COL, threshold_number4)


# # conglomerate()

# # global HR
# # global PULSE

# # print np.average(HR), np.average(PULSE)
# # print np.average(HR_sens), np.average(PULSE_sens)
# # print np.average(HR_spec), np.average(PULSE_spec)
# # plt.show()
