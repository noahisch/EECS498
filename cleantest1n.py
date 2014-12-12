

import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import sklearn.linear_model as sm
import math
import sklearn.svm as sv

COLORS = ['b+', 'r+', 'g+', 'k+', 'c+', 'c+']
ITER = 0
DOSECOLORS = ['y+', 'c+', 'b+', 'g+', 'r+', 'k+']

HR = []
HR_sens = []
HR_spec = []
PULSE = []
PULSE_sens = []
PULSE_spec = []


def plotRegr(x_col, y_col, figure, color, doses, col):

    index = 0
    plt.figure(figure)
   
    coef = []
    inter = []
    time = []
    while index < len(x_col):
        index2 = index + 6
        if index2 > len(x_col):
            index2 = len(x_col)

        x_col_w = np.array(x_col[index:index2])
        y_col_w = np.array(y_col[index:index2])
        regr = sm.LinearRegression()
        x_col_w_fit = x_col_w - np.min(x_col_w)
        #x_col_w_fit = [x_col_w_fit, x_col_w_fit**2, x_col_w_fit**3]
        x_col_w_fit = [x_col_w_fit]
        x_col_w_fit = np.transpose(x_col_w_fit)
        y_col_w = y_col_w[:, np.newaxis]
        x_col_w = x_col_w[:, np.newaxis]
        regr.fit(x_col_w_fit, y_col_w)

        coef.append(regr.coef_)
        inter.append(regr.intercept_)
        time.append(np.average(x_col_w))

        index = index2

    dist_plot = []
    dose_plot = []
    for i,x in enumerate(time):
        dist = -1000
        dose_ = 0
        for j,dose in enumerate(doses[0]):
            if(doses[1][j] == 0):
                continue
            if(dose >= x):
                if(dist == -1000):
                    dist = x-dose
                    dose_ = 0 
                continue
            if dist <= 30 and x-dose > dist:
                dist = x-dose
                dose_ = doses[1][j]
            elif dist > x-dose and x-dose >30:
                dist = x-dose
                dose_ = doses[1][j]

        dist_plot.append(dist)
        dose_plot.append(dose_)
    ##########
    coef_category = [[] for i in DOSECOLORS]
    dist_category = [[] for i in DOSECOLORS]
    color_category = [[] for i in DOSECOLORS]
    for i, x in enumerate(dist_plot):
        color = int(math.ceil(dose_plot[i]))
        if color > 5:
            color = 5
        if x < -100 or x > 400:
            continue
       # plt.plot(x, coef[i], DOSECOLORS[color])
        coef_category[color].append(coef[i])
        dist_category[color].append(x)
        color_category[color].append((coef[i], x))
    x__ = []
    y__ = []
    for i, color in enumerate(color_category):
        color_category[i] = sorted(color, key=lambda x:x[1])
        index = 0
        while index < len(color_category[i]):
            index2 = index
            for k, val in enumerate(color_category[i]):
                if k < index:
                    continue
                #if val[1]- abs(color_category[i][index][1]/10) <= color_category[i][index][1]:
                if val[1] -5 <= color_category[i][index][1]:
                    index2+=1
                    if(index2 >= len(color_category[i])):
                        break
          #  print len(color_category[i])
          #  print index2
          #  print color_category[i]
            avg1 = []
            avg2 = []
            index2 = len(color_category[i])
            for l in range(index, index2):
                avg1.append(color_category[i][l][1])
                avg2.append(color_category[i][l][0][0][0])
          #  print avg1,avg2
           # plt.scatter(np.average(avg1), np.std(avg2), c=DOSECOLORS[i][0])
            x__.append(avg2)
            print avg2
            ins = i;
            if np.average(avg1) > 100:
                if ins > 0:
                    ins = 1
            if(np.average(avg1) > 180):
                ins = 0
            y__.append([ins]*len(avg2))

            tmp = np.array(avg2)
            #plt.scatter(np.average(avg1), np.average(np.absolute(tmp)), c = DOSECOLORS[i][0])
          #  print np.average(avg1), np.std(avg2)
            index = index2

    LR = sm.LogisticRegression()
    #x_final = np.array(np.zeros((44,1)))
    if(len(x__) < 2):
        return
    good = 0
    for y in y__:
        if y__[0] != y:
            good = 1
    if(good == 0):
        return
    x_final = [[x] for x in x__]
    y__ = np.array(y__)

    #LR = sv.SVC()
    print(len(x_final),len(y__))
    print x_final, y__

    LR.fit(x_final, y__)
    
    plt.plot(x_final, y__, 'g+')
    y___ = LR.predict(x_final)
    x_final = sorted(x_final)
    plt.plot(x_final, LR.predict(x_final), 'b')
    count = 0
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    for i, y in enumerate(y__):
        print y__[i], y___[i]
        if (y__[i] > 0 and y___[i] > 0) or (y__[i] == y___[i]):
            count += 1
        if (y__[i] > 0 and y___[i] > 0):
            count1 += 1
        if(y__[i] > 0 and y___[i] == 0):
            count2 += 1
        if(y__[i] == 0 and y___[i] == 0):
            count3 += 1
        if(y__[i] == 0 and y___[i] > 0):
            count4 += 1
    print 'percent ', float(count)/float(len(y__))


    if(col == 'HR'):
        global HR
        HR.append(float(count)/float(len(y__)))
        HR_sens.append(float(count1)/float(count1+count2+1))
        HR_spec.append(float(count3)/float(count3+count4+1))
    elif(col == 'PULSE'):
        global PULSE
        PULSE.append(float(count)/float(len(y__)))
        PULSE_sens.append(float(count1)/float(count1+count2+1))
        PULSE_spec.append(float(count3)/float(count3+count4+1))
    return
    #############################
    index = 0
    dist_plot_plot = []
    std_coef = []
    dose_plot_plot = []
    print len(dist_plot), len(coef)
    while index < len(dist_plot):
        index2 = index + 10
        if index2 > len(dist_plot):
            index2 = len(dist_plot)
        dist_plot_plot.append(np.average(dist_plot[index:index2]))
        std_coef.append(np.std(coef[index:index2]))
        dose_plot_plot.append(np.average(dose_plot[index:index2]))
        index +=1
    global ITER
    plt.plot(dist_plot_plot, std_coef, COLORS[ITER])
    #print dist_plot_plot, std_coef



    

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
        #print in_data
        data = join(in_data, next_data)

    data_x = [ range(len(row)) for row in data]
    for j, row in enumerate(data):
        for i, val in reversed(list(enumerate(row))):
         #   print data[j][i], j, i
            if math.isnan(val) or val == 0:
                data[j].pop(i)
                data_x[j].pop(i)

    colors = ['b','g','r','c','b','m','k', 'y','b', 'g', 'r', 'c']



    plt.figure(figure)
    print columns
    index = 0

    for i,row in enumerate(data):
        plt.figure(figure*10 + i)
        plt.suptitle(columns[i])
     #   red_patch = mpatches.Patch(color='red', label='dose = 4')
     #   y_patch = mpatches.Patch(color='yellow', label='dose = 1')
     #   c_patch = mpatches.Patch(color='cyan', label='dose = 2')
     ##   b_patch = mpatches.Patch(color='blue', label='dose = 3')
      #  bl_patch = mpatches.Patch(color='black', label='dose = 5')
      #  plt.legend(handles=[y_patch, c_patch, b_patch, red_patch, bl_patch])
        if len(row) == 0:
            continue
        std = []
        avg = []
        time = []
        index = 0
        while index < len(row):
            index2 = index + 5
            if(len(row) < index2):
                index2 = len(row)
            std.append(np.std(row[index:index2]))
            avg.append(np.average(row[index:index2]))
            time.append(np.average(data_x[i][index:index2]))
            index = index2

       # col = {'HR' = 1, 'PULSE' = 2}
        plotRegr(time, row, figure*10+i, 'r', doseUnits, columns[i])
    global ITER
    ITER += 1

ITER = 0
doseUnits = displayDoses('13508-doses.csv', 1, '3184-06-20 19:07:21')
readAndCalculate('13508-n.txt', 60, 1, '13508', 'b', 293, doseUnits)

doseUnits = displayDoses('21673-doses.csv', 1, '3325-02-19 00:33:59')
readAndCalculate('21673-n.txt', 60, 1, '13508', 'b', 0, doseUnits)

doseUnits = displayDoses('8557-doses.csv', 1, '2746-06-10 04:25:32')
readAndCalculate('8557-n.txt', 60, 1, '8557', 'b', 1175, doseUnits)

doseUnits = displayDoses('11995-doses.csv', 1, '2617-10-03 22:44:28')
readAndCalculate('11995-n.txt', 60, 1, '11995', 'b', 76, doseUnits)

doseUnits = displayDoses('21805-doses.csv', 1, '3388-07-07 11:49:38')
readAndCalculate('21805-n-2.txt', 60, 1, '21805', 'b', 0, doseUnits)

doseUnits = displayDoses('21805-doses.csv', 1, '3388-07-07 02:16:38')
readAndCalculate('21805-n.txt', 60, 1, '21805', 'b', 0, doseUnits)



global HR
global PULSE

print np.average(HR), np.average(PULSE)
print np.average(HR_sens), np.average(PULSE_sens)
print np.average(HR_spec), np.average(PULSE_spec)
plt.show()
