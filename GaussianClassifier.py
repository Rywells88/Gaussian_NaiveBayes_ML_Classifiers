import numpy as np
from numpy import pi, exp
import collections
from scipy.io import loadmat
import matplotlib.pyplot as plt

data = loadmat('a1digits.mat') #dictionary types

training_data = data['digits_train']
testing_data = data['digits_test']




# this function returns the prior probabilities of the 2 classes taking in the label set as inputs
def calc_classPrior():
    return 0.01

def calc_mean(k, pixel): # k is the class for the mean we are trying to generate
    global training_data
    # for the number of training data points in class K, sum together all data points from each example
    sumK = 0

    #newArr = training_data[k-1][3].reshape(8,8)
    #plt.imshow(newArr, cmap='gray')
    #plt.show()

    for i in range(len(training_data[k-1])): # 700
        sumK += training_data[k-1][i][pixel] # go through example 0, 1,2,3,... 700

    return (sumK/len(training_data[k]))

def variance(pixel):
    global training_data
    var = 0
    sumK = 0
    for i in range(10): # for each number
        for j in range(700): # each training case for each number


            var += (training_data[i][j][pixel] - calc_mean(i, pixel))**2 # pixel is a value from 0-64

    return (var)

def feature_in_class_probability(x, k):
    var = 0
    sum = 0

    for i in range(64):
        # D is 64, there's 64 features
        sum += (x[i] - (calc_mean(k, i)))**2
        var += variance(i)

    print(var)
    power = -64/5
    firstTerm = ((2 * pi * var**2)**power)

    product = firstTerm * exp(-1/2*(var)*sum)

    return product






def main():
    global training_data
    global testing_data

    training_data= np.transpose(training_data)
    testing_data =np.transpose(testing_data)


    # -----  below are some examples to access the data.  ----------

    #print(training_data[0][0]) # digit 1, example 1 all pixels


    #print(training_data[1][0]) # digit 2, example 1 all pixels

    #print("mean for pixel 40 in class 4 is: ",  calc_mean(5, 40)) # number 4, pixel 40
    probability_per_class = []

    for i in range (1, 10):
        probability = feature_in_class_probability(training_data[2][1], i)
        print(probability)
        probability_per_class.append(probability)

    print(max(probability_per_class))






""""
    picture4 = np.zeros((8,8))
    counter = 0
    for i in range(8):
        for j in range(8):
            print(counter)
            picture4[i][j]=  calc_mean(4, counter)
            counter += 1

    plt.imshow(picture4, cmap = 'gray')
    plt.show()
"""
    #totalVariance = 0
   # for i in range(0,64):
      # totalVariance += variance(i)
    #print(totalVariance/70000*10)

    #for i in range(10):
        #Pxc = feature_in_class_probability( , i,totalVariance)

    #ans = pxc * calc_classPrior()





main()




