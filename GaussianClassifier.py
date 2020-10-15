import numpy as np
from numpy import pi, exp
from scipy.io import loadmat
import matplotlib.pyplot as plt

data = loadmat('a1digits.mat') #dictionary types

training_data = data['digits_train']
testing_data = data['digits_test']

class_dict = {}

fig, ax = plt.subplots(1, 10, figsize = (18,10))

# this function returns the prior probabilities of the 2 classes taking in the label set as inputs
def calc_classPrior():
    return 0.01

def calc_mean(k, pixel): # k is the class for the mean we are trying to generate, pixel is a value from 1-64
    global training_data

    sumK = 0

    for i in range(len(training_data[k])): # 700
        sumK += training_data[k][i][pixel] # go through example 0, 1,2,3,... 700

    #goes through each example for a given class and pixel, and finds the mean value of that pixel.
    return (sumK/len(training_data[k]))

def getVariance(x):
   return (np.var(x))


def feature_in_class_probability(x,variance, means):
    answer_list = []

    for j in range(10):
        sumTotal = 0
        sub = 0

        sub = (np.subtract(x,means[j]))**2
        sumTotal = np.sum(sub)

        product = exp(-sumTotal)

        PCkx = product * calc_classPrior()

        answer_list.append(PCkx)

    return answer_list




def main():
    global training_data
    global testing_data

    training_data= np.transpose(training_data)
    testing_data =np.transpose(testing_data)

    means = np.zeros((10, 64))

    for i in range(10):
        for j in range(64):
            means[i][j] = calc_mean(i, j)

    V = getVariance(training_data)

    correctCount = 0
    error_count = np.zeros(10)

    for k in range(10):
        for example in range(400):
            PxCk = feature_in_class_probability(testing_data[k][example], V, means)

            print(max(PxCk), " in class ",PxCk.index(max(PxCk)))
            print("\n")

            if(k == PxCk.index(max(PxCk))):
                correctCount += 1
            else:
                error_count[k] += 1

    print(correctCount/4000)
    print(error_count)

    names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

    for i in range(10):
        ax[i].imshow(means[i].reshape(8, 8), cmap='gray', interpolation='nearest')
        ax[i].set_title(names[i])
        ax[0].text(0,-5, "Standard deviation : 0.3049" )
    plt.show()

main()




