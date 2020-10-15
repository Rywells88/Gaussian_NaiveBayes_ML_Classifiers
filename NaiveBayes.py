import numpy as np
from numpy import pi, exp
from scipy.io import loadmat
import matplotlib.pyplot as plt

data = loadmat('a1digits.mat') #dictionary types

training_data = data['digits_train']
testing_data = data['digits_test']

fig, ax = plt.subplots(1, 10, figsize = (18,10))

def convert(data):
    binary = np.zeros((10,len(data[0]),64))
    for i in range(10):
        for j in range(len(data[0])):
            for m in range(64):
                if(data[i][j][m] >= 0.5):
                    binary[i][j][m] = 1
                else:
                    binary[i][j][m] = 0
    return binary

def find_PbCk(result):
    prob_1 = np.zeros((10,64))

    for i in range(10):
        for m in range(64):
            countOnes = 0
            for j in range(700):
                if(result[i][j][m] == 1):
                    countOnes += 1              # find probabiliy that each pixel is 1.
            prob_1[i][m] = (countOnes/700)
    return prob_1



def predict(X, prob_table): # test samples
    PcK = 1 / 10
    answer_list = []

    prob_list = []
    for Cls in range(10):

        for m in range(64):
            #print(X)
            if (X[m] == 1):
                prob_list.append(prob_table[Cls][m])
            else:
                prob_list.append(1-prob_table[Cls][m])
        product = (np.prod(prob_list))
        prob_list.clear()
        result = product * PcK
        answer_list.append(result)

    return answer_list

def main():
    global training_data
    global testing_data



    training_data = np.transpose(training_data)
    testing_data = np.transpose(testing_data)

    result_training = convert(training_data)
    result_testing = convert(testing_data)

    probability_table = find_PbCk(result_training)
    correct = 0
    error_count = np.zeros(10)
    for i in range(10):
        for j in range(400):
            ans = predict(result_testing[i][j], probability_table)

            print(max(ans), "in Class :", ans.index(max(ans)))

            if( i == ans.index(max(ans))):
                correct += 1
            else:
                error_count[i] += 1


    print(correct/4000)
    print(error_count)

    names = ['1','2','3','4','5','6','7','8','9','10']


    for i in range(10):
        ax[i].imshow(probability_table[i].reshape(8, 8), cmap='gray',interpolation='nearest')
        ax[i].set_title(names[i])
    plt.show()


main()