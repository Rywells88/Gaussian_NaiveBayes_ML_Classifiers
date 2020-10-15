import numpy as np
from numpy import pi, exp
from scipy.io import loadmat
ATTR_NAMES = [1,2,3,4,5,6,7,8,9,10]


def __calculate_mean_variance(self):
    self.__mean_variance = {}
    for c in self.__training_set["Class"].unique():
        filtered_set = self.__training_set[
            (self.__training_set['Class'] == c)]
        m_v = {}
        for attr_name in ATTR_NAMES:
            m_v[attr_name] = []
            m_v[attr_name].append(filtered_set[attr_name].mean())
            m_v[attr_name].append(
                math.pow(filtered_set[attr_name].std(), 2))
        self.__mean_variance[c] = m_v

def main():
    global training_data
    global testing_data

    training_data= np.transpose(training_data)
    testing_data =np.transpose(testing_data)