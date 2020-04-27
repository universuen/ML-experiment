from Perceptron import Perceptron
from sklearn.datasets import fetch_20newsgroups_vectorized
import numpy as np


# 数据获取和处理
train_set = fetch_20newsgroups_vectorized('train')
test_set = fetch_20newsgroups_vectorized('test')

train_x = train_set
train_y = np.array(train_set.target)

test_x = test_set
test_y = np.array(test_set.target)

accuracy = np.zeros(20)


def Perceptron_test():
    # 实例化感知机
    perceptron = Perceptron.Perceptron()
    # 针对每一个类别进行二元分类并计算准确率
    for i in range(20):
        temp_train_y = train_y
        temp_test_y = test_y
        # 规格化Target
        for j in range(temp_train_y.shape[0]):
            if temp_train_y[j] == i:
                temp_train_y[j] = 1
            else:
                temp_train_y[j] = -1
        for j in range(temp_test_y.shape[0]):
            if temp_test_y[j] == i:
                temp_test_y[j] = 1
            else:
                temp_test_y[j] = -1
        # 开始训练
        print('开始训练')
        perceptron.train(train_x, temp_train_y)
        result = perceptron.predict(test_x)
        correct = 0
        for j, k in zip(result, temp_test_y):
            correct += (j == k)
        print(correct/len(result))
        accuracy[i] = correct/len(result)
        np.save('accuracy.npy', accuracy)



Perceptron_test()