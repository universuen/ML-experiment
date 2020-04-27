import numpy as np


class Perceptron:
    def __init__(self, learning_rate = 1):
        self.w = 0
        self.b = 0
        self.lr = learning_rate
        self.len = 0

    def train(self, x, y):
        self.len = x.data.shape[1]  # 总词数
        self.w = np.zeros(self.len)
        print('进入while循环')
        print(self.len)
        data = x.data.data
        indptr = x.data.indptr
        indices = x.data.indices
        for i in range(x.data.shape[0]):
            if i % 100 == 99:
                print('TRAIN\t', i + 1)
            # 从稀疏矩阵中提取一个行向量
            temp = np.zeros(x.data.shape[1])
            for j in range(indptr[i+1]):
                # print(j)
                temp[indices[j]] = data[j]
            # print('向量取出完毕')
            # 权重优化
            if y[i] * (np.dot(self.w, temp)+self.b) <= 0:
                self.w += self.lr * y[i] * temp
                self.b += self.lr * y[i]
                # totally_correct = False
            # print(self.w)

    def predict(self, x):
        result = []
        data = x.data.data
        indptr = x.data.indptr
        indices = x.data.indices
        for i in range(x.data.shape[0]):
            if i % 100 == 99:
                print('TEST\t', i + 1)
            # 从稀疏矩阵中提取一个行向量
            temp = np.zeros(x.data.shape[1])
            for j in range(indptr[i + 1]):
                temp[indices[j]] = data[j]
            # 求解结果
            if np.dot(self.w, temp)+self.b <= 0:
                result.append(-1)
            else:
                result.append(1)
        return result
