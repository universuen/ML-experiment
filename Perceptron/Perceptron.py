import numpy as np


class Perceptron:
    def __init__(self, learning_rate = 1, epochs = 1):
        self.w = 0
        self.b = 0
        self.lr = learning_rate
        self.len = 0
        self.epochs = epochs

    def train(self, x, y):
        self.len = x.shape[1]  # 总词数
        self.w = np.zeros(self.len)
        data = x.data
        indptr = x.indptr
        indices = x.indices
        for _ in range(self.epochs):
            for i in range(x.shape[0]):
                # if i % 1000 == 999:
                #     print('TRAIN\t', i + 1)
                # 从稀疏矩阵中提取一个行向量
                temp = np.zeros(x.shape[1])
                for j in range(indptr[i], indptr[i+1]):
                    temp[indices[j]] = data[j]
                # 如果是误分类点则梯度下降
                if y[i] * (np.dot(self.w, temp)+self.b) <= 0:
                    self.w += self.lr * y[i] * temp
                    self.b += self.lr * y[i]


    def predict(self, x):
        result = []
        data = x.data
        indptr = x.indptr
        indices = x.indices
        for i in range(x.shape[0]):
            # if i % 1000 == 999:
            #     print('TEST\t', i + 1)
            # 从稀疏矩阵中提取一个行向量
            temp = np.zeros(x.shape[1])
            for j in range(indptr[i], indptr[i+1]):
                temp[indices[j]] = data[j]
            # 求解结果
            if np.dot(self.w, temp)+self.b <= 0:
                result.append(-1)
            else:
                result.append(1)
        return result
