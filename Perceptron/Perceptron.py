import numpy as np


class Perceptron:
    def __init__(self, learning_rate = 1):
        self.w = 0
        self.b = 0
        self.lr = learning_rate
        self.len = 0

    def train(self, x, y):
        self.len = x.data.shape[1]
        self.w = np.zeros(self.len)
        totally_correct = False
        print('进入while循环')
        print(self.len)
        epochs = 0
        while not totally_correct:
            totally_correct = True
            epochs += 1
            print(epochs)
            for i in range(self.len):
                # 从稀疏矩阵中提取一个行向量
                if i % 100 == 99:
                    print('\t', i + 1, self.w[0:2])
                data = x.data.data
                indptr = x.data.indptr
                indices = x.data.indices
                temp = np.zeros(x.data.shape[1])
                for j in range(indptr[i+1]):
                    # print(j)
                    temp[indices[j]] = data[j]
                # print('向量取出完毕')
                # 权重优化
                if y[i] * (np.dot(self.w, temp)+self.b) <= 0:
                    self.w += self.lr * y[i] * temp
                    self.b += self.lr * y[i]
                    totally_correct = False
                # print(self.w)

    def predict(self, x):
        result = []
        for i in range(self.len):
            # 从稀疏矩阵中提取一个行向量
            data = x.data.data
            indptr = x.data.indptr
            indices = x.data.indices
            temp = np.zeros(x.data.shape[1])
            for j in range(indptr[i + 1]):
                temp[indices[j]] = data[j]
            # 求解结果
            if np.dot(self.w, temp)+self.b <= 0:
                result.append(-1)
            else:
                result.append(1)
        return result
