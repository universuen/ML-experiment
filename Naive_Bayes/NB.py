import numpy as np
import math
# T->类别, X->属性


def PDF(x, loc, scale):  # 概率密度函数, 这里使用高斯分布
    # 分子
    numerator = math.exp(-((x-loc)**2)/(2*(scale**2)))
    denominator = math.sqrt(2*math.pi)*scale
    return numerator/denominator

class NB:
    def __init__(self, n_attribute: int = 10000, n_class: int = 20):
        self.Py = np.zeros(n_class)  # 存储P(Y)
        self.Pxy = np.zeros((n_class, n_attribute, 2))  # 存储P(X|Y)的参数值(均值和标准差)

    def train(self, x, y):
        data = x.data.data
        indptr = x.data.indptr
        indices = x.data.indices

        # 首先根据训练集估计P(Y)
        for i in range(self.Py.shape[0]):
            match = 0
            for j in y:
                if j == i:
                    match += 1
            self.Py[i] = match/y.shape[0]
        print(self.Py)

        # 然后计算在每一个类别Y下每一个属性X的均值和标准差
        for i in range(self.Py.shape[0]):
            container = []  # 暂时存放属于当前类别Y的所有样本
            for j in range(y.shape[0]):
                if y[j] == i:
                    # 从稀疏矩阵中提取与之匹配的行向量
                    temp = np.zeros(x.data.shape[1])
                    for k in range(indptr[j + 1]):
                        temp[indices[k]] = data[k]
                    container.append(temp)
            # 遍历所有属性
            for j in range(x.data.shape[1]):
                # 计算均值
                sum = 0
                for k in container:
                    sum += k[j]
                avg = sum/len(container)
                self.Pxy[i][j][0] = avg
                # 计算标准差
                sum = 0
                for k in container:
                    sum += (k - avg)**2
                self.Pxy[i][j][1] = math.sqrt(sum/len(container))
            print('TRAIN\t', i)

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
            Pyx = np.zeros(self.Py.shape[0])  # 用于记录所有P(Y|X)
            for j in range(self.Py.shape[0]):
                temp_result = self.Py[j]
                for k in range(self.Pxy.shape[1]):
                    temp_result *= PDF(temp[k], self.Pxy[j][k][0], self.Pxy[j][k][1])
                Pyx[j] = temp_result
            # 取概率最大的类
            result.append(np.argsort(Pyx)[0])
        return result
