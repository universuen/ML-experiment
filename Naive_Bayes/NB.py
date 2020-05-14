import numpy as np
import math
import threading
from sklearn.naive_bayes import MultinomialNB
# T->类别, X->属性


def PDF(x, loc, scale):  # 概率密度函数, 这里使用高斯分布
    # 引入esp防止分母为0
    esp = 1e-6
    # 分子
    numerator = math.exp(-((x-loc)**2)/(2*(scale**2) + esp))
    # 分母
    denominator = math.sqrt(2*math.pi)*scale + esp
    return numerator/denominator


class NB:
    def __init__(self, n_attribute: int = 10000, n_class: int = 20):
        self.Py = np.zeros(n_class)  # 存储P(Y)
        self.Pxy = np.zeros((n_class, n_attribute, 2))  # 存储后验概率P(X|Y)的均值和方差

    def train(self, x, y):
        data = x.data
        indptr = x.indptr
        indices = x.indices

        # 首先根据训练集估计P(Y)
        for i in range(self.Py.shape[0]):
            match = 0
            for j in y:
                if j == i:
                    match += 1
            self.Py[i] = match/y.shape[0]
        print(self.Py)
        threads = []
        # 然后计算在每一个类别Y下每一个属性X的后验概率
        for i in range(self.Py.shape[0]):
            t = threading.Thread(target=self.sub_train, args=(y, i, x, indptr, indices, data), name=str(i))
            threads.append(t)
        for t in threads:
            t.start()
            print('NB train: 线程'+ t.name + '开始运行')
        for t in threads:
            t.join()


    def sub_train(self, y, i, x, indptr, indices, data):
        container = []  # 暂时存放属于当前类别Y的所有样本
        for j in range(y.shape[0]):
            if y[j] == i:
                # 从稀疏矩阵中提取属于当前类的行向量
                temp = np.zeros(x.shape[1])
                for k in range(indptr[j + 1]):
                    temp[indices[k]] = data[k]
                container.append(temp)
        # 遍历所有属性
        for j in range(x.shape[1]):
            # 计算均值
            sum = 0
            for k in container:
                sum += k[j]
            avg = sum / len(container)
            self.Pxy[i][j][0] = avg
            # 计算标准差
            sum = 0
            for k in container:
                sum += (k[j] - avg) ** 2
            self.Pxy[i][j][1] = math.sqrt(sum / len(container))
        print('NB train: 线程' + str(i) + '运行结束')





    def predict(self, x):
        result = [None for _ in range(x.shape[0])]
        data = x.data
        indptr = x.indptr
        indices = x.indices
        threads = []
        for i in range(x.shape[0]):
            t = threading.Thread(target=self.sub_predict, args=(x, i, result, data, indptr, indices), name=str(i))
            threads.append(t)
        for t in threads:
            t.start()
            print('NB predict: 线程'+ t.name + '开始运行')
        for t in threads:
            t.join()
        return result

    def sub_predict(self, x, i, result, data, indptr, indices):
        # 从稀疏矩阵中提取一个行向量
        temp = np.zeros(x.shape[1])
        for j in range(indptr[i + 1]):
            temp[indices[j]] = data[j]
        Pyx = np.zeros(self.Py.shape[0])  # 用于记录所有P(Y|X)
        for j in range(self.Py.shape[0]):
            Pyx[j] = math.log2(self.Py[j] + 1)
            # 大量的乘积有可能造成浮点数下溢出，所以这里对结果取对数，变为求和形式
            for k in range(self.Pxy.shape[1]):
                Pyx[j] += math.log2(PDF(temp[k], self.Pxy[j][k][0], self.Pxy[j][k][1]) + 1)
        # 取概率最大的类
        print(np.argsort(Pyx))
        result[i] = np.argsort(Pyx)[len(Pyx) - 1]
        print('NB predict: 线程' + str(i) + '运行结束')

