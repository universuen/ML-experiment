import numpy as np
import threading
# Y->类别, X->属性


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
        # 然后计算在每一个类别Y下每一个属性X的均值和标准差
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
        for j in range(x.shape[0]):
            if y[j] == i:
                # 从稀疏矩阵中提取属于当前类的行向量
                temp = np.zeros(x.shape[1])
                for k in range(indptr[j], indptr[j+1]):
                    temp[indices[k]] = data[k]
                container.append(temp)
        # 将container转置，方便接下来的运算
        container = np.array(container).T
        # 遍历所有属性
        for j in range(x.shape[1]):
            # 计算均值
            self.Pxy[i][j][0] = np.mean(container[j])
            # 计算标准差
            self.Pxy[i][j][1] = np.std(container[j])
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
        for j in range(indptr[i], indptr[i+1]):
            temp[indices[j]] = data[j]
        Pyx = np.zeros((self.Py.shape[0]))  # 用于记录所有P(Y|X)
        for j in range(self.Py.shape[0]):
            Pyx[j] = np.log(self.Py[j])
            # 大量的乘积有可能造成浮点数下溢出，所以这里对结果取对数，变为求和形式
            for k in range(self.Pxy.shape[1]):
                miu = self.Pxy[j][k][0]
                sigma = self.Pxy[j][k][1]
                if miu*sigma == 0:
                    continue
                Pyx[j] += (-((temp[k] - miu)**2)/(2*(sigma**2)) - np.log(sigma)) # 常数已省略
        # 取概率最大的类
        result[i] = np.argmax(Pyx)
        print('NB predict: 线程' + str(i) + '运行结束', np.argmax(Pyx))

