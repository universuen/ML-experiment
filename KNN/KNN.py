import numpy as np


class KNN:
    def __init__(self, k: int = 5):
        self.k = k

    def classify(self, test_vector, data_set, label_set):
        data = data_set.data.data
        indptr = data_set.data.indptr
        indices = data_set.data.indices
        # 因为data_set是稀疏矩阵，所以无法直接得到差值矩阵并求距离。
        # 分别解析出稀疏矩阵的每个行向量并求其与测试向量的距离
        distances = np.zeros(data_set.data.shape[0])
        for i in range(data_set.data.shape[0]):
            if i % 100 == 99:
                print('CALCULATED\t', i + 1)
            # 从稀疏矩阵中提取一个行向量
            temp = np.zeros(data_set.data.shape[1])
            for j in range(indptr[i+1]):
                # print(j)
                temp[indices[j]] = data[j]
            distance = np.linalg.norm(test_vector - temp)
            distances[i] = distance

        # 排序后取前k项确定类别
        sorted_indicies = distances.argsort()
        class_count = {}
        for i in range(self.k):
            vote = label_set[sorted_indicies[i]]
            class_count[vote] = class_count.get(vote, 0) + 1  # default 0

        sorted_class_count = sorted(class_count.items(), key=lambda d: d[1], reverse=True)
        return sorted_class_count[0][0]
