from Perceptron import Perceptron
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from KNN import KNN
from Naive_Bayes import NB
import pickle
import threading


# 数据获取
train_set = fetch_20newsgroups(subset='train')
test_set = fetch_20newsgroups(subset='test')

# 导入stop list
with open('stopwords.txt') as f:
    stop_list = f.read().split()

# 实例化转换器
vectorizer = TfidfVectorizer(stop_words=stop_list, lowercase=True, max_features=10000)

# 将文本规格化并转换为tf-idf向量
train_x = vectorizer.fit_transform(train_set.data)
train_y = train_set.target
test_x = vectorizer.fit_transform(test_set.data)
test_y = test_set.target


def Perceptron_test():
    accuracy = np.zeros(20)
    # 实例化感知机
    perceptron = Perceptron.Perceptron()
    # 模型队列
    model = range(20)
    # 线程队列
    threads = []

    # 针对每一个类别进行二元分类并计算准确率
    for i in range(20):
        temp_train_y = train_set.target
        temp_test_y = test_set.target
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
        # print('开始训练')
        thread = threading.Thread(target=sub_Perceptron_test, args=(model, accuracy, i, perceptron, train_x, temp_train_y, temp_test_y), name=str(i))
        threads.append(thread)
    for t in threads:
        t.start()
        print('线程'+ t.name + '开始运行')

    # 所有线程完成后存储模型
    for t in threads:
        t.join()
    pickle.dump(model, open('Perceptron\\models.pkl', 'wb'))
    print(accuracy)

def sub_Perceptron_test(model, accuracy, i, perceptron, x, temp_train_y, temp_test_y):
    perceptron.train(x, temp_train_y)
    result = perceptron.predict(test_x)
    correct = 0
    for j, k in zip(result, temp_test_y):
        correct += (j == k)
    print(correct / len(result))
    accuracy[i] = correct / len(result)
    model[i] = perceptron
    print('线程' + str(i) + '运行结束!')


def KNN_test():
    # 实例化KNN类, 设置K值为5
    knn = KNN.KNN(5)
    prediction = knn.predict(test_x, train_x, train_y, 20)
    # 计算前20个向量的准确率
    correct = 0
    for j, k in zip(prediction, test_y[:20]):
        correct += (j == k)
    print(correct / 20)


def NB_test():
    # 实例化模型
    nb = NB.NB(train_x.shape[1], 20)
    # 训练模型
    nb.train(train_x, train_y)
    # 存储模型
    pickle.dump(nb, open('Naive_Bayes\\model.pkl', 'wb'))
    # with open('Naive_Bayes\\model.pkl', 'rb') as f:
    #     nb = pickle.load(f)
    # 计算准确率
    correct = 0
    prediction = nb.predict(test_x)
    for j, k in zip(prediction, test_y):
        correct += (j == k)
    print(correct / test_y.shape[0])

Perceptron_test()
# print(np.load('Perceptron_accuracy.npy'))
# KNN_test()
# NB_test()


