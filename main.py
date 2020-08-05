from Perceptron import Perceptron
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from Naive_Bayes import NB
import pickle
import threading
from sklearn.metrics import f1_score, accuracy_score

# This is a test line.

# 数据获取
train_set = fetch_20newsgroups(subset='train')
test_set = fetch_20newsgroups(subset='test')

# 导入stop list
with open('stopwords.txt') as f:
    stop_list = f.read().split()

# 实例化转换器
vectorizer = TfidfVectorizer(stop_words=stop_list, max_features=15000)

# 将文本规格化并转换为tf-idf向量
train_x = vectorizer.fit_transform(train_set.data)
train_y = train_set.target
test_x = vectorizer.transform(test_set.data)
test_y = test_set.target


P_lr_dic = []
P_ep_dic = []

def Perceptron_test(lr, epochs):
    accuracy = np.zeros(20)
    f1 = np.zeros(20)
    # 模型队列
    perceptrons = [None for _ in range(20)]
    # 线程队列
    threads = []
    temp_train_y = [None for _ in range(20)]
    temp_test_y = [None for _ in range(20)]
    # 针对每一个类别进行二元分类并计算准确率
    for i in range(20):
        perceptrons[i] = Perceptron.Perceptron(learning_rate=lr, epochs=epochs)
        temp_train_y[i] = train_y.copy()
        temp_test_y[i] = test_y.copy()
        # 规格化Target
        for j in range(temp_train_y[i].shape[0]):
            if temp_train_y[i][j] == i:
                temp_train_y[i][j] = 1
            else:
                temp_train_y[i][j] = -1
        for j in range(temp_test_y[i].shape[0]):
            if temp_test_y[i][j] == i:
                temp_test_y[i][j] = 1
            else:
                temp_test_y[i][j] = -1
        thread = threading.Thread(target=sub_Perceptron_test, args=(accuracy, f1, i, perceptrons[i], train_x, temp_train_y[i], temp_test_y[i]), name=str(i))
        threads.append(thread)
    for t in threads:
        t.start()
        # print('Perceptron: 线程'+ t.name + '开始运行')

    # 所有线程完成后存储模型和准确率
    for t in threads:
        t.join()
    # pickle.dump(model, open('Perceptron\\models.pkl', 'wb'))
    accuracy = np.mean(accuracy)
    f1 = np.mean(f1)
    P_lr_dic.append([lr, accuracy, f1])
    # pickle.dump(accuracy, open('Perceptron\\accuracy.pkl', 'wb'))

def sub_Perceptron_test(accuracy, f1, i, perceptron, x, temp_train_y, temp_test_y):
    perceptron.train(x, temp_train_y)
    result = perceptron.predict(test_x)
    accuracy[i] = accuracy_score(temp_test_y, result)
    f1[i] = f1_score(temp_test_y, result)
    # print('Perceptron: 线程' + str(i) + '运行结束!')




def NB_test():
    # 实例化模型
    nb = NB.NB(train_x.shape[1], 20)
    # 训练模型
    nb.train(train_x, train_y)
    # 存储模型
    # pickle.dump(nb, open('Naive_Bayes\\model_multimodial.pkl', 'wb'))
    # with open('Naive_Bayes\\model_Gaussian.pkl', 'rb') as f:
    #     nb = pickle.load(f)
    # 计算准确率
    correct = 0
    prediction = nb.predict(test_x)
    for j, k in zip(prediction, test_y):
        correct += (j == k)
    print(correct / test_y.shape[0])


if __name__ == '__main__':
    threads1 = []
    # for epochs in range(1, 11):
    for lr in np.arange(0.1, 1.6, 0.1):
        epochs = 1
        t1 = threading.Thread(target=Perceptron_test, args=(lr, epochs), name = str(epochs))
        threads1.append(t1)
    for t in threads1:
        t.start()
    for t in threads1:
        t.join()
        print('P:' + str(t.name) + 'finished')
    print('P FINISHED')
    print(P_lr_dic)
    pickle.dump(P_lr_dic, open('P_lr_dic', 'wb'))






