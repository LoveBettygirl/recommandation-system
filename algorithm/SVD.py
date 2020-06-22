import sys
import math
import random

class SVD(object):
    def __init__(self, n_factors=10, n_epochs=30, learn_rate=0.005, lambda_=0.02, scale=(0,100)):
        super().__init__()
        self.n_user = 0
        self.n_item = 0
        self.train_mean = 0.0
        self.learn_rate = learn_rate
        self.lambda_ = lambda_
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.scale = scale
        self.user_dict = {}
        self.item_dict = {}
        self.sprase_matrix = []
        self.bu = None
        self.bi = None
        self.p = None
        self.q = None

    def stat(self):
        train_file = []
        test_file = []
        print("Reading train and test file......")
        with open('../data-new/train.txt', 'r') as f1:
            train_file = f1.readlines()
            f1.close()
        with open('../data-new/test.txt', 'r') as f2:
            test_file = f2.readlines()
            f2.close()

        train_records = 0
        test_records = 0
        user_rates = {} # format: { userid1: [sco1, sco2, ...], ...}
        item_rates = {} # format: { itemid1: [sco1, sco2, ...], ...}
        rate_num = [0 for _ in range(0,101)]
        user_id = ''
        item_id = ''
        rate_str = ''
        uid = 0
        iid = 0
        rate_ = 0.0
        global_mean = 0.0
        min_user_id = sys.maxsize
        max_user_id = 0
        min_item_id = sys.maxsize
        max_item_id = 0

        print("Processing train file......")
        for train in train_file:
            line = train.strip()
            if line.find('|') != -1:
                user_id, user_item_count = line.split('|')
                uid = int(user_id)
                try:
                    u = self.user_dict[uid]
                except KeyError:
                    user_rates[uid] = []
                    self.user_dict[uid] = self.n_user
                    if max(uid, max_user_id) == uid:
                        max_user_id = uid
                    if min(uid, min_user_id) == uid:
                        min_user_id = uid
                    self.n_user += 1
            else:
                if line == "":
                    continue
                item_id, rate_str = line.split()
                train_records += 1
                iid = int(item_id)
                rate_ = float(rate_str)
                rate_num[math.floor(rate_)] += 1
                global_mean += rate_
                try:
                    item_rates[iid].append(rate_)
                except KeyError:
                    item_rates[iid] = []
                    item_rates[iid].append(rate_)
                    user_rates[uid].append(rate_)
                    self.item_dict[iid] = self.n_item
                    if max(iid, max_item_id) == iid:
                        max_item_id = iid
                    if min(iid, min_item_id) == iid:
                        min_item_id = iid
                    self.n_item += 1

        global_mean /= train_records

        print("Processing test file......")
        for test in test_file:
            line = test.strip()
            if line.find('|') != -1:
                user_id, user_item_count = line.split('|')
                uid = int(user_id)
                try:
                    u = self.user_dict[uid]
                except KeyError:
                    user_rates[uid] = []
                    self.user_dict[uid] = self.n_user
                    if max(uid, max_user_id) == uid:
                        max_user_id = uid
                    if min(uid, min_user_id) == uid:
                        min_user_id = uid
                    self.n_user += 1
            else:
                if line == "":
                    continue
                test_records += 1
                iid = int(line)
                try:
                    i = self.item_dict[iid]
                except KeyError:
                    self.item_dict[iid] = self.n_item
                    if max(iid, max_item_id) == iid:
                        max_item_id = iid
                    if min(iid, min_item_id) == iid:
                        min_item_id = iid
                    self.n_item += 1

        print("Making statistics files......")
        with open('stat.txt', 'w') as statf:
            statf.write('User count: %d\n' % self.n_user)
            statf.write('Item count: %d\n' % self.n_item)
            statf.write('Min user id: %d\n' % min_user_id)
            statf.write('Max user id: %d\n' % max_user_id)
            statf.write('Min item id: %d\n' % min_item_id)
            statf.write('Max item id: %d\n' % max_item_id)
            statf.write('Record counts of train file: %d\n' % train_records)
            statf.write('Record counts of test file: %d\n' % test_records)
            statf.write('The mean of all records in train file: %f\n' % global_mean)
            statf.close()

        with open('userMean.csv', 'w') as userMean:
            s = sorted(item_rates.items(), key = lambda x : x[0])
            for u, r in s:
                userMean.write(str(u))
                userMean.write(',')
                sum = 0.0
                for rate in r:
                    sum += rate
                if len(r) != 0:
                    sum /= len(r)
                userMean.write(str(sum))
                userMean.write('\n')
            userMean.close()

        with open('itemMean.csv', 'w') as itemMean:
            s = sorted(item_rates.items(), key = lambda x : x[0])
            for i, r in s:
                itemMean.write(str(i))
                itemMean.write(',')
                sum = 0.0
                for rate in r:
                    sum += rate
                if len(r) != 0:
                    sum /= len(r)
                itemMean.write(str(sum))
                itemMean.write('\n')
            itemMean.close()

        with open('rateDetail.csv', 'w') as rateDetail:
            for i, j in enumerate(rate_num):
                rateDetail.write('%d,%d\n' % (i, j))
            rateDetail.close()

    def divide_train_and_test_set(self, test=2):
        assert test >= 0 and test <= 10
        print('Dividing train and test set......')
        train_file = []
        with open('../data-new/train.txt', 'r') as f:
            train_file = f.readlines()
            f.close()
        
        user_id = ''
        item_id = ''
        rate_str = ''
        trainset = open('trainset.csv', 'w')
        testset = open('testset.csv', 'w')
        rank = test / 10

        for train in train_file:
            line = train.strip()
            if line.find('|') != -1:
                user_id, user_item_count = line.split('|')
            else:
                if line == "":
                    continue
                item_id, rate_str = line.split()
                if (random.random() >= rank):
                    trainset.write('%s,%s,%s\n' % (user_id, item_id, rate_str))
                else:
                    testset.write('%s,%s,%s\n' % (user_id, item_id, rate_str))

        trainset.close()
        testset.close()

    def _prepare(self):
        train = []
        with open('trainset.csv', 'r') as f:
            train = f.readlines()
            f.close()

        sqrt_dim = math.sqrt(self.n_factors)

        self.bu = [0.0 for _ in range(0, self.n_user)]
        self.bi = [0.0 for _ in range(0, self.n_item)]
        self.p = [[random.random() / sqrt_dim for _ in range(0, self.n_factors)] for _ in range(0, self.n_user)]
        self.q = [[random.random() / sqrt_dim for _ in range(0, self.n_factors)] for _ in range(0, self.n_item)]

        self.records = 0

        for line in train:
            if line == "":
                continue
            user_id, item_id, rate_str = line.split(',')
            uid = self.user_dict[int(user_id)]
            iid = self.item_dict[int(item_id)]
            rate_ = float(rate_str)
            self.sprase_matrix.append((uid, iid, rate_))
            self.train_mean += rate_
            self.records += 1
        
        self.train_mean /= self.records

    def _dot(self, u, i):
        sum = 0.0
        for k in range(0, self.n_factors):
            sum += (self.p[u][k] * self.q[i][k])
        return sum

    def train(self):
        print('Preparing......')
        self._prepare()
        print('Training......')
        
        for epoch in range(0, self.n_epochs):
            rmse = 0.0
            for u, i, r in self.sprase_matrix:
                rp = self.train_mean + self.bu[u] + self.bi[i] + self._dot(u, i)
                # 计算完分数之后一定要作边界检查
                if (rp < self.scale[0]):
                    rp = float(self.scale[0])
                if (rp > self.scale[1]):
                    rp = float(self.scale[1])
                err = r - rp
                self.bu[u] += self.learn_rate * (err - self.lambda_ * self.bu[u])
                self.bi[i] += self.learn_rate * (err - self.lambda_ * self.bi[i])
                for k in range(0, self.n_factors):
                    # trick 1: 每次迭代计算矩阵之前需要保存原值，否则越迭代值越大，最终导致超过浮点数范围
                    qik = self.q[i][k]
                    puk = self.p[u][k]
                    self.p[u][k] += self.learn_rate * (err * qik - self.lambda_ * puk)
                    self.q[i][k] += self.learn_rate * (err * puk - self.lambda_ * qik)
                rmse += err ** 2
            rmse /= self.records
            rmse = math.sqrt(rmse)
            # trick 2: 每次迭代之后需要降低学习率，以让结果尽快收敛
            self.learn_rate *= 0.5
            print('RMSE in epoch %d: %f' % (epoch, rmse))

    def test_model(self):
        print('Model testing......')
        records = 0
        rmse = 0.0
        resultf = open('model_test_result.csv', 'w')
        with open('testset.csv', 'r') as f:
            for line in f.readlines():
                if line == "":
                    continue
                user_id, item_id, rate_str = line.split(',')
                u = self.user_dict[int(user_id)]
                i = self.item_dict[int(item_id)]
                r = float(rate_str)
                rp = self.train_mean + self.bu[u] + self.bi[i] + self._dot(u, i)
                if (rp < self.scale[0]):
                    rp = float(self.scale[0])
                if (rp > self.scale[1]):
                    rp = float(self.scale[1])
                resultf.write('%d,%d,%f,%f\n' % (int(user_id), int(item_id), rp, r))
                err = r - rp
                rmse += err ** 2
                records += 1
            rmse /= records
            rmse = math.sqrt(rmse)
            f.close()
        print('RMSE in test set: %f' % rmse)
        resultf.close()

    def predict(self):
        print('Predicting......')
        test_file = []
        with open('../data-new/test.txt', 'r') as f:
            test_file = f.readlines()
            f.close()
        user_id = ''
        item_id = ''
        uid = 0
        iid = 0
        resultf = open('result.txt', 'w')
        for test in test_file:
            line = test.strip()
            if line.find('|') != -1:
                user_id, user_item_count = line.split('|')
                uid = int(user_id)
                resultf.write(line + '\n')
            else:
                if line == "":
                    continue
                iid = int(line)
                u = self.user_dict[uid]
                i = self.item_dict[iid]
                rp = self.train_mean + self.bu[u] + self.bi[i] + self._dot(u, i)
                if (rp < self.scale[0]):
                    rp = float(self.scale[0])
                if (rp > self.scale[1]):
                    rp = float(self.scale[1])
                resultf.write('%d  %f  \n' % (iid, rp))
                
        resultf.close()

if __name__ == "__main__":
    svd = SVD()
    svd.stat()
    #svd.divide_train_and_test_set()
    svd.train()
    svd.test_model()
    svd.predict()
    