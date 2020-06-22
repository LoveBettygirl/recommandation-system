import os
import sys
import math
import random
import struct
from collections import defaultdict
from scipy import optimize
import numpy as np
import timeit

class SVD_improved(object):
    def __init__(self, n_factors=10, n_epochs=20, learn_rate=0.005, _lambda=0.02, scale=(0,100)):
        super().__init__()
        self._TRAIN_FILE = '../data-new/train.txt'
        self._TEST_FILE = '../data-new/test.txt'
        self._ATTR_FILE = '../data-new/itemAttribute.txt'
        self._TRAIN_SET = 'trainset.csv'
        self._TEST_SET = 'testset.csv'
        self._BU_VECTOR = 'bu_vector.dat'
        self._BI_VECTOR = 'bi_vector.dat'
        self._P_MATRIX = 'p_matrix.dat'
        self._Q_MATRIX = 'q_matrix.dat'
        self._SPRASE = 'sprase.dat'
        self._all_time = 0.0
        self._do_divide = False # 是否要重新划分训练集和测试集
        self._do_train = False # 是否要重新训练
        self._n_user = 0
        self._n_item = 0
        self._train_mean = 0.0
        self._learn_rate = learn_rate
        self._lambda = _lambda
        self._n_factors = n_factors
        self._n_epochs = n_epochs
        self._scale = scale
        self._user_dict = {}
        self._item_dict = {}
        self._sprase_matrix = []
        self._sprase_matrix_with_err = []
        self._bu = None
        self._bi = None
        self._p = None
        self._q = None
        self._item_attrs = {}
        self._user_param = {}
        self._user_item_attrs = defaultdict(list) # format: { user1: [(item1_id, item1.err, item1.attr1, item1.attr2), ...], ... }

    @property
    def all_time(self):
        return self._all_time
    
    @property
    def do_train(self):
        return self._do_train

    @do_train.setter
    def do_train(self, dotrain):
        assert dotrain == True or dotrain == False
        self._do_train = dotrain

    @property
    def do_divide(self):
        return self._do_divide

    @do_divide.setter
    def do_divide(self, dodivide):
        assert dodivide == True or dodivide == False
        self._do_divide = dodivide
    
    @property
    def train_file(self):
        return self._TRAIN_FILE

    @train_file.setter
    def train_file(self, filename):
        assert filename != None
        assert isinstance(filename, str)
        self._TRAIN_FILE = filename

    @property
    def test_file(self):
        return self._TEST_FILE

    @test_file.setter
    def test_file(self, filename):
        assert filename != None
        assert isinstance(filename, str)
        self._TEST_FILE = filename

    @property
    def attr_file(self):
        return self._TEST_FILE

    @attr_file.setter
    def attr_file(self, filename):
        assert filename != None
        assert isinstance(filename, str)
        self._ATTR_FILE = filename
    
    @property
    def lambda_(self):
        return self._lambda

    @lambda_.setter
    def lambda_(self, l):
        assert isinstance(l, float) and l > 0.0
        self._lambda = l
    
    @property
    def learn_rate(self):
        return self._learn_rate

    @learn_rate.setter
    def learn_rate(self, lr):
        assert isinstance(lr, float) and lr > 0.0
        self._learn_rate = lr
    
    @property
    def epochs(self):
        return self._n_epochs

    @epochs.setter
    def epochs(self, n_epochs):
        assert isinstance(n_epochs, int) and n_epochs > 0
        self._n_epochs = n_epochs
    
    @property
    def factors(self):
        return self._n_factors

    @factors.setter
    def factors(self, n_factors):
        assert isinstance(n_factors, int) and n_factors > 0
        self._n_factors = n_factors

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, s):
        assert isinstance(s, tuple) and len(s) == 2
        assert isinstance(s[0], int) and isinstance(s[1], int)
        assert s[0] >= 0 and s[1] >= 0 and s[0] < s[1]
        self._scale = s
    
    def stat(self):
        self._all_time = 0.0
        print('In %s ......' % (self.__class__.__name__ + '.' + sys._getframe().f_code.co_name + '()'))
        start = timeit.default_timer()
        attr_lines = []
        train_lines = []
        test_lines = []
        with open(self._ATTR_FILE, 'r') as f:
            attr_lines = f.readlines()
            f.close()
        with open(self._TRAIN_FILE, 'r') as f:
            train_lines = f.readlines()
            f.close()
        with open(self._TEST_FILE, 'r') as f:
            test_lines = f.readlines()
            f.close()

        train_records = 0
        test_records = 0
        attr_records = 0
        user_rates = {} # format: { userid1: [rate1, rate2, ...], ... }
        item_rates = {} # format: { itemid1: [rate1, rate2, ...], ... }
        rate_num = [0 for _ in range(0, self._scale[1] - self._scale[0] + 1)]
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
        item_has_attr = {}
        attr_lines_ = []

        print("Processing item attr file......")
        # 将属性值None先处理为0（原数据中没有0的属性值）
        for attr_line in attr_lines:
            line = attr_line.strip()
            if line == "":
                continue
            item_id, attr1, attr2 = line.split('|')
            if attr1 == 'None':
                attr1 = 0
            else:
                attr1 = int(attr1)
            if attr2 == 'None':
                attr2 = 0
            else:
                attr2 = int(attr2)
            iid = int(item_id)
            if max(iid, max_item_id) == iid:
                max_item_id = iid
            if min(iid, min_item_id) == iid:
                min_item_id = iid
            self._item_dict[iid] = self._n_item
            item_has_attr[iid] = (iid, attr1, attr2)
            attr_lines_.append((iid, attr1, attr2))
            self._n_item += 1
            attr_records += 1

        # 统计所有属性的平均值
        attr1_sum = 0.0
        attr2_sum = 0.0
        for item_id, attr1, attr2 in attr_lines_:
            attr1_sum += attr1
            attr2_sum += attr2

        attr1_sum /= len(attr_lines_)
        attr2_sum /= len(attr_lines_)

        # 将缺失属性信息的item使用平均值处理
        attr_list = []
        item_no_attrs = 0
        for i in range(min_item_id, max_item_id + 1):
            try:
                attr_list.append([item_has_attr[i][1], item_has_attr[i][2]])
            except KeyError:
                attr_list.append([attr1_sum, attr2_sum])
                item_no_attrs += 1

        print("Processing train file......")
        for train in train_lines:
            line = train.strip()
            if line.find('|') != -1:
                user_id, user_item_count = line.split('|')
                uid = int(user_id)
                try:
                    u = self._user_dict[uid]
                except KeyError:
                    user_rates[uid] = []
                    self._user_dict[uid] = self._n_user
                    if max(uid, max_user_id) == uid:
                        max_user_id = uid
                    if min(uid, min_user_id) == uid:
                        min_user_id = uid
                    self._n_user += 1
            else:
                if line == "":
                    continue
                item_id, rate_str = line.split()
                train_records += 1
                iid = int(item_id)
                rate_ = float(rate_str)
                rate_num[math.floor(rate_)] += 1
                global_mean += rate_
                user_rates[uid].append(rate_)
                try:
                    i = self._item_dict[iid]
                except KeyError:
                    self._item_dict[iid] = self._n_item
                    if max(iid, max_item_id) == iid:
                        max_item_id = iid
                    if min(iid, min_item_id) == iid:
                        min_item_id = iid
                    self._n_item += 1
                try:
                    item_rates[iid].append(rate_)
                except KeyError:
                    item_rates[iid] = []
                    item_rates[iid].append(rate_)


        global_mean /= train_records

        print("Processing test file......")
        for test in test_lines:
            line = test.strip()
            if line.find('|') != -1:
                user_id, user_item_count = line.split('|')
                uid = int(user_id)
                try:
                    u = self._user_dict[uid]
                except KeyError:
                    user_rates[uid] = []
                    self._user_dict[uid] = self._n_user
                    if max(uid, max_user_id) == uid:
                        max_user_id = uid
                    if min(uid, min_user_id) == uid:
                        min_user_id = uid
                    self._n_user += 1
            else:
                if line == "":
                    continue
                test_records += 1
                iid = int(line)
                try:
                    i = self._item_dict[iid]
                except KeyError:
                    self._item_dict[iid] = self._n_item
                    if max(iid, max_item_id) == iid:
                        max_item_id = iid
                    if min(iid, min_item_id) == iid:
                        min_item_id = iid
                    self._n_item += 1

        print("Generating new item attr file......")
        with open('itemAttribute.csv', 'w') as f:
            for i in range(len(attr_list)):
                f.write('%d,%d,%d\n' % (i, attr_list[i][0], attr_list[i][1]))
            f.close()

        print("Making statistics files......")
        with open('stat.txt', 'w') as statf:
            statf.write('User count: %d\n' % self._n_user)
            statf.write('Item appeared count: %d\n' % self._n_item)
            statf.write('Min user id: %d\n' % min_user_id)
            statf.write('Max user id: %d\n' % max_user_id)
            statf.write('Min item id: %d\n' % min_item_id)
            statf.write('Max item id: %d\n' % max_item_id)
            statf.write('Item without attr count: %d\n' % item_no_attrs)
            statf.write('Record counts of train file: %d\n' % train_records)
            statf.write('Record counts of test file: %d\n' % test_records)
            statf.write('Record counts of item attr file: %d\n' % attr_records)
            statf.write('The mean of all records in train file: %f\n' % global_mean)
            statf.write('attr1 mean: %f\n' % attr1_sum)
            statf.write('attr2 mean: %f\n' % attr2_sum)
            statf.close()

        with open('userMean.csv', 'w') as userMean:
            s = sorted(user_rates.items(), key = lambda x : x[0])
            for u, r in s:
                userMean.write(str(u))
                userMean.write(',')
                sum = 0.0
                for rate in r:
                    sum += rate
                if len(r) != 0:
                    sum /= len(r)
                userMean.write(str(sum))
                userMean.write(',')
                userMean.write(str(len(r)))
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
                itemMean.write(',')
                itemMean.write(str(len(r)))
                itemMean.write('\n')
            itemMean.close()

        with open('rateDetail.csv', 'w') as rateDetail:
            for i, j in enumerate(rate_num):
                rateDetail.write('%d,%d\n' % (i + self._scale[0], j))
            rateDetail.close()

        end = timeit.default_timer()
        self._all_time += (end - start)
        print('Time cost in %s: %fs' % (self.__class__.__name__ + '.' + sys._getframe().f_code.co_name + '()', end - start))

    def divide_train_and_test_set(self, test=2):
        assert test >= 0 and test <= 10
        print('Dividing train and test set......')
        start = timeit.default_timer()
        train_file = []
        with open(self._TRAIN_FILE, 'r') as f:
            train_file = f.readlines()
            f.close()
        
        user_id = ''
        item_id = ''
        rate_str = ''
        trainset = open(self._TRAIN_SET, 'w')
        testset = open(self._TEST_SET, 'w')
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

        end = timeit.default_timer()
        self._all_time += (end - start)
        print('Time cost: %fs' % (end - start))

    def _prepare(self):
        print('In %s ......' % (self.__class__.__name__ + '.' + sys._getframe().f_code.co_name + '()'))
        start = timeit.default_timer()
        train = []
        with open(self._TRAIN_SET, 'r') as f:
            train = f.readlines()
            f.close()

        sqrt_dim = math.sqrt(self._n_factors)

        self._bu = [0.0 for _ in range(0, self._n_user)]
        self._bi = [0.0 for _ in range(0, self._n_item)]
        self._p = [[random.random() / sqrt_dim for _ in range(0, self._n_factors)] for _ in range(0, self._n_user)]
        self._q = [[random.random() / sqrt_dim for _ in range(0, self._n_factors)] for _ in range(0, self._n_item)]

        self.records = 0

        for train_line in train:
            line = train_line.strip()
            if line == "":
                continue
            user_id, item_id, rate_str = line.split(',')
            uid = self._user_dict[int(user_id)]
            iid = self._item_dict[int(item_id)]
            rate_ = float(rate_str)
            self._sprase_matrix.append((uid, iid, rate_))
            self._train_mean += rate_
            self.records += 1
        
        self._train_mean /= self.records

        end = timeit.default_timer()
        self._all_time += (end - start)
        print('Time cost in %s: %fs' % (self.__class__.__name__ + '.' + sys._getframe().f_code.co_name + '()', end - start))

    def _dot(self, u, i):
        sum = 0.0
        for k in range(0, self._n_factors):
            sum += (self._p[u][k] * self._q[i][k])
        return sum

    def train(self):
        print('Preparing......')
        self._prepare()
        print('Training......')
        
        for epoch in range(0, self._n_epochs):
            rmse = 0.0
            start = timeit.default_timer()
            self._sprase_matrix_with_err = []
            for u, i, r in self._sprase_matrix:
                rp = self._train_mean + self._bu[u] + self._bi[i] + self._dot(u, i)
                # 计算完分数之后一定要作边界检查
                if (rp < self._scale[0]):
                    rp = float(self._scale[0])
                if (rp > self._scale[1]):
                    rp = float(self._scale[1])
                err = r - rp
                self._bu[u] += self._learn_rate * (err - self._lambda * self._bu[u])
                self._bi[i] += self._learn_rate * (err - self._lambda * self._bi[i])
                for k in range(0, self._n_factors):
                    # trick 1: 每次迭代计算矩阵之前需要保存原值，否则越迭代值越大，最终导致超过浮点数范围
                    qik = self._q[i][k]
                    puk = self._p[u][k]
                    self._p[u][k] += self._learn_rate * (err * qik - self._lambda * puk)
                    self._q[i][k] += self._learn_rate * (err * puk - self._lambda * qik)
                rmse += err ** 2
                self._sprase_matrix_with_err.append((u, i, r, err))
            rmse /= self.records
            rmse = math.sqrt(rmse)
            # trick 2: 每次迭代之后需要降低学习率，以让结果尽快收敛
            self._learn_rate *= 0.5
            end = timeit.default_timer()
            print('RMSE in epoch %d: %f' % (epoch, rmse))
            self._all_time += (end - start)
            print('Time cost: %fs' % (end - start))

        print('Iteration finished')
        print('Writing train result data to files......')
        start = timeit.default_timer()
        with open(self._BU_VECTOR, 'wb') as f:
            f.write(struct.pack('i', self._n_user))
            for bu in self._bu:
                f.write(struct.pack('d', bu))
            f.close()
        with open(self._BI_VECTOR, 'wb') as f:
            f.write(struct.pack('i', self._n_item))
            for bi in self._bi:
                f.write(struct.pack('d', bi))
            f.close()
        with open(self._P_MATRIX, 'wb') as f:
            f.write(struct.pack('i', self._n_user))
            f.write(struct.pack('i', self._n_factors))
            for p in self._p:
                for pi in p:
                    f.write(struct.pack('d', pi))
            f.close()
        with open(self._Q_MATRIX, 'wb') as f:
            f.write(struct.pack('i', self._n_item))
            f.write(struct.pack('i', self._n_factors))
            for q in self._q:
                for qi in q:
                    f.write(struct.pack('d', qi))
            f.close()
        with open(self._SPRASE, 'wb') as f:
            f.write(struct.pack('d', self._train_mean))
            f.write(struct.pack('i', self.records))
            for u, i, r, err in self._sprase_matrix_with_err:
                f.write(struct.pack('i', u))
                f.write(struct.pack('i', i))
                f.write(struct.pack('d', r))
                f.write(struct.pack('d', err))
            f.close()
        end = timeit.default_timer()
        self._all_time += (end - start)
        print('Time cost: %fs' % (end - start))
        

    def _prepare_item_attr(self):
        item_lines = []
        with open('itemAttribute.csv', 'r') as f:
            item_lines = f.readlines()
            f.close()

        for item_line in item_lines:
            line = item_line.strip()
            if line == "":
                continue
            item_id, attr1, attr2 = line.split(',')
            try:
                self._item_attrs[self._item_dict[int(item_id)]] = (float(attr1), float(attr2))
            except KeyError:
                pass

        for u, i, r, err in self._sprase_matrix_with_err:
            self._user_item_attrs[u].append((i, err, self._item_attrs[i][0], self._item_attrs[i][1]))

    def _linear(self, user, user_item_attr):
        # equation: err = a*attr1 + b*attr2 + c
        # 为每个用户分析出方程参数
        def func(x, y, p): # 回归函数
            a, b, c = p
            return a * x + b * y + c
        def residuals(p, z, x, y): # 残差函数
            return z - func(x, y, p)
        l = len(self._user_item_attrs[user])
        x = np.array([self._user_item_attrs[user][i][2] for i in range(0, l)])
        y = np.array([self._user_item_attrs[user][i][3] for i in range(0, l)])
        z = np.array([self._user_item_attrs[user][i][1] for i in range(0, l)])
        plsq = optimize.leastsq(residuals, [0, 0, 0], args=(z, x, y)) # 最小二乘法拟合
        a, b, c = plsq[0] # 获得拟合结果 
        return (a, b, c)
        

    def linear(self):
        if self._do_divide:
            self.divide_train_and_test_set()
        elif os.path.exists(self._TRAIN_SET) == False or os.path.exists(self._TEST_SET) == False:
            self.divide_train_and_test_set()

        if self._do_train:
            self.train()
        elif os.path.exists(self._BI_VECTOR) == False or os.path.exists(self._BU_VECTOR) == False or os.path.exists(self._P_MATRIX) == False or os.path.exists(self._Q_MATRIX) == False or os.path.exists(self._SPRASE) == False:
            self.train()
        else: # 如果有训练结果文件，就可以直接读取训练矩阵了
            print('Loading train data from file......')
            start = timeit.default_timer()
            self._bi = []
            self._bu = []
            self._p = []
            self._q = []
            with open(self._BU_VECTOR, 'rb') as f:
                byte_str = f.read(4)
                user_len = struct.unpack('i', byte_str)[0]
                for i in range(0, user_len):
                    byte_str = f.read(8)
                    l = struct.unpack('d', byte_str)[0]
                    self._bu.append(l)
                f.close()
            with open(self._BI_VECTOR, 'rb') as f:
                byte_str = f.read(4)
                item_len = struct.unpack('i', byte_str)[0]
                for i in range(0, item_len):
                    byte_str = f.read(8)
                    l = struct.unpack('d', byte_str)[0]
                    self._bi.append(l)
                f.close()
            with open(self._P_MATRIX, 'rb') as f:
                byte_str = f.read(4)
                user_len = struct.unpack('i', byte_str)[0]
                byte_str = f.read(4)
                factor_len = struct.unpack('i', byte_str)[0]
                for i in range(0, user_len):
                    new_list = []
                    for j in range(0, factor_len):
                        byte_str = f.read(8)
                        l = struct.unpack('d', byte_str)[0]
                        new_list.append(l)
                    self._p.append(new_list)
                f.close()
            with open(self._Q_MATRIX, 'rb') as f:
                byte_str = f.read(4)
                item_len = struct.unpack('i', byte_str)[0]
                byte_str = f.read(4)
                factor_len = struct.unpack('i', byte_str)[0]
                for i in range(0, item_len):
                    new_list = []
                    for j in range(0, factor_len):
                        byte_str = f.read(8)
                        l = struct.unpack('d', byte_str)[0]
                        new_list.append(l)
                    self._q.append(new_list)
                f.close()
            with open(self._SPRASE, 'rb') as f:
                byte_str = f.read(8)
                self._train_mean = struct.unpack('d', byte_str)[0]
                byte_str = f.read(4)
                self.records = struct.unpack('i', byte_str)[0]
                for ii in range(0, self.records):
                    byte_str = f.read(4)
                    u = struct.unpack('i', byte_str)[0]
                    byte_str = f.read(4)
                    i = struct.unpack('i', byte_str)[0]
                    byte_str = f.read(8)
                    r = struct.unpack('d', byte_str)[0]
                    byte_str = f.read(8)
                    err = struct.unpack('d', byte_str)[0]
                    self._sprase_matrix_with_err.append((u, i, r, err))
                f.close()
            end = timeit.default_timer()
            self._all_time += (end - start)
            print('Time cost: %fs' % (end - start))

        print('Loading item attrs......')
        start = timeit.default_timer()
        self._prepare_item_attr()
        end = timeit.default_timer()
        self._all_time += (end - start)
        print('Time cost: %fs' % (end - start))

        print('Linear analysis......')
        start = timeit.default_timer()
        for k, v in self._user_item_attrs.items():
            param = self._linear(k, v)
            self._user_param[k] = param
        end = timeit.default_timer()
        self._all_time += (end - start)
        print('Time cost: %fs' % (end - start))

    def _linear_predict(self, u, i):
        return self._user_param[u][0] * self._item_attrs[i][0] + self._user_param[u][1] * self._item_attrs[i][1] + self._user_param[u][2]

    def test_model(self):
        self.linear()
        print('Model testing......')
        start = timeit.default_timer()
        records = 0
        rmse = 0.0
        rmse_improved = 0.0
        resultf = open('model_test_result.csv', 'w')
        with open(self._TEST_SET, 'r') as f:
            for line in f.readlines():
                if line == "":
                    continue
                user_id, item_id, rate_str = line.split(',')
                u = self._user_dict[int(user_id)]
                i = self._item_dict[int(item_id)]
                r = float(rate_str)
                rp1 = self._train_mean + self._bu[u] + self._bi[i] + self._dot(u, i)
                rp2 = rp1 + self._linear_predict(u, i)
                if (rp1 < self._scale[0]):
                    rp1 = float(self._scale[0])
                if (rp1 > self._scale[1]):
                    rp1 = float(self._scale[1])
                err = r - rp1
                rmse += err ** 2
                if (rp2 < self._scale[0]):
                    rp2 = float(self._scale[0])
                if (rp2 > self._scale[1]):
                    rp2 = float(self._scale[1])
                err = r - rp2
                rmse_improved += err ** 2
                resultf.write('%d,%d,%f,%f,%f\n' % (int(user_id), int(item_id), rp1, rp2, r))
                records += 1
            rmse /= records
            rmse = math.sqrt(rmse)
            rmse_improved /= records
            rmse_improved = math.sqrt(rmse_improved)
            f.close()
        resultf.close()
        end = timeit.default_timer()
        print('RMSE in test set: %f' % rmse)
        print('Improved RMSE in test set: %f' % rmse_improved)
        self._all_time += (end - start)
        print('Time cost: %fs' % (end - start))

    def predict(self):
        print('Predicting......')
        start = timeit.default_timer()
        test_file = []
        with open(self._TEST_FILE, 'r') as f:
            test_file = f.readlines()
            f.close()
        user_id = ''
        item_id = ''
        uid = 0
        iid = 0
        resultf1 = open('result1.txt', 'w') # 优化前
        resultf2 = open('result2.txt', 'w') # 优化后
        for test in test_file:
            line = test.strip()
            if line.find('|') != -1:
                user_id, user_item_count = line.split('|')
                uid = int(user_id)
                resultf1.write(line + '\n')
                resultf2.write(line + '\n')
            else:
                if line == "":
                    continue
                iid = int(line)
                u = self._user_dict[uid]
                i = self._item_dict[iid]
                rp1 = self._train_mean + self._bu[u] + self._bi[i] + self._dot(u, i)
                rp2 = rp1 + self._linear_predict(u, i)
                if (rp1 < self._scale[0]):
                    rp1 = float(self._scale[0])
                if (rp1 > self._scale[1]):
                    rp1 = float(self._scale[1])
                if (rp2 < self._scale[0]):
                    rp2 = float(self._scale[0])
                if (rp2 > self._scale[1]):
                    rp2 = float(self._scale[1])
                resultf1.write('%d  %f  \n' % (iid, rp1))
                resultf2.write('%d  %f  \n' % (iid, rp2))
                
        resultf1.close()
        resultf2.close()
        end = timeit.default_timer()
        self._all_time += (end - start)
        print('Time cost: %fs' % (end - start))

if __name__ == "__main__":
    svd = SVD_improved()
    svd.stat()
    #svd.divide_train_and_test_set()
    #svd.train()
    #svd.linear()
    svd.test_model()
    svd.predict()
    