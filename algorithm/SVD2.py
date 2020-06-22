from surprise import Dataset, Reader, BaselineOnly, accuracy, SVD, SVDpp
from surprise.model_selection import cross_validate, KFold, PredefinedKFold, train_test_split
import os
train_file = []
user_id = ''
item_id = ''
rate_str = ''
write_file = open('train.csv', 'w')
with open('../data-new/train.txt', 'r') as f:
    train_file = f.readlines()
    f.close()
for train in train_file:
    line = train.strip()
    if line.find('|') != -1:
        user_id, user_item_count = line.split('|')
    else:
        if line == "":
            continue
        item_id, rate_str = line.split()
        write_file.write('%s,%s,%s\n' % (user_id, item_id, rate_str))
write_file.close()
print("reading......")
reader = Reader(line_format='user item rating', sep=',', rating_scale=(0,100))
data = Dataset.load_from_file("train.csv", reader=reader)

algo = SVD(n_factors=10, n_epochs=10, lr_all=0.015, reg_all=0.01)
'''
bsl_options = {'method': 'als',
               'n_epochs': 5,
               'reg_u': 12,
               'reg_i': 5
               }
'''
#algo = BaselineOnly(bsl_options=bsl_options)
'''
kf = KFold(n_splits=3) 
print('------begin train user cf model------------')
for trainset, testset in kf.split(train_cf):
    # 训练并测试算法
    print("aaa")
    algo.fit(trainset)
    predictions = algo.test(testset)
    accuracy.rmse(predictions, verbose=True)
'''
#cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("spilting......")
trainset, testset = train_test_split(data, test_size=.20)
print("fitting......")
algo.fit(trainset)
predictions = algo.test(testset)

# RMSE
accuracy.rmse(predictions)
