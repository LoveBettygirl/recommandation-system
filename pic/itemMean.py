import matplotlib.pyplot as plt
import numpy as np
rate_num = []
rate = []
with open('../itemMean.csv', 'r') as f:
    for line in f.readlines():
        l = line.strip()
        if l == "":
            continue
        itemid, rate_, num = l.split(',')
        rate_num.append(int(num))
        rate.append(float(rate_))
    f.close()
print(max(rate_num))
bins = np.arange(0, 500, 10)
plt.hist(rate_num,bins=bins,facecolor='fuchsia', alpha=0.5, edgecolor='black')
#plt.hist(rate,bins=bins,facecolor='purple', alpha=0.5, edgecolor='black')#alpha设置透明度，0为完全透明
plt.xlabel('item rate counts') 
plt.ylabel('count')
'''
plt.xlim(0,100)#设置x轴分布范围
plt.ylim(0)
'''
plt.show() 