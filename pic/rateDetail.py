import matplotlib.pyplot as plt
rate_num = []
with open('../rateDetail.csv', 'r') as f:
    for line in f.readlines():
        l = line.strip()
        if l == "":
            continue
        rate, num = l.split(',')
        rate_num.append(int(num))
    f.close()
print(rate_num)
plt.bar(range(0,101), rate_num, color='fuchsia', alpha=0.5)
#plt.hist(rate_num,bins=len(rate_num),color='fuchsia')#alpha设置透明度，0为完全透明
plt.xlabel('scores') 
plt.ylabel('count')
'''
plt.xlim(0,100)#设置x轴分布范围
plt.ylim(0)
'''
plt.show() 