import matplotlib.pyplot as plt
x = [i for i in range(1, 21)]
y = [27.096739,22.715155,21.110017,20.390432,20.002774,19.774146,19.626631,19.528458,
19.462330,19.421008,19.398327,19.386584,19.380626,19.377627,19.376123,19.375370,19.374993,
19.374805,19.374710,19.374663]
plt.plot(x,y,'-s')
plt.xticks(x)
plt.xlabel('epochs')
plt.ylabel('RMSE in train set')
plt.show()