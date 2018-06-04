# tırmanma yarışması :)
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(123)
all_walks = []

for i in range(5):
    random_walk = [0]
    for x in range(100):
        step = random_walk[-1]
        dice = np.random.randint(1, 7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1, 7)
            # sakarlık payı :)
        if np.random.rand() <= 0.001:
            step = 0
        random_walk.append(step)
    all_walks.append(random_walk)

# all_walks np array çevrilir
np_aw = np.array(all_walks)
plt.plot(np_aw)
plt.show()

# plot temizleniyor
plt.clf()

# np_aw içerisindeki sub arrayler np_aw_t ye transpose ediliyor
np_aw_t = np.transpose(np_aw)
plt.plot(np_aw_t)
plt.show()
