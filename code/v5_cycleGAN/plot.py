import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 'horse2zebra', 1067, 10.
# 'apple2orange', 995, 10.
# 'night2day', 1049, 10.
# 'summer2winter_yosemite', 962, 10.

dataset, step_per_epoch, gen_scale = 'vangogh2photo', 400, 10.
loss_arr = np.load("loss_arr/" + dataset + '.npy')
gen_loss = loss_arr[:,0]
d_b_loss = loss_arr[:,1]
d_a_loss = loss_arr[:,2]

avg_gen = np.mean(gen_loss.reshape(-1,step_per_epoch), axis=1)
avg_d_b = np.mean(d_b_loss.reshape(-1,step_per_epoch), axis=1)
avg_d_a = np.mean(d_a_loss.reshape(-1,step_per_epoch), axis=1)
plt.cla()
plt.plot(avg_gen * 1. / gen_scale, label='Generator / %d' %gen_scale)
plt.plot(avg_d_b, label='Discriminator B')
plt.plot(avg_d_a, label='Discriminator A')
plt.grid()
#plt.axis([-5, 110, .1, 1.0])
plt.title('Cycle GAN Losses (' + dataset + ')')
plt.legend()
plt.savefig("loss_plot/" + dataset + "21.jpg")