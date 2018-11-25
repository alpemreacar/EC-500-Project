# Reference https://github.com/LynnHo/CycleGAN-Tensorflow-PyTorch

from utils import *

# Hyperparameters
dataset = 'apple2orange'
load_size = 286
crop_size = 256
epoch = 75
batch_size = 1
learning_rate = 2e-4
channel = 3
cycle_loss_coef = 10
beta1 = .5
## Graph

# Placeholders
a_real_img_plhdr = tf.placeholder(tf.float32, (None, crop_size, crop_size, channel), name='a_real_images')
b_real_img_plhdr = tf.placeholder(tf.float32, (None, crop_size, crop_size, channel), name='b_real_images')
a_b_histry_plhdr = tf.placeholder(tf.float32, (None, crop_size, crop_size, channel), name='a_to_b_history_images')
b_a_histry_plhdr = tf.placeholder(tf.float32, (None, crop_size, crop_size, channel), name='b_to_a_history_images')

# CycleGAN Network

# Generators
gnrt_a_b     = generator(a_real_img_plhdr, scope='a_b')
gnrt_b_a     = generator(b_real_img_plhdr, scope='b_a')

gnrt_b_a_b   = generator(gnrt_b_a        , scope='a_b')
gnrt_a_b_a   = generator(gnrt_a_b        , scope='b_a')

# Discriminators
disc_a        = discriminator(a_real_img_plhdr, scope='a')
disc_b_a_real = discriminator(gnrt_b_a        , scope='a')
disc_b_a_hist = discriminator(b_a_histry_plhdr, scope='a')

disc_b        = discriminator(b_real_img_plhdr, scope='b')
disc_a_b_real = discriminator(gnrt_a_b        , scope='b')
disc_a_b_hist = discriminator(a_b_histry_plhdr, scope='b')

# Losses

g_loss_a_b    = tf.losses.mean_squared_error(labels=tf.ones_like(disc_a_b_real), predictions=disc_a_b_real)
g_loss_b_a    = tf.losses.mean_squared_error(labels=tf.ones_like(disc_b_a_real), predictions=disc_b_a_real)
cyc_loss_a    = tf.losses.absolute_difference(labels=a_real_img_plhdr          , predictions=gnrt_a_b_a)
cyc_loss_b    = tf.losses.absolute_difference(labels=b_real_img_plhdr          , predictions=gnrt_b_a_b)
sum_g_loss    = g_loss_a_b + g_loss_b_a + cycle_loss_coef * (cyc_loss_a + cyc_loss_b)

d_loss_a_real = tf.losses.mean_squared_error(labels=tf.ones_like(disc_a)        , predictions=disc_a)
d_loss_a_hist = tf.losses.mean_squared_error(labels=tf.zeros_like(disc_b_a_hist), predictions=disc_b_a_hist)
sum_d_loss_a  = d_loss_a_real + d_loss_a_hist

d_loss_b_real = tf.losses.mean_squared_error(labels=tf.ones_like(disc_b)        , predictions=disc_b)
d_loss_b_hist = tf.losses.mean_squared_error(labels=tf.zeros_like(disc_a_b_hist), predictions=disc_a_b_hist)
sum_d_loss_b  = d_loss_b_real + d_loss_b_hist

# Optimizers

all_vars = tf.trainable_variables()

disc_a_vars = [var for var in all_vars if 'a_discriminator' in var.name]
disc_b_vars = [var for var in all_vars if 'b_discriminator' in var.name]
gnrt_vars   = [var for var in all_vars if 'generator'       in var.name]

d_a_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(sum_d_loss_a, var_list=disc_a_vars)
d_b_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(sum_d_loss_b, var_list=disc_b_vars)
g_opt   = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(sum_g_loss  , var_list=gnrt_vars  )


## Images

sess = tf.Session()
# Train
trn_img_a_list  = glob.glob('../../data/' + dataset + '/trainA/*.jpg')
trn_img_b_list  = glob.glob('../../data/' + dataset + '/trainB/*.jpg')
trn_img_a_objt = ImageData(sess, trn_img_a_list,  batch_size=batch_size, load_size=load_size, crop_size=crop_size)
trn_img_b_objt = ImageData(sess, trn_img_b_list,  batch_size=batch_size, load_size=load_size, crop_size=crop_size)
# Test
tst_img_a_list  = glob.glob('../../data/' + dataset + '/testA/*.jpg')
tst_img_b_list  = glob.glob('../../data/' + dataset + '/testB/*.jpg')
tst_img_a_objt = ImageData(sess, tst_img_a_list,  batch_size=batch_size, load_size=load_size, crop_size=crop_size)
tst_img_b_objt = ImageData(sess, tst_img_b_list,  batch_size=batch_size, load_size=load_size, crop_size=crop_size)

a_b_hist_objt = ImageHistory()
b_a_hist_objt = ImageHistory()

# Train
file = open("loss_log/" + dataset + ".txt", "w")
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
iter_num = min(len(trn_img_a_objt), len(trn_img_b_objt))// batch_size
loss_arr = np.zeros((epoch * iter_num, 3)) # Store sum_g_loss, sum_d_loss_a, and sum_d_loss_b for each batch
for e in range(epoch):
	for itr in range(iter_num):
		# Get batch
		batch_a = trn_img_a_objt.batch()
		batch_b = trn_img_b_objt.batch()
		batch_a_b, batch_b_a = sess.run([gnrt_a_b, gnrt_b_a], feed_dict={a_real_img_plhdr:batch_a, b_real_img_plhdr:batch_b})
		batch_a_b_hist = np.array(a_b_hist_objt(list(batch_a_b)))
		batch_b_a_hist = np.array(b_a_hist_objt(list(batch_b_a)))
		# Optimize
		_, batch_g_loss   = sess.run([g_opt, sum_g_loss], feed_dict={a_real_img_plhdr:batch_a, b_real_img_plhdr:batch_b})
		_, batch_d_loss_b = sess.run([d_b_opt, sum_d_loss_b], feed_dict={b_real_img_plhdr:batch_b, a_b_histry_plhdr:batch_a_b_hist})
		_, batch_d_loss_a = sess.run([d_a_opt, sum_d_loss_a], feed_dict={a_real_img_plhdr:batch_a, b_a_histry_plhdr:batch_b_a_hist})

		loss_arr[e * iter_num + itr,:] = np.asarray([batch_g_loss, batch_d_loss_b, batch_d_loss_a])
		if (itr+1) % 10 == 0:
			print("Epoch: %3d/%3d, Iteration: %5d/%5d, Disc a loss: %.4f, Disc b loss: %.4f, Gen loss: %.4f" %(e+1, epoch, itr+1, iter_num, batch_d_loss_a, batch_d_loss_b, batch_g_loss))
			file.write("Epoch: %3d/%3d, Iteration: %5d/%5d, Disc a loss: %.4f, Disc b loss: %.4f, Gen loss: %.4f \n" %(e+1, epoch, itr+1, iter_num, batch_d_loss_a, batch_d_loss_b, batch_g_loss))

		# Save images during training
		if (itr+1) % iter_num == 0:
			batch_a = tst_img_a_objt.batch()
			batch_b = tst_img_b_objt.batch()
			batch_a_b, batch_b_a, batch_b_a_b, batch_a_b_a = sess.run([gnrt_a_b, gnrt_b_a, gnrt_b_a_b, gnrt_a_b_a], feed_dict={a_real_img_plhdr:batch_a, b_real_img_plhdr:batch_b})
			image_concat = np.concatenate((batch_a, batch_a_b, batch_a_b_a, batch_b, batch_b_a, batch_b_a_b), axis=0)
			img_write(img_merge(image_concat,row=2,col=3), './Pictures/Training/%s/Epoch_%d_%d_Iteration_%d_%d.jpg' %(dataset,e+1, epoch, itr+1, iter_num))

# Save 10 more images after training
for i in range(10):
	batch_a = tst_img_a_objt.batch()
	batch_b = tst_img_b_objt.batch()
	batch_a_b, batch_b_a, batch_b_a_b, batch_a_b_a = sess.run([gnrt_a_b, gnrt_b_a, gnrt_b_a_b, gnrt_a_b_a], feed_dict={a_real_img_plhdr:batch_a, b_real_img_plhdr:batch_b})
	image_concat = np.concatenate((batch_a, batch_a_b, batch_a_b_a, batch_b, batch_b_a, batch_b_a_b), axis=0)
	img_write(img_merge(image_concat,row=2,col=3), './Pictures/Training/%s/FinalModel_%d.jpg' %(dataset,i+1))

np.save("loss_arr/" + dataset + '.npy', loss_arr)
saver.save(sess, '../../models/v5_CycleGAN/' + dataset + '_gen.ckpt')
sess.close()

