# Reference https://github.com/LynnHo/CycleGAN-Tensorflow-PyTorch

from utils import *

# Hyperparameters
dataset = 'night2day'
crop_size = 256
channel = 3
## Graph

# Placeholders
a_real_img_plhdr = tf.placeholder(tf.float32, (None, crop_size, crop_size, channel), name='a_real_images')
b_real_img_plhdr = tf.placeholder(tf.float32, (None, crop_size, crop_size, channel), name='b_real_images')


# CycleGAN Network

# Generators, we don't need discriminator in test time!
gnrt_a_b     = generator(a_real_img_plhdr, scope='a_b')
gnrt_b_a     = generator(b_real_img_plhdr, scope='b_a')

gnrt_b_a_b   = generator(gnrt_b_a        , scope='a_b')
gnrt_a_b_a   = generator(gnrt_a_b        , scope='b_a')

## Restore model
all_vars = tf.trainable_variables()
gnrt_vars   = [var for var in all_vars if 'generator'       in var.name]
saver = tf.train.Saver(var_list=None)

sess = tf.Session()
saver.restore(sess, '../../models/v6_CycleGAN/' + dataset + '_gen.ckpt')

## Images
tst_img_a_list  = glob.glob('../../data/' + dataset + '/testA/*.jpg')
tst_img_b_list  = glob.glob('../../data/' + dataset + '/testB/*.jpg')

save_path_a = './Pictures/Test/' + dataset + '/testA'
save_path_b = './Pictures/Test/' + dataset + '/testB'

# Convert A to B!
for i in range(len(tst_img_a_list)):
    a_img = img_read_resize(tst_img_a_list[i], crop_size, crop_size)
    a_img = np.reshape(a_img, [1, crop_size, crop_size, channel]) # Reshape img for tf operations!
    a_b_gen, a_b_a_gen = sess.run([gnrt_a_b, gnrt_a_b_a], feed_dict={a_real_img_plhdr:a_img})
    image_concat = np.concatenate((a_img, a_b_gen, a_b_a_gen), axis=0)
    img_name = tst_img_a_list[i].split('/')[-1]
    img_write(img_merge(image_concat,row=1,col=3), './Pictures/Test/' + dataset + '/testA/' +img_name)

print("A -> B conversion is done!")

# Convert B to A!
for i in range(len(tst_img_b_list)):
    b_img = img_read_resize(tst_img_b_list[i], crop_size, crop_size)
    b_img = np.reshape(b_img, [1, crop_size, crop_size, channel]) # Reshape img for tf operations!
    b_a_gen, b_a_b_gen = sess.run([gnrt_b_a, gnrt_b_a_b], feed_dict={b_real_img_plhdr:b_img})
    image_concat = np.concatenate((b_img, b_a_gen, b_a_b_gen), axis=0)
    img_name = tst_img_b_list[i].split('/')[-1]
    img_write(img_merge(image_concat,row=1,col=3), './Pictures/Test/' + dataset + '/testB/' +img_name)

print("B -> A conversion is done!")
