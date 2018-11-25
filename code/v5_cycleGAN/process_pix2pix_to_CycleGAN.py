import glob
import scipy.misc
import numpy as np

dataset_pix2p = "../../data/night2day_pix2pix"
dataset_destn = "../../data/night2day"
### There are 17823 training images, 10 validation images, 2287 test images
### There are 20000 images in total!
### Each image has shape of (256, 512, 3)

trn_img_list  = glob.glob(dataset_pix2p + '/train/*.jpg')
val_img_list  = glob.glob(dataset_pix2p + '/val/*.jpg')
tst_img_list  = glob.glob(dataset_pix2p + '/test/*.jpg')

width = 256

# Unpair images for CycleGAN!!
# Gather train and validation images and put them in trainA/B folders
trn_count = 0
for i in range(len(trn_img_list)):
	path = trn_img_list[i]
	img = scipy.misc.imread(path).astype(np.uint8)
	img_a = img[:,:width,:]
	img_b = img[:,width:,:]
	img_name = 'trn_' + path.split('/')[-1]
	scipy.misc.imsave(dataset_destn + "/trainA/" + img_name, img_a)
	scipy.misc.imsave(dataset_destn + "/trainB/" + img_name, img_b)
	trn_count += 1

print('train folder is parsed!')

for i in range(len(val_img_list)):
	path = val_img_list[i]
	img = scipy.misc.imread(path).astype(np.uint8)
	img_a = img[:,:width,:]
	img_b = img[:,width:,:]
	img_name = 'val_' + path.split('/')[-1]
	scipy.misc.imsave(dataset_destn + "/trainA/" + img_name, img_a)
	scipy.misc.imsave(dataset_destn + "/trainB/" + img_name, img_b)
	trn_count += 1

print('val folder is parsed!')

# Gather test images and put them in testA/B folders
tst_count = 0
for i in range(len(tst_img_list)):
	path = tst_img_list[i]
	img = scipy.misc.imread(path).astype(np.uint8)
	img_a = img[:,:width,:]
	img_b = img[:,width:,:]
	img_name = 'tst_' + path.split('/')[-1]
	scipy.misc.imsave(dataset_destn + "/testA/" + img_name, img_a)
	scipy.misc.imsave(dataset_destn + "/testB/" + img_name, img_b)
	tst_count += 1

print('test folder is parsed!')
print("Each Train folder has %d images." %(trn_count))
print("Each Test  folder has %d images." %(tst_count))

# Each Train folder has 17833 images.
# Each Test  folder has 2287 images.


