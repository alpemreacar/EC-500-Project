import glob
import scipy.misc
import numpy as np
from skimage.measure import compare_ssim as ssim

pix2pix_dest = '../../pytorch-CycleGAN-and-pix2pix-master/results/day2night_pix2pix/test_latest/images/'
cyclGAN_dest = '../output_pictures/v5_Pictures/Test/night2day/testB/'

cyclGAN_nig = '../data/night2day/testA/tst_'
cyclGAN_day = '../data/night2day/testB/tst_'

picture_list = glob.glob(cyclGAN_dest + "*.jpg")
print("Number of test cases: %d" %len(picture_list))

width   = 256
channel = 3
pix2pix_MSE = 0
cyclGAN_MSE = 0
pix2pix_ssim = 0
cyclGAN_ssim = 0

for picture in picture_list:
	p_id = picture.split('/')[-1][4:-4]
	img_cyc = scipy.misc.imread(picture).astype(np.uint8)
	img_cyc_day_real = img_cyc[:,:width,        :]
	img_cyc_nig_gnrt = img_cyc[:,width: width*2,:]

	img_cyc_day_real = scipy.misc.imread(cyclGAN_day + p_id + ".jpg").astype(np.uint8)
	img_cyc_nig_real = scipy.misc.imread(cyclGAN_nig + p_id + ".jpg").astype(np.uint8)

	img_pix_day_real = scipy.misc.imread(pix2pix_dest + p_id + "_real_A.png").astype(np.uint8)
	img_pix_nig_gnrt = scipy.misc.imread(pix2pix_dest + p_id + "_fake_B.png").astype(np.uint8)
	img_pix_nig_real = scipy.misc.imread(pix2pix_dest + p_id + "_real_B.png").astype(np.uint8)
	
	cyclGAN_MSE += np.sqrt(np.sum((img_cyc_nig_gnrt-img_cyc_nig_real)**2))/(width*width*channel)
	pix2pix_MSE += np.sqrt(np.sum((img_pix_nig_gnrt-img_pix_nig_real)**2))/(width*width*channel)

	cyclGAN_ssim += ssim(img_cyc_nig_gnrt, img_cyc_nig_real, multichannel=True)
	pix2pix_ssim += ssim(img_pix_nig_gnrt, img_pix_nig_real, multichannel=True)

cyclGAN_MSE = cyclGAN_MSE / len(picture_list)
pix2pix_MSE = pix2pix_MSE / len(picture_list)

cyclGAN_ssim = cyclGAN_ssim / len(picture_list)
pix2pix_ssim = pix2pix_ssim / len(picture_list)

print("MSE : CycleGAN %.7f, pix2pix %.7f" %(cyclGAN_MSE, pix2pix_MSE))
print("SSIM: CycleGAN %.7f, pix2pix %.7f" %(cyclGAN_ssim, pix2pix_ssim))

"""
# Print statements
Number of test cases: 2287
MSE : CycleGAN 0.0229170, pix2pix 0.0226542
SSIM: CycleGAN 0.3790817, pix2pix 0.3774814
"""
