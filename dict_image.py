import utils
import numpy as np
from tqdm import tqdm 

path = '../../chair_only_rotation_database/chair_rotation_00/'
input_Sym = utils.load_image_com(path + 'chair_rotation_00_only_00.png')[:,:,:3]
for i in tqdm(range(1,1000)):
	tmp_image = utils.load_image_com(path + 'chair_rotation_00_only_%.2d.png' % i)[:,:,:3]
	input_Sym = np.concatenate((input_Sym,tmp_image),axis = -1)
np.save('/home/gaopeng/dict_image.npy' , input_Sym)