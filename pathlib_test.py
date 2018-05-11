from pathlib import Path
import os
import utils
import numpy as np

pathlist = Path('../../3D-R2N2/ShapeNet/ShapeNetRendering/').glob('**/*00.png')
print(len(list(pathlist)))

for path in pathlist:
	input_a = utils.load_image_com(str(path))[:,:,:3]
	print(input_a.shape)
	break