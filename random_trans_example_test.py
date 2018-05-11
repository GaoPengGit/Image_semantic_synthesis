import numpy as np

for _ in range(1):
	distance = np.random.uniform(-1, 1, size=3)

	a = '%f %f %f %f %f %f' % (0,0,0,distance[0],distance[1],distance[2])
	
	with open("trans_test_example_rotation.txt","a") as f:
		f.write(a + '\n')