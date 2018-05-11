import numpy as np

for _ in range(50):
	# angle = np.random.uniform(-3, 3, size = 3)
	angle = np.loadtxt('one_rotation_test_example.txt')
	distance = np.random.uniform(-1, 1, size = 3)

	a = '%f %f %f %f %f %f' % (angle[0], angle[1], angle[2], distance[0], distance[1], distance[2])
	
	with open("one_rotation_trans_test_example.txt","a") as f:
		f.write(a + '\n')