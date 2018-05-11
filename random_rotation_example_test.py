import numpy as np

for _ in range(50):
	angle = np.random.uniform(-3, 3, size=3)

	a = '%f %f %f %f %f %f' % (angle[0], angle[1], angle[2], 0, 0, 0)
	
	with open("only_rotation_test_example.txt","a") as f:
		f.write(a + '\n')