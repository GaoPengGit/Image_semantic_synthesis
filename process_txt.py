import numpy as np

a = np.loadtxt('one_rotation_trans_test_example.txt')

for i in range(50):
	s = '%f %f %f %f %f %f' % (0,0,0,a[i][3],a[i][4],a[i][5])
	with open('one_rotation_trans_test_example_test.txt','a') as f:
		f.write(s + '\n')