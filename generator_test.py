import numpy as np

def generator():
	x = 0
	for j in range(10):
		for i in range(5):
			x += 1
			yield x

y = generator()
for j in range(10):
	for i in range(5):
		print(y.__next__())