import sys,os


number = 10
anchor_size = 9
N = 6
M = 8

grid_size = N * M
size_t = anchor_size * grid_size

for n in range(number):
	for a in range(anchor_size):
		for p in range(grid_size):
			total = n * anchor_size * grid_size + a * grid_size + p
			print n, a, p
			new_n = total / (size_t)
			new_a = (total / grid_size) % (anchor_size)
			new_p = total % grid_size
			print new_n, new_a, new_p

			