from PIL import Image
import numpy as np
import sys
from network import Network
import pickle

net = Network([784, 100, 100, 100, 10])

try:
	with open('./learning.pkl', 'rb') as f:
		net.layers = pickle.load(f)
except:
	pass

im = Image.open(sys.argv[1])

arr = np.array(im)

flatten = [[]]

for i in range(arr.shape[0]):
	for j in range(arr.shape[1]):
		flatten[0].append(arr[i][j][0])

y = net.predict(flatten)

print('아마', y[0].index(max(y[0])), '일듯')

input()