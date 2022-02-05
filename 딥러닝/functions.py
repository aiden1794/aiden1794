import math
import random
from matrix import *

class ReLu:
	def __init__(self):
		self.grad_mask = None

	def forward(self, x):
		out = arr_like(x)

		self.grad_mask = arr_like(x)

		for i in range(len(x)):
			for j in range(len(x[0])):
				if x[i][j]>0:
					out[i][j] = x[i][j]
					self.grad_mask[i][j] = 1

		return out

	def backward(self, dout):
		dx = dout

		for i in range(len(dx)):
			for j in range(len(dx[0])):
				if not self.grad_mask[i][j]:
					dx[i][j] = 0

		return dx

def He_init(i_size, j_size, in_node):

	return [[random.gauss(0, math.sqrt(2/in_node)) for j in range(j_size)] for j in range(i_size)]

class Mat_mul:
	def __init__(self, W, b):
		self.W = W
		self.b = b

		self.x = None

		self.dW = None
		self.db = None

	def forward(self, x):
		self.x = x

		out = bias(mul(x, self.W), self.b)

		return out

	def backward(self, dout):
		dx = mul(dout, T(self.W))

		self.dW = mul(T(self.x), dout)
		self.db = arr_like(self.b)

		for i in range(len(dout)):
			for j in range(len(dout[0])):
				self.db[0][j] += dout[i][j]

		return dx

class SoftmaxCEE:
	def __init__(self):
		self.y = None
		self.t = None

	def forward(self, x, t):

		loss = 0

		self.t = t
		self.y = arr_like(x)

		for i in range(len(x)):
			s = 0
			c = max(x[i])

			for j in range(len(x[0])):
				s += math.exp(x[i][j]-c)

			for j in range(len(x[0])):
				self.y[i][j] = math.exp(x[i][j]-c)/s

		for i in range(len(self.y)):
			for j in range(len(self.y[0])):
		
				loss -= self.t[i][j]*math.log(self.y[i][j])

		return loss/len(self.y)

	def backward(self, dout=1):
		dx = con(sub(self.y, self.t), 1/len(self.y))

		return dx

if __name__ == '__main__':
	x = [[1,2,3]]
	t = [[0.09003057317038046, 0.24472847105479767, 0.6652409557748219]]

	layer = SoftmaxCEE()

	print(layer.forward(x, t))
	print(layer.backward(1))