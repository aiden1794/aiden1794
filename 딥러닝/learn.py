from matrix import *
from network import Network
from functions import *
import math
import random
import mnist
import pickle

(train, train_label), (test, test_label) = mnist.load_mnist()
train, train_label, test, test_label = train.tolist(), train_label.tolist(), test.tolist(), test_label.tolist()

net = Network([784, 100, 100, 100, 10])

try:
	with open('D:/learning.pkl', 'rb') as f:
		net.layers = pickle.load(f)
except:
	pass

test_data_acc = 0

iters = 0
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

try:
	while True:
		x = []
		t = []

		for j in range(batch_size):
			index = random.randrange(0, len(train))

			x.append(train[index])
			t.append(train_label[index])

		loss = net.loss(x, t)

		dout = 1
		dout = net.lastLayer.backward(dout)

		for layer in list(reversed(net.layers)):
			dout = layer.backward(dout)

		for index in range(len(net.layers)):

			if isinstance(net.layers[index], Mat_mul):

				net.layers[index].W = sub(net.layers[index].W, con(net.layers[index].dW, learning_rate))
				net.layers[index].b = sub(net.layers[index].b, con(net.layers[index].db, learning_rate))

		correct = 0

		test_index = random.randrange(0, len(test))

		print('훈련:', loss)
		print('시험:', net.loss([test[test_index]], [test_label[test_index]]))
		
except KeyboardInterrupt:
	with open('D:/learning.pkl', 'wb') as f:
		pickle.dump(net.layers, f)