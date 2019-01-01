import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

import copy
import math
import os
import progressbar
import numpy as np

DEBUG = False

# Hyperparameters
num_epochs = 5
num_classes = 10
batch_size = 256
learning_rate = 0.001

# Dataset
#DATA_PATH = '/data/mnist'
DATA_PATH = '~/work/mnist'
MODEL_STORE_PATH = 'saved_models'

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# MNIST dataset
data_parts = ['train', 'valid']
mnist_datasets = dict()
#mnist_datasets['train'] = torchvision.datasets.MNIST(root=DATA_PATH, train=True, transform=trans, download=False)
#mnist_datasets['valid'] = torchvision.datasets.MNIST(root=DATA_PATH, train=False, transform=trans, download=False)
mnist_datasets['train'] = torchvision.datasets.MNIST(root=DATA_PATH, train=True, transform=trans, download=True)
mnist_datasets['valid'] = torchvision.datasets.MNIST(root=DATA_PATH, train=False, transform=trans, download=True)

dataloaders = {p: DataLoader(mnist_datasets[p], batch_size=batch_size,
		shuffle=True, num_workers=4) for p in data_parts}

dataset_sizes = {p: len(mnist_datasets[p]) for p in data_parts}
num_batch = dict()
num_batch['train'] = math.ceil(dataset_sizes['train'] / batch_size)
num_batch['valid'] = math.ceil(dataset_sizes['valid'] / batch_size)
print(num_batch)


"""
# Define model
class TheModelClass(nn.Module):

	def __init__(self):
		super(TheModelClass, self).__init__()
		# class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

		self.conv1 = nn.Conv2d(1, 6, 5)
		self.pool  = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1   = nn.Linear(16 * 5 * 5, 120)
		self.fc2   = nn.Linear(120, 84)
		self.fc3   = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x
"""

"""
class TheModelClass(nn.Module):  # Net
	def __init__(self):
		super(TheModelClass, self).__init__()
		self.conv1 = nn.Conv2d(1, 20, 5, 1)
		self.conv2 = nn.Conv2d(20, 50, 5, 1)
		self.fc1 = nn.Linear(4*4*50, 500)
		self.fc2 = nn.Linear(500, 10)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x, 2, 2)
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, 2, 2)
		x = x.view(-1, 4*4*50)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)
"""

conv = lambda x, f1, f2, k, s=1, p=0: F.relu(nn.Conv2d(f1, f2, k, s)(x))

class TheModelClass(nn.Module):  # Net
	def __init__(self):
		super(TheModelClass, self).__init__()
		self.conv1 = nn.Conv2d(1, 20, 5, 1)
		self.conv2 = nn.Conv2d(20, 50, 5, 1)
		self.fc1 = nn.Linear(4*4*50, 500)
		self.fc2 = nn.Linear(500, 10)

	def forward(self, x):
		x = F.relu(nn.Conv2d(1, 20, 5, 1)(x))
		#x = conv(x, f1=1, f2=8, k=4)
		x = F.max_pool2d(x, 2, 2)
		x = F.relu(nn.Conv2d(20, 50, 5, 1)(x))
		#x = conv(x, f1=8, f2=50,  k=4)
		x = F.max_pool2d(x, 2, 2)
		x = x.view(-1, 4*4*50)
		x = F.relu(nn.Linear(4*4*50, 500)(x))
		x = nn.Linear(500, 10)(x)
		return F.log_softmax(x, dim=1)		


def accuracy_top1(outputs, labels):

	batch_size = len(outputs)
	res = np.zeros(batch_size, dtype=int)
	for i in range(batch_size):
		output = outputs[i].detach().cpu().numpy()
		label = int(labels[i])
		predict = np.argmax(output)
		res[i] = 1 if label==predict else 0
		if DEBUG: print('i={}: res={} (label={}, predict={})'.format(i, res[i], label, predict))
	return np.mean(res)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
	
	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))

		# Each epoch has a training and validation phase
		for phase in data_parts:
			if phase == 'train':
				scheduler.step()
				model.train()  # Set model to training mode
			else:
				model.eval()   # Set model to evaluate mode

			running_loss = 0.0
			running_corrects = 0

			bar = progressbar.ProgressBar(maxval=num_batch[phase]).start()

			# Iterate over data.
			acc1_list = []

			for i_batch, (inputs, labels) in enumerate(dataloaders[phase]):
				inputs = inputs.to(device)
				labels = labels.to(device)

				bar.update(i_batch)

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward
				# track history if only in train
				with torch.set_grad_enabled(phase == 'train'):

					outputs = model(inputs)
					_, preds = torch.max(outputs, 1)
					loss = criterion(outputs, labels)

					# backward + optimize only if in training phase
					if phase == 'train':
						loss.backward()
						optimizer.step()


				# statistics
				acc1 = accuracy_top1(outputs, labels)
				#if DEBUG: print('-')
				#if DEBUG: print('accuracy(2): acc1={}, acc6={}'.format(acc1, acc6))

				acc1_list.append(acc1)

				if DEBUG:
					print('epoch {} [{}]: {}/{}'.format(epoch, phase, i_batch, num_batch[phase]))
					#print('preds: ', preds)
					#print('labels:', labels.data)
					print('match: ', int(torch.sum(preds == labels.data)))
					print('acc1={:.4f}'.format(acc1))

				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)			 

			bar.finish()   

			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_acc = running_corrects.double() / dataset_sizes[phase]
			epoch_acc1 = np.mean(acc1_list)

			print('Epoch {} [{}]: loss={:.4f}, acc={:.4f}, top1={:.4f}' .
				format(epoch, phase, epoch_loss, epoch_acc, epoch_acc1))

			# deep copy the model
			if phase == 'valid' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())

		print()

	#time_elapsed = time.time() - since
	#print('Training complete in {:.0f}m {:.0f}s'.format(
	#	time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))

	# load best model weights
	model.load_state_dict(best_model_wts)
	return model


if __name__ == '__main__':

	# Initialize model
	model = TheModelClass()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	#device = torch.device("cpu")
	model = model.to(device)
	#model = model.cuda()

	# Print model's state_dict
	print("Model's state_dict:")
	for param_tensor in model.state_dict():
		print(param_tensor, "\t", model.state_dict()[param_tensor].size())


	# Initialize optimizer
	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	# Decay LR by a factor of 0.1 every 7 epochs
	exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

	# Print optimizer's state_dict
	print("Optimizer's state_dict:")
	for var_name in optimizer.state_dict():
		print(var_name, "\t", optimizer.state_dict()[var_name])


	criterion = nn.CrossEntropyLoss()	

	model = train_model(model, criterion, optimizer, exp_lr_scheduler,
	num_epochs=30)			

	PATH = 'saved/model'
	#torch.save(model, PATH)  