import numpy as np
import pickle

labels_test = []
labels_train = []

with open('label_test.txt', 'r') as file1:
	for line in file1:
		labels_test.append(int(line.strip()))

pickle.dump(labels_test, open('label_test.p', 'wb'))		 


with open('label_train.txt', 'r') as file1:
	for line in file1:
		labels_train.append(int(line.strip()))

pickle.dump(labels_train, open('label_train.p', 'wb'))		 
