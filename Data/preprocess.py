import re
import numpy as np 
import os
from nltk import word_tokenize
import pickle

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def extract_labels(fileName):
	fp=open(fileName,'r',encoding="utf8")
	labels = []
	for line in fp:
	    if '\t' in line:
	        arr= line.split('\t')
	        labels.append( int(arr[2].strip()))
	fp.close()
	labels = to_categorical(labels)
	return labels

def prepare_labels():
	train_data_original  = 'WikiQA-train.txt'
	test_data_original = 'WikiQA-test.txt'
	valid_data_original = 'WikiQA-dev.txt'

	train_labels = extract_labels(train_data_original)
	test_labels = extract_labels(test_data_original )
	valid_labels = extract_labels(valid_data_original)

	pickle.dump( train_labels, open( "train_labels.p", "wb" ) )
	pickle.dump( test_labels, open( "test_labels.p", "wb" ) )
	pickle.dump( valid_labels, open( "valid_labels.p", "wb" ) )

def zero_pad(X, seq_len):
    return np.array([ [0] * max(seq_len - len(x), 1) + x[:seq_len - 1] for x in X])

def read_instances(file_name, type):
	max_q_len= 27
	max_ans_len = 286

	fp=open(file_name,'r',encoding="utf8")
	embedding_matrix = []
	for line in fp:
		arr= line.strip().split()
		print(arr)
		arr = np.asarray([int(a) for a in arr])
			
		embedding_matrix.append(arr)      
      
	fp.close()
	if type == 'a':
		embedding_matrix = zero_pad(np.asarray(arr), seq_len = max_ans_len)
	elif type == 'q':
		embedding_matrix = zero_pad(np.asarray(arr), seq_len = max_q_len)
	return np.asarray(embedding_matrix)

def prepare_sequences():
	types1 = ['test', 'train']
	types2 = ['ques', 'ans']
	types3 = ['glove', 'word2vec']

	for type1 in types1:
		for type2 in types2:
			for type3 in types3:
				file_name = 'final_'+ type1 +'_'+ type2 + '_' + type3 + '.txt'
				if(type2 == 'ques'):
					embedding_matrix = read_instances('data_abhishek/'+ file_name, 'q')
				if(type2 == 'ans'):
					embedding_matrix = read_instances('data_abhishek/'+ file_name, 'a')
						
				pickle.dump(embedding_matrix,  open(file_name + '_embedding.p', "wb"))
				

				








def main():
	prepare_labels()
	prepare_sequences()

if __name__ == '__main__':
		main()	