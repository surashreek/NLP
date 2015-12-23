from main import replace_accented
from sklearn import svm
from sklearn import neighbors
from sets import Set
from itertools import groupby
import nltk
from collections import Counter
import sys

# don't change the window size
window_size = 10

# A.1
def build_s(data):
    '''
    Compute the context vector for each lexelt
    :param data: dic with the following structure:
        {
			lexelt: [(instance_id, left_context, head, right_context, sense_id), ...],
			...
        }
    :return: dic s with the following structure:
        {
			lexelt: [w1,w2,w3, ...],
			...
        }

    '''
    s = {}

    # implement your code here

    punctuation = [',', ';', '!', ')', '(']
    for lexelt in data:
	for instance in data[lexelt]:
		# tokenize left and right context of each lexelt
		left_context = nltk.word_tokenize(instance[1])
		right_context = nltk.word_tokenize(instance[3])
		head = instance[2]

		# remove punctuation
		left_context = filter(lambda a: a not in punctuation, left_context)
		right_context = filter(lambda a: a not in punctuation, right_context)

		# fetch words in window_size = 10
		left_context = left_context[-(window_size):] + [head]
		right_context = right_context[:(window_size)]

		# perform union of left and right contexts
		union = left_context + right_context

		# add lexelt and context in dictionary
		if lexelt in s:
			s[lexelt].extend(union)
		else:
			s[lexelt] = union
    return s


# A.1
def vectorize(data, s):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
			[(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :param s: list of words (features) for a given lexelt: [w1,w2,w3, ...]
    :return: vectors: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }

    '''
    vectors = {}
    labels = {}
    s_count = {}
    punctuation = [',', ';', '!', ')', '(']
    # implement your code here

    # count frequency of each word
    '''
    for word in s:
	if word in s_count:
		s_count[word] += 1
	else:
		s_count[word] = 1
    '''

    s = set(s)
    # compute vectors & labels
    for instance in data:
	occurrences = []
	instance_id = instance[0]
	sense_id = instance[4]
	
	# tokenize left and right context of each lexelt
	left_context = nltk.word_tokenize(instance[1])
	right_context = nltk.word_tokenize(instance[3])
	head = instance[2]

	# remove punctuation
	left_context = filter(lambda a: a not in punctuation, left_context)
	right_context = filter(lambda a: a not in punctuation, right_context)

	# fetch words in window_size = 10
	left_context = left_context[-(window_size):] + [head]
	right_context = right_context[:(window_size)]

	# perform union of left and right contexts
 	union = left_context + right_context

	word_count = Counter(union)
	for word in s:
		if word in union:
			occurrences.append(word_count[word])
		else:
			occurrences.append(0)

	vectors[instance_id] = occurrences	
	labels[instance_id] = sense_id

    return vectors, labels


# A.2
def classify(X_train, X_test, y_train):
    '''
    Train two classifiers on (X_train, and y_train) then predict X_test labels

    :param X_train: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param X_test: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }

    :return: svm_results: a list of tuples (instance_id, label) where labels are predicted by LinearSVC
             knn_results: a list of tuples (instance_id, label) where labels are predicted by KNeighborsClassifier
    '''

    svm_results = []
    knn_results = []

    svm_clf = svm.LinearSVC()
    knn_clf = neighbors.KNeighborsClassifier()

    # implement your code here

    X = []
    Y = []

    for instance in X_train:
	y_val = y_train[instance]
	x_val = X_train[instance]

	X.append(x_val)
	Y.append(y_val)
	
    T = []
    for instance in X_test:
	item = X_test[instance]
	T.append(item)
   
    svm_clf.fit(X, Y)
    knn_clf.fit(X, Y)

    labels_svm = svm_clf.predict(T)
    labels_knn = knn_clf.predict(T)

    i = 0
    for instance in X_test:
	result_svm = (instance, labels_svm[i])
	result_knn = (instance, labels_knn[i])
	svm_results.append(result_svm)
	knn_results.append(result_knn)
	i += 1

    return svm_results, knn_results

# A.3, A.4 output
def print_results(results, output_file):
    '''

    :param results: A dictionary with key = lexelt and value = a list of tuples (instance_id, label)
    :param output_file: file to write output

    '''

    # implement your code here
    # don't forget to remove the accent of characters using main.replace_accented(input_str)
    # you should sort results on instance_id before printing

    reload(sys)
    sys.setdefaultencoding('utf-8')
    output = open(output_file, 'w')

    ret = []
    for item in results:
	instance_list = results[item]
	for instance in instance_list:
		item = replace_accented(item)
		instance_id = replace_accented(instance[0])
		label = replace_accented(unicode(instance[1]))
		line = (item, instance_id, label)
		ret.append(line)

    sorted(ret, key = lambda x: x[1])

    for line in ret:
	output_str = line[0] + " " + line[1] + " " + line[2] + "\n"
	output.write(output_str)

    output.close()
# run part A
def run(train, test, language, knn_file, svm_file):
    s = build_s(train)
    svm_results = {}
    knn_results = {}

    i = 0
    for lexelt in s:
	#print "processing lexelt " + str(lexelt)
        X_train, y_train = vectorize(train[lexelt], s[lexelt])
        X_test, _ = vectorize(test[lexelt], s[lexelt])
        svm_results[lexelt], knn_results[lexelt] = classify(X_train, X_test, y_train)
	#if i > 2:
	#	break
	#i += 1

    print_results(svm_results, svm_file)
    print_results(knn_results, knn_file)



