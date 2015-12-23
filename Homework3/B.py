import A
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import nltk
from sklearn.feature_extraction import DictVectorizer
from nltk.data import load
from collections import Counter
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet as wn
import math
import itertools
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn import neighbors

# You might change the window size
window_size = 15
_POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'

# B.1.a,b,c,d
def extract_features(data, language):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
			[(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :return: features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }
    '''
    features = {}
    labels = {}
    Nc = {}
    Nsc = {}
    Nsc_bar = {}
    context = {}
    senses = set()
    tagger = load(_POS_TAGGER)
    collocation_window = 2
    punctuation = [',','!',';',')','(','?','-','_','\'']
    stop_words = stopwords.words('english')
    stemmer = SnowballStemmer("english", ignore_stopwords = True)

    # implement your code here

    for instance in data:
	tokens = []
	instance_features = {}
	instance_id = instance[0]
	head = instance[2]
	sense_id = instance[4]
  
	# tokenize left and right context of each lexelt
        left_context = nltk.word_tokenize(instance[1])
        right_context = nltk.word_tokenize(instance[3])

        # remove punctuation + stop words
        left_context = filter(lambda a: a.isalpha(), left_context) + [head]
        right_context = filter(lambda a: a.isalpha(), right_context)


	# stem words
	'''
	for word in left_context:
		word = stemmer.stem(word)
	for word in right_context:
		word = stemmer.stem(word)
        '''
        # fetch surrounding words
        tokens = left_context[-(collocation_window):] + right_context[:(collocation_window)]
	tagged = tagger.tag(tokens)

	# add collocational features (B.a)
	w = -1 * (collocation_window - 1)
	for item in tagged:
		key_w = 'w_' + str(w)
		key_pos = 'pos_' + str(w)
		value_w = item[0]
		value_pos = item[1]

		instance_features[key_w] = value_w
		instance_features[key_pos] = value_pos
		w += 1

        # fetch words in window_size
        left_context = left_context[-(window_size):]
        right_context = right_context[:(window_size)]

	union = left_context + right_context
        word_count = Counter(union)
	#bigrams = ngrams(union, 2)

	# fetch first noun and verb before and after head
	'''
        left_tags = tagger.tag(left_context)
        right_tags = tagger.tag(right_context)

	before_v, after_v = get_first(left_tags, right_tags, 'verb')
	before_n, after_n = get_first(left_tags, right_tags, 'noun')

	if before_v:
		instance_features['before_v'] = before_v
	if after_v:
		instance_features['after_v'] = after_v
	if before_n:
		instance_features['before_n'] = before_n
	if after_n:
		instance_features['after_n'] = after_n
	'''

	for word in word_count:
		sc = (sense_id, word)
		instance_features[word] = word_count[word]
		'''
		if word not in Nc:
			Nc[word] = 1
		else:
			Nc[word] += 1

		if sc not in Nsc:
			Nsc[sc] = 1
		else:
			Nsc[sc] += 1
	
	if sense_id not in context:
		context[sense_id] = union
	else:
		context[sense_id].extend(union)

	i = 0
	for bigram in bigrams:
		val = bigram[0] + ' ' + bigram[1]
		key = 'bigram_' + str(i)
		instance_features[key] = val
		i += 1
	'''
	# add hypernyms, synonyms, hyponyms	
	if language == 'English':
		target = instance_id.split('.')
		synset_ip = str(target[0] + '.' + target[1] + '.' + '01')
		synonyms = wn.synset(synset_ip)
		hypernyms = synonyms.hypernyms()
		hyponyms = synonyms.hyponyms()
	
		synonyms = synonyms.name().split('.')
		instance_features['syn'] = synonyms[0]
	'''
	cnt = 0
	for hypernym in hypernyms:
		key_hyper = 'hyper_' + str(cnt)
		hyper = hypernym.name().split('.')
		instance_features[key_hyper] = hyper[0]
		cnt += 1
	
	cnt = 0
	for hyponym in hyponyms:
		key_hypo = 'hypo_' + str(cnt)
		hypo = hyponym.name().split('.')
		instance_features[key_hypo] = hypo[0]	
		cnt += 1
	'''
	# add it all to the feature list
	features[instance_id] = instance_features
	labels[instance_id] = sense_id

    # compute relevance score
    '''   
    Nsc_bar = getNsc_bar(Nsc, Nc)

    for instance in data:
	instance_features = []
	instance_id = instance[0]
	existing_features = features[instance_id]
	sense_id = instance[4]
	words = context[sense_id]
	relevance_scores = {}
	for word in words:
		prob_num = Nsc[(sense_id, word)] + 1
		prob_den = Nsc_bar[(sense_id, word)] + 1
		score = math.log(prob_num, 2) - math.log(prob_den, 2)
		relevance_scores[word] = score
	sorted_scores = sorted(relevance_scores.items(), key = lambda x: x[1])
	top_scores = sorted_scores[-5:]

	#print "Sorted"
	#print sorted_scores
	#print "top"
	#print top_scores

	cnt = 0
	for item in top_scores:
		key = 'top_' + str(cnt)
		existing_features.update({key: item[0]})
		cnt += 1

	features[instance_id] = existing_features
    '''
    return features, labels

# implemented for you
def vectorize(train_features,test_features):
    '''
    convert set of features to vector representation
    :param train_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :param test_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :return: X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
            X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    '''
    X_train = {}
    X_test = {}

    vec = DictVectorizer()
    vec.fit(train_features.values())
    for instance_id in train_features:
        X_train[instance_id] = vec.transform(train_features[instance_id]).toarray()[0]

    for instance_id in test_features:
        X_test[instance_id] = vec.transform(test_features[instance_id]).toarray()[0]

    return X_train, X_test

#B.1.e
def feature_selection(X_train,X_test,y_train):
    '''
    Try to select best features using good feature selection methods (chi-square or PMI)
    or simply you can return train, test if you want to select all features
    :param X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }
    :return:
    '''
    # implement your code here
    #return chi_square(X_train, y_train, X_test) 

    # or return all feature (no feature selection):
    return X_train, X_test

# B.2
def classify(X_train, X_test, y_train):
    '''
    Train the best classifier on (X_train, and y_train) then predict X_test labels

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

    :return: results: a list of tuples (instance_id, label) where labels are predicted by the best classifier
    '''
    svm_results = []

    # implement your code here
 
    svm_clf = svm.LinearSVC()

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

    labels_svm = svm_clf.predict(T)

    i = 0
    for instance in X_test:
        result_svm = (instance, labels_svm[i])
        svm_results.append(result_svm)
        i += 1
   
    return svm_results

# run part B
def run(train, test, language, answer):
    results = {}

    for lexelt in train:
        train_features, y_train = extract_features(train[lexelt], language)
        test_features, _ = extract_features(test[lexelt], language)

        X_train, X_test = vectorize(train_features,test_features)
        X_train_new, X_test_new = feature_selection(X_train, X_test,y_train)
        results[lexelt] = classify(X_train_new, X_test_new,y_train)
	#break

    A.print_results(results, answer)

def get_first(left_tags, right_tags, pos):
    tagger = load(_POS_TAGGER)
    verb_tags = ['VB','VBD','VBZ','VBN','VBG','VBP']
    noun_tags = ['NN','NNS','NNP','NNPS']
    before = None
    after = None

    if pos == 'verb':
	target_tags = verb_tags
    else:
	target_tags = noun_tags

    for item in reversed(left_tags):
	tag = item[1]
	if tag in target_tags:
		before = item[0]
		break

    for item in right_tags:
	tag = item[1]
	if tag in target_tags:
		after = item[0]
		break

    return before, after

#B.1.e
def chi_square(X_train, y_train, X_test):
    X = []
    Y = []
    X_train_new = {}
    X_test_new = {}
    '''
    for instance in X_train:
        y_val = y_train[instance]
        x_val = X_train[instance]

        X.append(x_val)
        Y.append(y_val)

    T = []
    for instance in X_test:
        item = X_test[instance]
        T.append(item)

    f = neighbors.KNeighborsClassifier()
_fit = SelectKBest(chi2, k = 10).fit(X, Y)
    train_new = X_fit.transform(X)
    test_new = X_fit.transform(T)

    i = 0
    for instance in X_train:
	X_train_new[instance] = train_new[i]
	i += 1

    i = 0
    for instance in X_test:
	X_test_new[instance] = test_new[i]
	i += 1
    '''

    for instance in X_train:
	X = []
	Y = []
	T = []
	
	y_val = y_train[instance]
	x_val = X_train[instance]

	if instance in X_test:
		t_val = X_test[instance]

		X.append(x_val)
		Y.append(y_val)
		T.append(t_val)

		X_fit = SelectBest(chi2, k = 10).fit(X,Y)
		train_new = X_fit.transform(X)
		test_new = X_fit.transform(T)

		X_train_new[instance] = train_new
		X_test_new[instance] = test_new

    return X_train_new, X_test_new

def getNsc_bar(Nsc, Nc):
   Nsc_bar = {}
   for item in Nsc:
	Nsc_val = Nsc[item]
	Nsc_bar_val = Nc[item[1]] - Nsc_val
	Nsc_bar[item] = Nsc_bar_val

   return Nsc_bar
