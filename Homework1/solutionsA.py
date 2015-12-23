import math
import nltk
import time

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):
    unigram_p = {}
    bigram_p = {}
    trigram_p = {}
    uni_words_dict = {}
    bi_words_dict = {}
    sentence_count = 0

    # --------------------------------------------------------------------------------
    # Calculate log probabilities for unigrams
    for sentence in training_corpus:
	sentence_count += 1
	sentence = START_SYMBOL + " " + sentence + " " + STOP_SYMBOL
	words = sentence.split()
	for word in words:
		if word != START_SYMBOL:
			if word in unigram_p:
				unigram_p[word] += 1
			else:
				unigram_p[word] = 1

    # maintain a unigram word dictionary
    uni_words_dict = unigram_p.copy()
    uni_words_dict["*"] = sentence_count

    # Get total word count
    word_count = sum(unigram_p.values())

    logP_word = math.log(word_count, 2)

    # update unigram_p to hold log probabilities of each word
    unigram_p.update((x, math.log(y, 2) - logP_word) for x, y in unigram_p.items())

    # --------------------------------------------------------------------------------
    # Calculate log probabilities for bigrams
    for sentence in training_corpus:
	sentence = START_SYMBOL + " " + sentence + " " + STOP_SYMBOL
	words = sentence.split()
	bigram_list = list(nltk.bigrams(words))
	for item in bigram_list:
		if item in bigram_p:
			bigram_p[item] += 1
		else:
			bigram_p[item] = 1

    # maintain a bigram word dictionary
    bi_words_dict = bigram_p.copy()
    bi_words_dict[('*','*')] = sentence_count

    # update bigram_p to hold log probabilities of each tuple/word pair
    bigram_p.update((x, math.log(y, 2) - math.log(uni_words_dict[x[0]], 2)) for x, y in bigram_p.items())

    # --------------------------------------------------------------------------------
    # Calculate log probabilities for trigrams
    for sentence in training_corpus:
	sentence = START_SYMBOL +" " + START_SYMBOL + " " + sentence + " " + STOP_SYMBOL
	words = sentence.split()
	trigram_list = list(nltk.trigrams(words))
	for item in trigram_list:
		if item in trigram_p:
			trigram_p[item] += 1
		else:
			trigram_p[item] = 1

    # update trigram_p to hold log probabilities of each tuple/word pair

    for item in trigram_p:
	bi = []
 	bi.append(item[0])
	bi.append(item[1])
	freq1 = trigram_p[item]
	freq2 = bi_words_dict[tuple(bi)]
	trigram_p[item] = math.log(freq1, 2) - math.log(freq2, 2)

    return unigram_p, bigram_p, trigram_p

# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()    
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc. 
def score(ngram_p, n, corpus):
    scores = []
    two_prob = math.log(2)
   
    # --------------------------------------------------------------------------------------
    # Calculate score for unigrams
    if n == 1: 
    	for sentence in corpus:
		score = 1
		sentence = sentence + " " + STOP_SYMBOL
		words = sentence.split()

		for word in words:
			log_prob = ngram_p[word]
			probability = math.exp(log_prob * two_prob)
			score = score * probability
	
		if score > 0:
			result = math.log(score, 2)
		else:
			result = MINUS_INFINITY_SENTENCE_LOG_PROB

		# append final computed score
		scores.append(result)

    # --------------------------------------------------------------------------------------
    # Calculate score for bigrams

    if n == 2:
	for sentence in corpus:
		score = 1
		sentence = START_SYMBOL + " " + sentence + " " + STOP_SYMBOL
		words = sentence.split()
		bigram_list = list(nltk.bigrams(words))

		for item in bigram_list:
			log_prob = ngram_p[item]
			probability = math.exp(log_prob * two_prob)
			score = score * probability

		if score > 0:
			result = math.log(score, 2)
		else:
			result = MINUS_INFINITY_SENTENCE_LOG_PROB

		# append final computed score
		scores.append(result)

    # --------------------------------------------------------------------------------------
    # Calculate score for trigrams
    if n == 3:
	for sentence in corpus:
		score = 1
		sentence = START_SYMBOL + " " + START_SYMBOL + " " + sentence + " " + STOP_SYMBOL
		words = sentence.split()
		trigram_list = list(nltk.trigrams(words))

		for item in trigram_list:
			log_prob = ngram_p[item]
			probability = math.exp(log_prob * two_prob)
			score = score * probability

		if score > 0:
			result = math.log(score, 2)
		else:
			result = MINUS_INFINITY_SENTENCE_LOG_PROB

		# append final computed score
		scores.append(result)

    return scores

# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, corpus):
    scores = []
    two_prob = math.log(2)
    lambda_val = math.log(1,2) - math.log(3,2)
   
    for sentence in corpus:
	sentence = START_SYMBOL + " " + START_SYMBOL + " " + sentence + " " + STOP_SYMBOL
	words = sentence.split()
	trigram_list = list(nltk.trigrams(words))

	score = 0
	for item in trigram_list:
		bi = []
		bi.append(item[1])
		bi.append(item[2])

		uni = item[2]
		if item in trigrams:
			log_prob_tri = trigrams[item]
			probability_tri = math.exp(log_prob_tri * two_prob)

			log_prob_bi = bigrams[tuple(bi)]
			probability_bi = math.exp(log_prob_bi * two_prob)

			log_prob_uni = unigrams[uni]
			probability_uni = math.exp(log_prob_uni * two_prob)

			score = score + (lambda_val + math.log((probability_tri + probability_bi + probability_uni), 2))

		elif tuple(bi) in bigrams:
			log_prob_bi = bigrams[tuple(bi)]
			probability_bi = math.exp(log_prob_bi * two_prob)

			log_prob_uni = unigrams[uni]
			probability_uni = math.exp(log_prob_uni * two_prob)

			score = score + (lambda_val + math.log((probability_bi + probability_uni), 2))

		elif uni in unigrams:
			log_prob_uni = unigrams[uni]
			probability_uni = math.exp(log_prob_uni * two_prob)

			score = score + (lambda_val + math.log(probability_uni, 2))
	
		else:
			score = MINUS_INFINITY_SENTENCE_LOG_PROB

	scores.append(score)

    return scores


DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close() 

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print "Part A time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
