import sys
import nltk
import math
import time
import pdb

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000


# TODO: IMPLEMENT THIS FUNCTION
# Receives a list of tagged sentences and processes each sentence to generate a list of words and a list of tags.
# Each sentence is a string of space separated "WORD/TAG" tokens, with a newline character in the end.
# Remember to include start and stop symbols in yout returned lists, as defined by the constants START_SYMBOL and STOP_SYMBOL.
# brown_words (the list of words) should be a list where every element is a list of the tags of a particular sentence.
# brown_tags (the list of tags) should be a list where every element is a list of the tags of a particular sentence.
def split_wordtags(brown_train):
    brown_words = []
    brown_tags = []
    prepend_tag = START_SYMBOL + "/" + START_SYMBOL
    append_tag = STOP_SYMBOL + "/" + STOP_SYMBOL
    for sentence in brown_train:
	sentence = prepend_tag + " " + prepend_tag + " " + sentence + " " + append_tag
	words = sentence.split()
	
	for word in words:
		k = word.rfind("/")
		brown_words.append(word[0:k])

		brown_tags.append(word[k+1:])

    return brown_words, brown_tags


# TODO: IMPLEMENT THIS FUNCTION
# This function takes tags from the training data and calculates tag trigram probabilities.
# It returns a python dictionary where the keys are tuples that represent the tag trigram, and the values are the log probability of that trigram
def calc_trigrams(brown_tags):
    q_values = {}
    unigram_p = {}
    bigram_p = {}
    trigram_p = {}

    for word in brown_tags:
    	if word != START_SYMBOL:
        	if word in unigram_p:
                	unigram_p[word] += 1
                else:
                        unigram_p[word] = 1

    unigram_p[START_SYMBOL] = len(brown_tags)
    # maintain a unigram word dictionary
    uni_words_dict = unigram_p.copy()

    # Get total word count
    word_count = sum(unigram_p.values())

    logP_word = math.log(word_count, 2)

    # update unigram_p to hold log probabilities of each word
    unigram_p.update((x, math.log(y, 2) - logP_word) for x, y in unigram_p.items())

    # --------------------------------------------------------------------------------
    # Calculate log probabilities for bigrams

    bigram_list = list(nltk.bigrams(brown_tags))
    for item in bigram_list:
    	if item in bigram_p:
        	bigram_p[item] += 1
        else:
                bigram_p[item] = 1

    # maintain a bigram word dictionary
    #bigram_p[(START_SYMBOL, START_SYMBOL)] = len(brown_tags)
    bi_words_dict = bigram_p.copy()

    # update bigram_p to hold log probabilities of each tuple/word pair
    bigram_p.update((x, math.log(y, 2) - math.log(uni_words_dict[x[0]], 2)) for x, y in bigram_p.items())

    # --------------------------------------------------------------------------------
    # Calculate log probabilities for trigrams
    trigram_list = list(nltk.trigrams(brown_tags))
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

    q_values = trigram_p.copy()    
    return q_values

# This function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(q_values, filename):
    outfile = open(filename, "w")
    trigrams = q_values.keys()
    trigrams.sort()  
    for trigram in trigrams:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and returns a set of all of the words that occur more than 5 times (use RARE_WORD_MAX_FREQ)
# brown_words is a python list where every element is a python list of the words of a particular sentence.
# Note: words that appear exactly 5 times should be considered rare!
def calc_known(brown_words):
    brown_dict = {}
    known_words = set([])

    for item in brown_words:
	if item in brown_dict:
		brown_dict[item] += 1
	else:
		brown_dict[item] = 1

    for item in brown_dict:
	if brown_dict[item] > RARE_WORD_MAX_FREQ:
		known_words.add(item)

    return known_words

# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and a set of words that should not be replaced for '_RARE_'
# Returns the equivalent to brown_words but replacing the unknown words by '_RARE_' (use RARE_SYMBOL constant)
def replace_rare(brown_words, known_words):
    brown_words_rare = []

    sentence = []
    for word in brown_words:
	if word == "STOP":
		#sentence = sentence + " " + word
		sentence.append(word)
		brown_words_rare.append(sentence)
		sentence = []
	elif word in known_words:
		#sentence = sentence + " " + word
		sentence.append(word)
	else:
		#sentence = sentence + " " + RARE_SYMBOL
		sentence.append(RARE_SYMBOL)

    return brown_words_rare

# This function takes the ouput from replace_rare and outputs it to a file
def q3_output(rare, filename):
    outfile = open(filename, 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates emission probabilities and creates a set of all possible tags
# The first return value is a python dictionary where each key is a tuple in which the first element is a word
# and the second is a tag, and the value is the log probability of the emission of the word given the tag
# The second return value is a set of all possible tags for this data set
def calc_emission(brown_words_rare, brown_tags):
    e_values = {}
    taglist = set([])
    brown_words_list = []
    tags_dict = {}

    # Compute a frequency dictionary for tags
    for tag in brown_tags:
	taglist.add(tag)
	if tag in tags_dict:
		tags_dict[tag] += 1
	else:
		tags_dict[tag] = 1

    # Split words from sentences in brown_words_rare and append to a list
    for sentence in brown_words_rare:
	#words = sentence.split()
	words = sentence
	for word in words:
		brown_words_list.append(word)

    word_count = len(brown_words_list) #same as length of brown_tags

    i = 0
    while i < word_count:
 	temp = []
	temp.append(brown_words_list[i])
	temp.append(brown_tags[i])
	if tuple(temp) in e_values:
		e_values[tuple(temp)] += 1
	else:
		e_values[tuple(temp)] = 1
	i += 1

    # update e_values to hold log probabilities of each word/tag tuple
    e_values.update((x, math.log(y, 2) - math.log(tags_dict[x[1]], 2)) for x, y in e_values.items())

    return e_values, taglist

# This function takes the output from calc_emissions() and outputs it
def q4_output(e_values, filename):
    outfile = open(filename, "w")
    emissions = e_values.keys()
    emissions.sort()  
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# This function takes data to tag (brown_dev_words), a set of all possible tags (taglist), a set of all known words (known_words),
# trigram probabilities (q_values) and emission probabilities (e_values) and outputs a list where every element is a tagged sentence 
# (in the WORD/TAG format, separated by spaces and with a newline in the end, just like our input tagged data)
# brown_dev_words is a python list where every element is a python list of the words of a particular sentence.
# taglist is a set of all possible tags
# known_words is a set of all known words
# q_values is from the return of calc_trigrams()
# e_values is from the return of calc_emissions()
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. Remember also that the output should not contain the "_RARE_" symbol, but rather the
# original words of the sentence!
def viterbi(brown_dev_words, taglist, known_words, q_values, e_values):
    tagged = []
    two_prob = math.log(2)
   
    for to_tag in brown_dev_words:
	sentence = to_tag[:]
	original = to_tag[:]
 	n = len(sentence)
	y = [None] * n

 	# replace rare words with _RARE_
    	for x in range(0, len(sentence)):
		if sentence[x] not in known_words:
			sentence[x] = RARE_SYMBOL
   
    	S = []
    	S0 = ['*']
    	S.append(S0)	# S[-1] = S[0] = {*}
    	S.append(S0)

    	for i in range(1, n + 1):	# Sk = S for k from {1 .. n}
		S.append(taglist)
    
    	trellis = {}	# viterbi dictionary of key tuple(k, u, v) and value = max probability of tag sequence
    	backptr = {}	# backpointer matrix

    	# initialization step
    	init = (0, '*', '*')
    	trellis[init] = 0	#log probability of 1

    	y = [None] * n

    	prev_tag = '*'
    	# Recurse- Forward pass
    	for k in range(2, n + 1, 1):
		for current in S[k]:
			max_trellis = None
			max_tag = None
			for u in S[k - 2]:
				for v in S[k - 1]:
					tuple1 = (k - 2, u, v)
					if tuple1 in trellis:
						prob_trellis = trellis[tuple1]
					else:
						continue

					tuple2 = (u, v, current)
					if tuple2 in q_values:
						prob_q = q_values[tuple2]
					else:
						prob_q = 0

					tuple3 = (sentence[k - 2], current)
					if tuple3 in e_values:
						prob_e = e_values[tuple3]
					else:
						continue

					fin_prod = prob_trellis + prob_q + prob_e
					if fin_prod > max_trellis:
						max_trellis = fin_prod
						max_tag = v
			tuple4 = (k - 1, prev_tag, current)
			if max_trellis != None:
				trellis[tuple4] = max_trellis
				backptr[tuple4] = max_tag
				y[k - 1] = max_tag
				prev_tag = max_tag
   
    	# find last two tags
    	last_one = None
    	last = None
    	max_trellis = None
    	for u in taglist:
		for v in taglist:
			tuple5 = (n - 1, u, v)
			tuple6 = (u, v, '.')
	
			if tuple5 in trellis:
				prod1 = trellis[tuple5]
			else:
				continue

			if tuple6 in q_values:
				prod2 = q_values[tuple6]
			else:
				prod2 = 0
	
			product = prod1 + prod2
			if product > max_trellis:
				max_trellis = product
				last_one = u
				last = v

    	y.append(last) 
        y.append('.') 

	# put it all together
	tagged_sentence = ""
 	for index in range(0, len(original)):
		tagged_sentence += original[index] + "/" + y[index + 2] + " "
	tagged_sentence += "\n"
	tagged.append(tagged_sentence)
 
    return tagged

# This function takes the output of viterbi() and outputs it to file
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# This function uses nltk to create the taggers described in question 6
# brown_words and brown_tags is the data to be used in training
# brown_dev_words is the data that should be tagged
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. 
def nltk_tagger(brown_words, brown_tags, brown_dev_words):
    '''
    # Stripping START and STOP symbols from words & tags does not help
    updated_tags = []
    updated_words = []
    for tag in brown_tags:
	if tag == START_SYMBOL or tag == STOP_SYMBOL:
		continue
	else:
		updated_tags.append(tag)

    for word in brown_words:
	if word == START_SYMBOL or word == STOP_SYMBOL:
		continue
	else:
		updated_words.append(word)
    '''
    # Hint: use the following line to format data to what NLTK expects for training
    training = [ zip(brown_words[i], brown_tags[i]) for i in xrange(len(brown_words)) ]

    # IMPLEMENT THE REST OF THE FUNCTION HERE
    default_tagger = nltk.DefaultTagger("NOUN")
    bigram_tagger = nltk.BigramTagger(training, backoff = default_tagger)
    trigram_tagger = nltk.TrigramTagger(training, backoff = bigram_tagger)

    tagged = []
    for sentence in brown_dev_words:
	tokens = sentence
	tagged_sentence = trigram_tagger.tag(tokens)    	
	new_sentence = ""
	for item in tagged_sentence:
		new_sentence += item[0] + "/" + item[1] + " "
	new_sentence += "\n"
	tagged.append(new_sentence)	

    return tagged

# This function takes the output of nltk_tagger() and outputs it to file
def q6_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

def main():
    # start timer
    time.clock()

    # open Brown training data
    infile = open(DATA_PATH + "Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    # split words and tags, and add start and stop symbols (question 1)
    brown_words, brown_tags = split_wordtags(brown_train)

    # calculate tag trigram probabilities (question 2)
    q_values = calc_trigrams(brown_tags)

    # question 2 output
    q2_output(q_values, OUTPUT_PATH + 'B2.txt')

    # calculate list of words with count > 5 (question 3)
    known_words = calc_known(brown_words)

    # get a version of brown_words with rare words replace with '_RARE_' (question 3)
    brown_words_rare = replace_rare(brown_words, known_words)

    # question 3 output
    q3_output(brown_words_rare, OUTPUT_PATH + "B3.txt")

    # calculate emission probabilities (question 4)
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)

    # question 4 output
    q4_output(e_values, OUTPUT_PATH + "B4.txt")

    # delete unneceessary data
    del brown_train
    del brown_words_rare

    # open Brown development data (question 5)
    infile = open(DATA_PATH + "Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    # format Brown development data here
    brown_dev_words = []
    for sentence in brown_dev:
        brown_dev_words.append(sentence.split(" ")[:-1])

    # do viterbi on brown_dev_words (question 5)
    viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)

    # question 5 output
    q5_output(viterbi_tagged, OUTPUT_PATH + 'B5.txt')

    # do nltk tagging here
    nltk_tagged = nltk_tagger(brown_words, brown_tags, brown_dev_words)

    # question 6 output
    q6_output(nltk_tagged, OUTPUT_PATH + 'B6.txt')

    # print total time to run Part B
    print "Part B time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
