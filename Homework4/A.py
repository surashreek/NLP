import nltk
from nltk.align import IBMModel1
from nltk.align import IBMModel2
from nltk.corpus import comtrans

# TODO: Initialize IBM Model 1 and return the model.
def create_ibm1(aligned_sents):
	num_iters = 10
	ibm1 = IBMModel1(aligned_sents, num_iters)
	return ibm1


# TODO: Initialize IBM Model 2 and return the model.
def create_ibm2(aligned_sents):
	num_iters = 10
	ibm2 = IBMModel2(aligned_sents, num_iters)
	return ibm2

# TODO: Compute the average AER for the first n sentences
#       in aligned_sents using model. Return the average AER.
def compute_avg_aer(aligned_sents, model, n):
	total_aer = 0
	for index, aligned_sent in enumerate(aligned_sents[:50]):
		my_als = model.align(aligned_sent)
		gold_als = comtrans.aligned_sents()[:350][index]
		aer = gold_als.alignment_error_rate(my_als)
		total_aer += aer

	avg_aer = float(total_aer) / float(n)
	return avg_aer

# TODO: Computes the alignments for the first 20 sentences in
#       aligned_sents and saves the sentences and their alignments
#       to file_name. Use the format specified in the assignment.
def save_model_output(aligned_sents, model, file_name):
	f = open(file_name, 'w')
	for aligned_sent in aligned_sents[:20]:
		my_als = model.align(aligned_sent)
		source = my_als.words
		target = my_als.mots
		alignments = my_als.alignment

		f.write(str(source) + "\n")
		f.write(str(target))
		f.write("\n")
		f.write(str(alignments))
		f.write("\n")
		f.write("\n\n")
		#break

	f.close()

def compareModels(aligned_sents, model1, model2):
    for i in range(0,10):
        rst1 = model1.align(aligned_sents[i])
        rst2 = model2.align(aligned_sents[i])
        AER1 = rst1.alignment_error_rate(aligned_sents[i])
        AER2 = rst2.alignment_error_rate(aligned_sents[i])

	# Compare AER
        if(AER1 < AER2):
            print("Model1 better")
        elif (AER1 > AER2):
            print("Model2 better")

        print("Gold: ");
	print(aligned_sents[i].words)
	print(aligned_sents[i].mots)
        print(aligned_sents[i].alignment)
        print("Model1: "+ str(AER1))
        print(rst1.alignment)
        print("Model2: "+ str(AER2))
        print(rst2.alignment)
	print "\n"

def main(aligned_sents):

    	ibm1 = create_ibm1(aligned_sents)
    	save_model_output(aligned_sents, ibm1, "ibm1.txt")
    	avg_aer = compute_avg_aer(aligned_sents, ibm1, 50)

    	print ('IBM Model 1')
    	print ('---------------------------')
    	print('Average AER: {0:.3f}\n'.format(avg_aer))

    	ibm2 = create_ibm2(aligned_sents)
    	save_model_output(aligned_sents, ibm2, "ibm2.txt")
    	avg_aer = compute_avg_aer(aligned_sents, ibm2, 50)
    
    	print ('IBM Model 2')
    	print ('---------------------------')
    	print('Average AER: {0:.3f}\n'.format(avg_aer))

	# Uncomment this to check comparison between 2 models
	#compare(aligned_sents, ibm1, ibm2)
