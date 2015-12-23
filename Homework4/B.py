import nltk
import math
import A
from nltk.align import Alignment, AlignedSent
from nltk.align import IBMModel1
from collections import defaultdict

class BerkeleyAligner():

    def __init__(self, align_sents, num_iter):
        self.t, self.q = self.train(align_sents, num_iter)

    # TODO: Computes the alignments for align_sent, using this model's parameters. Return
    # an AlignedSent object, with the sentence pair and the alignments computed.
    def align(self, align_sent):
	alignment = []

        l = len(align_sent.words)
        m = len(align_sent.mots)

        for j, ej in enumerate(align_sent.words):
            
            # Initialize maximum probability as None
            max_align_prob = (self.t[ej][None]*self.q[0][j+1][l][m], None)

            for i, fi in enumerate(align_sent.mots):
                # Calculate maximum probability
		prod = (self.t[ej][fi] * self.q[i+1][j+1][l][m], i)
		max_align_prob = max(max_align_prob, prod)

            # If max probability is not None, append it to alignments list
            if max_align_prob[1] is not None:
                alignment.append((j, max_align_prob[1]))

        return AlignedSent(align_sent.words, align_sent.mots, alignment)
    
    # TODO: Implement the EM algorithm. num_iters is the number of iterations. Returns the 
    # translation and distortion parameters as a tuple.
    def train(self, aligned_sents, num_iters):
	fr_vocab = set()
        en_vocab = set()

        for alignSent in aligned_sents:
            en_vocab.update(alignSent.words)
            fr_vocab.update(alignSent.mots)

	# invert aligned sentences
        aligned_sents_inv = [e.invert() for e in aligned_sents]

	# initialise parameters t and q
        t_ef, align = self.initParameters(aligned_sents)
        t_ef_inv, align_inv = self.initParameters(aligned_sents_inv)

	# Main Expectation-Maximization logic
        for i in range(0, num_iters):
            # EM for forward aligned sentence
            fr_vocab.add(None)
            t_ef, align = self.performEM(t_ef, align, en_vocab, fr_vocab, aligned_sents)
            fr_vocab.remove(None)

            # EM for backward aligned sentence
            en_vocab.add(None)
            t_ef_inv, align_inv = self.performEM(t_ef_inv, align_inv, fr_vocab, en_vocab, aligned_sents_inv)
            en_vocab.remove(None)

	# Berkeley Aligner step: average forward + backward
	# Average t for forward and backward
	t_ef_new = defaultdict(lambda: defaultdict(lambda: 0.0))
        t_ef_inv_new = defaultdict(lambda: defaultdict(lambda: 0.0))
        totalfi = defaultdict(lambda: 0.0)
        for e in t_ef:
            for f in t_ef[e]:
                t_ef_new[e][f] = (t_ef[e][f] + t_ef_inv[f][e]) / 2.0
                totalfi[f] += t_ef_new[e][f]
        for e in t_ef:
            for f in t_ef[e]:
                t_ef_new[e][f] = t_ef_new[e][f] / totalfi[f]

	# Average q for forward and backward
        align_new = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0))))
        total_align = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0)))
        for f_i in align:
            for e_i in align[f_i]:
                for l in align[f_i][e_i]:
                    for m in align[f_i][e_i][l]:
                        align_new[f_i][e_i][l][m] = (align[f_i][e_i][l][m] + align_inv[e_i][f_i][m][l]) / 2.0
                        total_align[e_i][l][m] += align_new[f_i][e_i][l][m]
        for f_i in align:
            for e_i in align[f_i]:
                for l in align[f_i][e_i]:
                    for m in align[f_i][e_i][l]:
                        align_new[f_i][e_i][l][m] /= total_align[e_i][l][m]

	# return final t and q
        return t_ef_new, align_new

    def initParameters(self, align_sents):
	# Compute t probabilities from IBMModel1
	num_iters = 15
        ibm1 = IBMModel1(align_sents, num_iters)
        t_ef = ibm1.probabilities

        align = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0))))

        # Initialise alignment probability with uniform distribution
        for alignSent in align_sents:
            en_set = alignSent.words
            fr_set = [None] + alignSent.mots
            m = len(fr_set) - 1
            l = len(en_set)
            for i in range(0, m + 1):
                for j in range(1, l + 1):
                    align[i][j][l][m] = 1.0 / (m + 1)

        return t_ef, align

    def performEM(self, t_ef, align, en_vocab, fr_vocab, align_sents):
        count_ef = defaultdict(lambda: defaultdict(float))
        totalfi = defaultdict(float)

        count_align = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0))))
        total_align = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0)))

        totalej = defaultdict(float)

        for alignSent in align_sents:
            en_set = alignSent.words
            fr_set = [None] + alignSent.mots
            m = len(fr_set) - 1
            l = len(en_set)

            # Do normalization step
            for j in range(1, l + 1):
                ej = en_set[j - 1]
                totalej[ej] = 0
                for i in range(0, m + 1):
                    totalej[ej] += t_ef[ej][fr_set[i]] * align[i][j][l][m]

            # Collect all counts
            for j in range(1, l + 1):
                ej = en_set[j - 1]
                for i in range(0, m + 1):
                    fi = fr_set[i]
                    delta = (t_ef[ej][fi] * align[i][j][l][m]) / totalej[ej]
                    count_ef[ej][fi] += delta
                    totalfi[fi] += delta
                    count_align[i][j][l][m] += delta
                    total_align[j][l][m] += delta

        # Compute t and q
        t_ef = defaultdict(lambda: defaultdict(lambda: 0.0))
        align = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0))))
       
	# Compute new t
        for f in fr_vocab:
            for e in en_vocab:
                t_ef[e][f] = count_ef[e][f] / totalfi[f]

        # Compute new q
        for alignSent in align_sents:
            en_set = alignSent.words
            fr_set = [None] + alignSent.mots
            m = len(fr_set) - 1
            l = len(en_set)
            for i in range(0, m + 1):
                for j in range(1, l + 1):
                    align[i][j][l][m] = count_align[i][j][l][m] / total_align[j][l][m]

        return t_ef, align


def main(aligned_sents):
    ba = BerkeleyAligner(aligned_sents, 10)
    A.save_model_output(aligned_sents, ba, "ba.txt")
    avg_aer = A.compute_avg_aer(aligned_sents, ba, 50)

    print ('Berkeley Aligner')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))

