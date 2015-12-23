from nltk.corpus import comtrans
import A
import B
#import EC
import time

if __name__ == '__main__':
    start_time = time.time()
    aligned_sents = comtrans.aligned_sents()[:350]
    A.main(aligned_sents)
    B.main(aligned_sents)
    #EC.main(aligned_sents)
    print("--- %s seconds ---" % (time.time() - start_time))

