COMS 4705 NLP Homework 2
(Surashree Kulkarni- ssk2197)

1) Dependency Graphs

a. Please find the dependency graph images for the English, Danish, Korean and Swedish training datasets.

b. How to determine if a dependency graph is projective:
A dependency graph is projective only if it has no arcs that intersect each other, i.e., it has no crossing dependency edges.

After looking at the _is_projective() method in transitionparser.py, it's easy to see that the algorithm checks that for every arc in the arc list:
   i) The child (address) node id is less than parent (head) node id
   ii) For all nodes BEFORE the child node and AFTER the parent node (say x), and for all nodes BETWEEN the child & parent nodes (say y), there are no arcs going from 		(x to y) or (y to x)
   iii) From ii, we know that if such an arc exists, it cuts the arc going from the child to the parent (in consideration), and hence the given sentence is non-projective

In short, one has to check that for every arc going from say node a to b node, there exists no arc between any node (token) before a/after b and a node between a and b.

c. 
English sentence with a projective dependency graph:
John gambled at the casino. 

English sentence with a non-projective dependency graph:
John will gamble at a casino tomorrow which is the hottest new place in Vegas.  
 
2) Performance of badfeatures.model

Output:
This is not a very good feature extractor!
UAS: 0.229038040231 
LAS: 0.125473013344

The parser does not work very well because the model does not take into consideration the synctactic structure of a sentence- it does not use information like the part of speech tag, lemma, etc for the tokens in a sentence. This information extraction is equivalent to the feature addition we perform in the next part which gives a much better model. 

3) Feature Extraction
a. Please find featureextractor.py modified with several new features. 
b. Please find models generated for English, Danish and Swedish with the new improved parser.
c. Updated scores:

English:
UAS: 0.798742138365 
LAS: 0.745283018868

Danish:
UAS: 0.804191616766 
LAS: 0.726147704591

Swedish: (this is the highest LAS score I received in one of the runs, sometimes the LAS goes as low as 0.667)
UAS: 0.778331009759 
LAS: 0.670782712607

d. After trying out several combinations, the combination of features that maximizes UAS/LAS scores is:
STK[0]: FORM, POSTAG, CPOSTAG, FEATS, LDEP + RDEP
STK[1]: POSTAG
BUF[0]: FORM, POSTAG, CPOSTAG, FEATS, LDEP
BUF[1]: POSTAG
BUF[2]: POSTAG

One can see that I have given preference to the POSTAG feature. This is because it's beneficial to have a wide window around the target tokens (STK[0], BUF[0]) for the POSTAG feature, and a narrow one for the FORM feature. I also eliminated the RDEP feature in BUF[0]. 

Discussion of features:

i) Part of Speech tags (POSTAG/CPOSTAG)
Part of speech tags are a particularly important feature because given unigram, bigram and trigram models for the same data, one can easily find how likely a given transition is based on it's probability. Eg., for a set of tokens 'She', 'eats', 'pizza', and part of speech tags marked appropriately, the classifier can check the following:
	-Given 'She' on the stack and 'eats' in the buffer, how likely is a VBZ form after PRP? (crude example)
	-Given a VBZ PRP, how likely is a NN form ('pizza') after that?
For POS tags, it's helpful to keep a wide window of tokens under consideration so as to make the best transition possible after considering (at least) 3-token sequences and their parts of speech together. Hence I add it to STK[0], STK[1], BUF[0], BUF[1], BUF[2]. 

Since CPOSTAG is a coarse-grained part of speech tag, it's always helpful to include it.

ii) Word form (FORM)
Word form ties in with the POS tag feature, such that, given a head word (wh) and a part-of-speech tag (ph), a dependent word (wd) and it's part-of-speech tag (pd), the classifier can check the set of all possible combined features (wh, ph, wd, pd), (wh, ph, wd), etc. to make the right transition. The form feature is important because if the classifier is given information regarding a word and the words it frequently co-occurs with, it can make a more informed decision. I've kept the window narrow for the Form feature since that's what seemed to work best. 

iii) Syntactic Features (FEATS)
This feature is extremely important too as it gives more information about a token like degree, gender, case, tense, etc, especially when it comes to parsing morphologically rich languages like, say, Swedish, or Hindi. This feature helps identify labels correctly, and most importantly, helps reduce ambiguity. 

Note that features like LEMMA did not improve accuracy at all, probably because for most tokens they were empty, or "non-informative". Overall, I noticed a high jump in LAS values on including the POSTAG feature in my feature vector. 

Trade-offs in the Nivre's arc-eager dependency parsing algorithm:
Nivre’s algorithm is a linear-time algorithm and is very fast compared to standard bottom-up parsers. One limitation is that this system can only derive projective dependency trees, however, non-projective dependencies can be captured using Nivre's pseudo-projective parsing technique. This is a serious limitation given that linguistically adequate syntactic representations sometimes require non-projective dependency trees. However, the advantage is that the parsing algorithm is less sensitive to error propagation as it considers all possible transitions (even sub-optimal ones) and has high accuracy. 

References:
Dependency Parsing (Sandra Kubler, Ryan McDonald, Joakin Nivre)
On the Role of Morphosyntactic Features in Hindi Dependency Parsing (Bharat Ram Ambati, Samar Husain, Joakim Nivre† and Rajeev Sangal)

4) Parsing arbitrary sentences
a. Please find a file parse.py that can be called as follows:
cat englishfile | python parse.py english.model > englishfile.conll

Note: The heads and dependencies for all tokens are marked at '0' and '_'. This probably means there are some issues with the classifier, however, as one of the TAs pointed out, having all words connected to the ROOT still constitues a valid parse.
