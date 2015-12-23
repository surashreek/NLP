from nltk.compat import python_2_unicode_compatible

printed = False

@python_2_unicode_compatible
class FeatureExtractor(object):
    @staticmethod
    def _check_informative(feat, underscore_is_informative=False):
        """
        Check whether a feature is informative
        """

        if feat is None:
            return False

        if feat == '':
            return False

        if not underscore_is_informative and feat == '_':
            return False

        return True

    @staticmethod
    def find_left_right_dependencies(idx, arcs):
        left_most = 1000000
        right_most = -1
        dep_left_most = ''
        dep_right_most = ''
        for (wi, r, wj) in arcs:
            if wi == idx:
                if (wj > wi) and (wj > right_most):
                    right_most = wj
                    dep_right_most = r
                if (wj < wi) and (wj < left_most):
                    left_most = wj
                    dep_left_most = r
        return dep_left_most, dep_right_most

    @staticmethod
    def extract_features(tokens, buffer, stack, arcs):
        """
        This function returns a list of string features for the classifier

        :param tokens: nodes in the dependency graph
        :param stack: partially processed words
        :param buffer: remaining input words
        :param arcs: partially built dependency tree

        :return: list(str)
        """

        """
        Think of some of your own features here! Some standard features are
        described in Table 3.2 on page 31 of Dependency Parsing by Kubler,
        McDonald, and Nivre

        [http://books.google.com/books/about/Dependency_Parsing.html?id=k3iiup7HB9UC]
        """

        result = []
	tags = []
	depth_stack = len(stack)
	depth_buffer = len(buffer)

        global printed
        if not printed:
            print("This is not a very good feature extractor!")
            printed = True

	# my implementation of feature extraction
	stack_idx0 = stack_idx1 = False 
	if depth_stack >= 2:
	    stack_idx0 = stack[-1]
	    stack_idx1 = stack[-2]
	elif depth_stack == 1:
	    stack_idx0 = stack[-1]

	buffer_idx0 = buffer_idx1 = buffer_idx2 = buffer_idx3 = False 
	if depth_buffer >= 4:
	    buffer_idx0 = buffer[0]
	    buffer_idx1 = buffer[1]
	    buffer_idx2 = buffer[2]
	    buffer_idx3 = buffer[3]
	elif depth_buffer >= 3:
	    buffer_idx0 = buffer[0]
	    buffer_idx1 = buffer[1]
	    buffer_idx2 = buffer[2]
	elif depth_buffer >= 2:
	    buffer_idx0 = buffer[0]
	    buffer_idx1 = buffer[1]
	elif depth_buffer == 1:
	    buffer_idx0 = buffer[0]

        # an example set of features:
        if stack_idx0:
            token = tokens[stack_idx0]
            if FeatureExtractor._check_informative(token['word'], True):
                result.append('STK_0_FORM_' + token['word'])

	    if 'tag' in token and FeatureExtractor._check_informative(token['tag']):
                result.append('STK_0_POSTAG_' + token['tag'])
	    if 'ctag' in token and FeatureExtractor._check_informative(token['ctag']):
                result.append('STK_0_CPOSTAG_' + token['ctag'])

            if 'feats' in token and FeatureExtractor._check_informative(token['feats']):
                feats = token['feats'].split("|")
                for feat in feats:
                    result.append('STK_0_FEATS_' + feat)

            # Left most, right most dependency of stack[0]
            dep_left_most, dep_right_most = FeatureExtractor.find_left_right_dependencies(stack_idx0, arcs)

            if FeatureExtractor._check_informative(dep_left_most):
                result.append('STK_0_LDEP_' + dep_left_most)
            if FeatureExtractor._check_informative(dep_right_most):
                result.append('STK_0_RDEP_' + dep_right_most)
	
        if stack_idx1:
            token = tokens[stack_idx1]
	   
	    if 'tag' in token and FeatureExtractor._check_informative(token['tag']):
                result.append('STK_1_POSTAG_' + token['tag'])

        if buffer_idx0:
            token = tokens[buffer_idx0]
            if FeatureExtractor._check_informative(token['word'], True):
                result.append('BUF_0_FORM_' + token['word'])

            if 'feats' in token and FeatureExtractor._check_informative(token['feats']):
                feats = token['feats'].split("|")
                for feat in feats:
                    result.append('BUF_0_FEATS_' + feat)
  
	    if 'tag' in token and FeatureExtractor._check_informative(token['tag']):
                result.append('BUF_0_POSTAG_' + token['tag'])
	    if 'ctag' in token and FeatureExtractor._check_informative(token['ctag']):
                result.append('BUF_0_CPOSTAG_' + token['ctag'])

            dep_left_most, dep_right_most = FeatureExtractor.find_left_right_dependencies(buffer_idx0, arcs)

            if FeatureExtractor._check_informative(dep_left_most):
                result.append('BUF_0_LDEP_' + dep_left_most)
            #if FeatureExtractor._check_informative(dep_right_most):
             #   result.append('BUF_0_RDEP_' + dep_right_most)

	if buffer_idx1:
            token = tokens[buffer_idx1]
            #if FeatureExtractor._check_informative(token['word'], True):
             #   result.append('BUF_1_FORM_' + token['word'])
		
	    if 'tag' in token and FeatureExtractor._check_informative(token['tag']):
                result.append('BUF_1_POSTAG_' + token['tag'])
	    '''
            if 'feats' in token and FeatureExtractor._check_informative(token['feats']):
                feats = token['feats'].split("|")
                for feat in feats:
                    result.append('BUF_1_FEATS_' + feat)
	    '''

	if buffer_idx2:
            token = tokens[buffer_idx2]
            if 'tag' in token and FeatureExtractor._check_informative(token['tag']):
                result.append('BUF_2_POSTAG_' + token['tag'])

	'''	
	if buffer_idx3:
            token = tokens[buffer_idx3]
            if 'tag' in token and FeatureExtractor._check_informative(token['tag']):
                result.append('BUF_3_POSTAG_' + token['tag'])
	'''
        return result
