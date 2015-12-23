import sys
from providedcode.transitionparser import TransitionParser
from providedcode.dependencygraph import DependencyGraph

if __name__ == '__main__':
    model = sys.argv[1]
    data = sys.stdin.readlines()
    for item in data:
        sentence = DependencyGraph.from_sentence(item)
        tp = TransitionParser.load(model)
        parsed = tp.parse([sentence])
        print parsed[0].to_conll(10).encode('utf-8')
	sys.stdout.flush()
