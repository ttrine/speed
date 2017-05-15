from fish.classify import ClassifierContainer
from experiments.final_submission_1.train import construct

if __name__ == '__main__':
	import sys # basic arg parsing, infer name
	name = sys.argv[0].split('/')[-2]
	
	if len(sys.argv) < 3:
		print "Usage: submit weight_file clip"
		sys.exit()

	c = ClassifierContainer(name,construct(),32,"adam")
	c.evaluate(str(sys.argv[1]), bool(sys.argv[2]))
