from container import ModelContainer
from experiments.conv_test.train import construct

if __name__ == '__main__':
	import sys # basic arg parsing, infer name
	name = sys.argv[0].split('/')[-2]
	
	if len(sys.argv) < 2:
		print "Usage: submit weight_file"
		sys.exit()

	c = ModelContainer(name,construct(),10,"adam")
	c.evaluate(str(sys.argv[1]))
