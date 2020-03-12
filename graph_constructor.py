import parser
from pprint import pprint

out = parser.parse("imagenet_resnet152v2.graph")

pprint(out)
