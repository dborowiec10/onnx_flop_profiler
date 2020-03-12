import onnx
from pprint import pprint
from onnx import ModelProto

mod = onnx.load("imagenet_mobilenet_alpha_0.25.onnx")
for node in mod.graph.node:
    # print(node.input)
    print(node.output)
    print(node.op_type)
    print(node)


for init in mod.graph.initializer:
    continue
    # print(init.dims)
    # print(init.name)

for inp in mod.graph.input:
    print(inp)

for inp in mod.graph.output:
    print(inp)

# pprint(mod)

# import parser
# from pprint import pprint
# import networkx as nx
# import matplotlib.pyplot as plt

# G = nx.Graph()


# def find_op_by_output_name(out_name, ops):
#     for o in reversed(ops):
#         if o["output"]["name"] == out_name:
#             return o
#     return None


# # plt.subplot(121)
# # nx.draw(G, with_labels=True, font_weight='bold')
# # plt.show()


# out = parser.parse("imagenet_vgg16.graph")

# G.add_node("Output", name=out["return"], type="variable")
# o = find_op_by_output_name(G.nodes["Output"]["name"], out["operations"])
# print(o)
# G.add_node(o["operation"], arguments=o["arguments"])

# print(G.nodes["Output"])


# # for ops in reversed(out["operations"]):




# # pprint(out)
