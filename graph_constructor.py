from absl import app
from absl import flags
import onnx
from onnx import ModelProto
from onnx import AttributeProto
from pprint import pprint
from hooks import hook

model_inputs = {}
model_initializers = {}
model_outputs = {}
model_nodes = {}
node_inputs_outputs = {}
mod = None

FLAGS = flags.FLAGS
flags.DEFINE_string('input', 'models/imagenet_xception.onnx', 'The filename of the model in *.onnx format')

# adds or updates input output for model nodes
# type of io - model_input, model_output, model_initializer, intermediary
# producer - name of node which produced this io as output
# consumer - name of node which consumed this io as input 
def add_node_io(name, _type, producer=None, consumer=None):
    nio = {}
    if name in node_inputs_outputs:
        nio = node_inputs_outputs[name]
    else:
        nio["name"] = name
        nio["producer"] = None
        nio["consumers"] = []
        nio["type"] = _type
        nio["data"] = None
    if producer != None:
        nio["producer"] = producer
    if consumer != None:
        nio["consumers"].append(consumer)
    node_inputs_outputs[name] = nio

# retrieves identifier type by name
def get_identifier_type(name):
    if name in model_inputs:
        return "model_input"
    elif name in model_initializers:
        return "model_initializer"
    elif name in model_outputs:
        return "model_output"
    elif name in model_nodes:
        return "model_node"
    return "intermediary"

# retrieves identifier by name
def get_identifier(name):
    global mod
    if name in model_inputs:
        return mod.graph.input[model_inputs[name]]
    elif name in model_outputs:
        return mod.graph.output[model_outputs[name]]
    elif name in model_initializers:
        return mod.graph.initializer[model_initializers[name]]
    elif name in model_nodes:
        return mod.graph.node[model_nodes[name]]
    return None


# initialize data structures
def init(filename):
    global mod
    # load model
    mod = onnx.load(filename)    
    # gather model input(s)
    for i, inp in enumerate(mod.graph.input):
        model_inputs[inp.name] = i
    # gather model initializers
    for i, init in enumerate(mod.graph.initializer):
        model_initializers[init.name] = i
    # gather model output(s)
    for i, out in enumerate(mod.graph.output):
        model_outputs[out.name] = i
    # gather model nodes
    for i, node in enumerate(mod.graph.node):
        model_nodes[node.name] = i
    # create node io link table
    for node in mod.graph.node:
        for nin in node.input:
            _type = get_identifier_type(nin)
            add_node_io(nin, _type, consumer=node.name)
        for nout in node.output:
            _type = get_identifier_type(nout)
            add_node_io(nout, _type, producer=node.name)

# retrieve value of the attribute
def get_attr_value(a):
    return None if int(a.type) < 1 or int(a.type) > 12 else {
        1: a.f,
        2: a.i,
        3: a.s,
        4: a.t,
        5: a.g,
        6: a.floats,
        7: a.ints,
        8: a.strings,
        9: a.tensors,
        10: a.graphs,
        11: a.sparse_tensor,
        12: a.sparse_tensors
    }[a.type]


def run():
    global mod
    for node in mod.graph.node:
        hook_fun = hook[node.op_type] if node.op_type in hook else None
        if hook_fun != None:
            _inputs = []
            _outputs = []
            _attributes = {}
            for inp in node.input:
                ident = get_identifier(inp)
                if ident != None:
                    node_inputs_outputs[inp]["data"] = ident
                _inputs.append(node_inputs_outputs[inp])
            for out in node.output:
                ident = get_identifier(out)
                if ident != None:
                    node_inputs_outputs[out]["data"] = ident
                _outputs.append(node_inputs_outputs[out])
            for a in node.attribute:
                _attributes[a.name] = get_attr_value(a)
            retval = hook[node.op_type](_inputs, _outputs, _attributes)
        else:
            print("Unsupported operation:", node.op_type)

# main program function
def main(argv):
    del argv
    init(FLAGS.input)
    run()

# entrypoint
if __name__ == "__main__":
    app.run(main)





