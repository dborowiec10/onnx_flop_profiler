from absl import app
from absl import flags
from os import walk
from pprint import pprint
import onnx
from onnx import numpy_helper
from onnx import helper

from hooks import hook


model_inputs = {}
model_initializers = {}
model_outputs = {}
model_nodes = {}
node_inputs_outputs = {}
mod = None

statistics = {}

FLAGS = flags.FLAGS
flags.DEFINE_string('input', 'default', 'The filename of the model in *.onnx format')
flags.DEFINE_boolean('test_all', False, 'whether to test all models in the ./models/ directory')
flags.DEFINE_integer('batch_size', 1, 'assumed input data mini-batch size')
flags.DEFINE_boolean('modify_batch_size', True, 'Whether to modify the mini-batch size on model input. Modifies to <batch_size>')


# adds towards accumulated footprint for an operation
def add_footprint(op_type, footprint):
    if op_type not in statistics:
        statistics[op_type] = footprint
    else:
        statistics[op_type]["operations"]["flops"] += footprint["operations"]["flops"]
        statistics[op_type]["operations"]["multiply_adds"] += footprint["operations"]["multiply_adds"]
        statistics[op_type]["operations"]["comps"] += footprint["operations"]["comps"]
        statistics[op_type]["operations"]["additions"] += footprint["operations"]["additions"]
        statistics[op_type]["operations"]["divisions"] += footprint["operations"]["divisions"]
        statistics[op_type]["operations"]["exponentials"] += footprint["operations"]["exponentials"]
        statistics[op_type]["memory"]["parameters"] += footprint["memory"]["parameters"]
        statistics[op_type]["memory"]["activations"] += footprint["memory"]["activations"]
        statistics[op_type]["memory"]["other_memory"] += footprint["memory"]["other_memory"]

# print statistics for the model
def print_stats():
    for key in statistics:
        print("----------------------------------------")
        print("Operation:", key)
        print("-- Computations:")
        print("-- -- FLOPs:", statistics[key]["operations"]["flops"])
        print("-- -- MACCs:", statistics[key]["operations"]["multiply_adds"])
        print("-- -- Comps:", statistics[key]["operations"]["comps"])
        print("-- -- Additions:", statistics[key]["operations"]["additions"])
        print("-- -- Divisions:", statistics[key]["operations"]["divisions"])
        print("-- -- Exponentials:", statistics[key]["operations"]["exponentials"])
        print("-- Memory:")
        print("-- -- Parameters:", statistics[key]["memory"]["parameters"])
        print("-- -- Activations:", statistics[key]["memory"]["activations"])
        print("-- -- Other:", statistics[key]["memory"]["other_memory"])
        print("----------------------------------------")

# adds or updates input output for model nodes
# type of io - model_input, model_output, model_initializer, intermediary
# producer - name of node which produced this io as output
# consumer - name of node which consumed this io as input 
def add_node_io(name, _type, producer=None, consumer=None, data=None):
    nio = {}
    if name in node_inputs_outputs:
        nio = node_inputs_outputs[name]
    else:
        nio["name"] = name
        nio["producer"] = None
        nio["consumers"] = []
        nio["type"] = _type
        nio["data"] = data
    if nio["data"] == None:
        nio["data"] = data
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

def convert_tensor_shape_to_array(t):
    arr = []
    for d in t.type.tensor_type.shape.dim:
        if d.dim_value != 0:
            arr.append(d.dim_value)
        else:
            arr.append(d.dim_param)
    return arr

# initialize data structures
def init(filename):
    global mod
    global model_inputs
    global model_initializers
    global model_outputs
    global model_nodes
    global node_inputs_outputs

    mod = None
    model_inputs = {}
    model_initializers = {}
    model_outputs = {}
    model_nodes = {}
    node_inputs_outputs = {}
    print(filename) 
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
            shape = []
            identifier = None
            _type = get_identifier_type(nin)
            if _type == "model_input":
                id = get_identifier(nin)
                shape = convert_tensor_shape_to_array(id)
                if FLAGS.modify_batch_size:
                    shape[0] = FLAGS.batch_size
                identifier = id.name
            add_node_io(nin, _type, consumer=node.name, data={
                "shape": shape,
                "identifier": identifier
            })

        for nout in node.output:
            shape = []
            identifier = None
            _type = get_identifier_type(nout)
            if _type == "model_output":
                id = get_identifier(nout)
                shape = convert_tensor_shape_to_array(id)
                if FLAGS.modify_batch_size:
                    shape[0] = FLAGS.batch_size
                identifier = id.name
            add_node_io(nout, _type, producer=node.name, data={
                "shape": shape,
                "identifier": identifier
            })

def run():
    global mod
    for node in mod.graph.node:
        # print("****************************** Performing node:", node.name)
        hook_fun = hook[node.op_type] if node.op_type in hook else None
        if hook_fun != None:
            _inputs = []
            _outputs = []
            _attributes = {}
            for inp in node.input:
                io = node_inputs_outputs[inp]
                if io["data"]["identifier"] not in model_inputs:
                    io["data"]["identifier"] = get_identifier(inp)
                _inputs.append(io)
            for out in node.output:
                io = node_inputs_outputs[out]
                if io["data"]["identifier"] not in model_outputs:
                    io["data"]["identifier"] = get_identifier(out)
                _outputs.append(io)
            for a in node.attribute:
                _attributes[a.name] = helper.get_attribute_value(a)
            footprint = hook[node.op_type](_inputs, _outputs, _attributes)
            add_footprint(node.op_type, footprint)
        else:
            print("Unsupported operation:", node.op_type)

# main program function
def main(argv):
    del argv
    f = []
    for (_, _, filenames) in walk("./models"):
        f.extend(filenames)
        break
    if FLAGS.test_all:
        for md in f:
            init("./models/" + md)
            run()
        return
    elif FLAGS.input != "default":
        init(FLAGS.input)
    else:
        init("./models/" + f[0])
    run()
    print_stats()

# entrypoint
if __name__ == "__main__":
    app.run(main)





