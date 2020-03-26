from absl import app
from absl import flags
from os import walk
from pprint import pprint
import onnx
from onnx import numpy_helper
from onnx import helper
import copy
import csv

from hooks import hook

model_inputs = {}
model_initializers = {}
model_outputs = {}
model_nodes = {}
node_inputs_outputs = {}
mod = None

mod_producer_name = ""
mod_producer_ver = ""
mod_domain = ""
mod_name = ""
mod_filename = ""
stat_preamble = ""

layer_statistics = []
operation_statistics = {}
total_statistics = {}

FLAGS = flags.FLAGS
flags.DEFINE_string('input', 'default', 'The filename of the model in *.onnx format')
flags.DEFINE_integer('batch_size', 1, 'assumed input data mini-batch size')
flags.DEFINE_boolean('modify_batch_size', True, 'Whether to modify the mini-batch size on model input. Modifies to <batch_size>')

def flatten_dict(init, lkey=''):
    ret = {}
    for rkey,val in init.items():
        key = lkey+rkey
        if isinstance(val, dict):
            ret.update(flatten_dict(val, key+'_'))
        else:
            ret[key] = val
    return ret

def save_stats():
    global layer_statistics
    global operation_statistics
    global total_statistics

    prep_stats()

    fields = [
        "op_type",
        "operations_flops",
        "operations_multiply_adds",
        "operations_additions",
        "operations_comparisons",
        "operations_divisions",
        "operations_exponentials",
        "memory_activations",
        "memory_parameters",
        "memory_other_memory"
    ]
    with open(stat_preamble + "_node_wise.csv", 'w') as node_wise_file:
        node_wise_file.write("Model Producer Name, Model Producer Ver., Model Domain, Model Filename, Model Name\n")
        node_wise_file.write(mod_producer_name + "," + mod_producer_ver + "," + mod_domain + "," + mod_filename + "," + mod_name + "\n")
        node_wise_file.write(",,,,\n")
        writer = csv.DictWriter(node_wise_file, fieldnames=fields)
        writer.writeheader()
        for i in layer_statistics:
            writer.writerow(flatten_dict(i))

    with open(stat_preamble + "_op_wise.csv", 'w') as op_wise_file:
        op_wise_file.write("Model Producer Name, Model Producer Ver., Model Domain, Model Filename, Model Name\n")
        op_wise_file.write(mod_producer_name + "," + mod_producer_ver + "," + mod_domain + "," + mod_filename + "," + mod_name + "\n")
        op_wise_file.write(",,,,\n")
        writer = csv.DictWriter(op_wise_file, fieldnames=fields)
        writer.writeheader()
        for k,v in operation_statistics.items():
            v["op_type"] = k
            writer.writerow(v)

    with open(stat_preamble + "_total.csv", 'w') as total_file:
        total_file.write("Model Producer Name, Model Producer Ver., Model Domain, Model Filename, Model Name\n")
        total_file.write(mod_producer_name + "," + mod_producer_ver + "," + mod_domain + "," + mod_filename + "," + mod_name + "\n")
        total_file.write(",,,,\n")
        writer = csv.DictWriter(total_file, fieldnames=fields[1:])
        writer.writeheader()
        writer.writerow(total_statistics)

def prep_stats():
    global operation_statistics
    global total_statistics

    for i in layer_statistics:
        if i["op_type"] not in operation_statistics:
            operation_statistics[i["op_type"]] = flatten_dict({k:v for k,v in i.items() if k != "op_type"})
        else:
            layer = flatten_dict({k:v for k,v in i.items() if k != "op_type"})
            operation_statistics[i["op_type"]]["operations_flops"] += layer["operations_flops"] 
            operation_statistics[i["op_type"]]["operations_multiply_adds"] += layer["operations_multiply_adds"]
            operation_statistics[i["op_type"]]["operations_additions"] += layer["operations_additions"]
            operation_statistics[i["op_type"]]["operations_comparisons"] += layer["operations_comparisons"]
            operation_statistics[i["op_type"]]["operations_divisions"] += layer["operations_divisions"]
            operation_statistics[i["op_type"]]["operations_exponentials"] += layer["operations_exponentials"] 
            operation_statistics[i["op_type"]]["memory_activations"] += layer["memory_activations"] 
            operation_statistics[i["op_type"]]["memory_parameters"] += layer["memory_parameters"] 
            operation_statistics[i["op_type"]]["memory_other_memory"] += layer["memory_other_memory"]

    total_statistics = {
        "operations_flops": 0, "operations_multiply_adds": 0,
        "operations_additions": 0, "operations_comparisons": 0, 
        "operations_divisions": 0, "operations_exponentials": 0, 
        "memory_activations": 0, "memory_parameters": 0, "memory_other_memory": 0
    }

    for k,v in operation_statistics.items():
        total_statistics["operations_flops"] += v["operations_flops"] 
        total_statistics["operations_multiply_adds"] += v["operations_multiply_adds"]
        total_statistics["operations_additions"] += v["operations_additions"]
        total_statistics["operations_comparisons"] += v["operations_comparisons"]
        total_statistics["operations_divisions"] += v["operations_divisions"]
        total_statistics["operations_exponentials"] += v["operations_exponentials"] 
        total_statistics["memory_activations"] += v["memory_activations"] 
        total_statistics["memory_parameters"] += v["memory_parameters"] 
        total_statistics["memory_other_memory"] += v["memory_other_memory"]


# adds towards accumulated footprint for an operation
def add_footprint(op_type, footprint):
    global layer_statistics
    footprint["op_type"] = op_type
    # multiply memory footprint by 4 for float-byte size
    footprint["memory"]["activations"] *= 4
    footprint["memory"]["parameters"] *= 4
    footprint["memory"]["other_memory"] *= 4
    layer_statistics.append(footprint)


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
    global mod_producer_name
    global mod_producer_ver
    global mod_domain
    global mod_name
    global mod_filename
    global stat_preamble

    # load model
    mod = onnx.load(filename)

    # gather metadata
    mod_producer_name = mod.producer_name
    mod_producer_ver = mod.producer_version
    mod_domain = mod.domain
    mod_name = mod.graph.name
    mod_filename = filename
    stat_preamble = mod_domain + "_" + mod_producer_name + mod_producer_ver + "_" + mod_name

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

# run performance counting
def run():
    global mod
    for node in mod.graph.node:
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
    if FLAGS.input != "default":
        init(FLAGS.input)
    else:
        init("./models/" + f[0])
    run()
    save_stats()

# entrypoint
if __name__ == "__main__":
    app.run(main)