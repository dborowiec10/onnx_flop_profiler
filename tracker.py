from .hooks import *
from .util.misc import *
from onnx import numpy_helper
from onnx import helper
import numpy as np
import os
import csv

INPUT_IDENTIFIER = "model_input"
OUTPUT_IDENTIFIER = "model_output"
FIELDS = [
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


class ModelStats(object):
  def __init__(self, model_name, onnx_model):
    super().__init__()
    assert onnx_model is not None, "ONNX model is none"
    self.metadata = {}
    self.metadata["model_name"] = model_name
    self.model_inputs = {}
    self.model_initializers = {}
    self.model_sparse_initializer = {}
    self.model_quantization_annotation = {}
    self.model_outputs = {}
    self.model_nodes = {}
    self.model_node_io = {}
    self.layer_statistics = []
    self.operation_statistics = {}
    self.total_statistics = {}
    self.model = onnx_model
    self._init()

  def _get_identifier_type(self, name):
    if name in self.model_inputs:
        return "model_input"
    if name in self.model_initializers:
        return "model_initializer"
    if name in self.model_outputs:
        return "model_output"
    if name in self.model_nodes:
        return "model_node"
    if name in self.model_sparse_initializer:
        raise ValueError()
        # TODO: when we come across this, let's check what need to be done.
        return "model_sparse_initializer"
    if name in self.model_quantization_annotation:
        raise ValueError()
        # TODO: when we come across this, let's check what need to be done.
        return "model_quantization_annotation"
    return "intermediary"

  def _init_meta(self):
    # gather metadata
    self.metadata["producer_name"] = self.model.producer_name
    self.metadata["producer_ver"] = self.model.producer_version
    self.metadata["domain"] = self.model.domain
    self.metadata["name"] = self.model.graph.name
    # self.metadata["ir_version"] = self.model.graph.ir_version
  
  def _init_model_graph(self):
    # inputs
    for i, inp in enumerate(self.model.graph.input):
      self.model_inputs[inp.name] = i
    # initializers
    for i, init in enumerate(self.model.graph.initializer):
      self.model_initializers[init.name] = i
    # outputs 
    for i, out in enumerate(self.model.graph.output):
      self.model_outputs[out.name] = i

    # nodes
    for i, n in enumerate(self.model.graph.node):
      node_name = n.name.strip()
      if node_name == "":
        # TODO: temp fix because no name in pytorch by default.?
        node_name = n.op_type+"_"+str(i)
        n.name=node_name
      self.model_nodes[n.name] = i
    
    # sparse initializer
    for i, sinit in enumerate(self.model.graph.sparse_initializer):
      self.model_sparse_initializer[sinit.name] = i
      raise ValueError()

    # quantization annotation
    for i, qa in enumerate(self.model.graph.quantization_annotation):
      self.model_quantization_annotation[qa.name] = i
      raise ValueError()

    # io.
    for n in self.model.graph.node:
      for n_inp in n.input:
        self._add_node_io(n, n_inp, INPUT_IDENTIFIER)
      for n_out in n.output:
        self._add_node_io(n, n_out, OUTPUT_IDENTIFIER)
  
  # adds or updates input output for model nodes
  # type of io - model_input, model_output, model_initializer, intermediary
  # producer - name of node which produced this io as output
  # consumer - name of node which consumed this io as input 
  def set_node_io(self, name, _type, producer=None, consumer=None, data=None):
      nio = {}
      if name in self.model_node_io:
          nio = self.model_node_io[name]
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
      self.model_node_io[name] = nio


  def _add_node_io(self, node, i_or_o, io_identifier):
    shape = []
    identifier = None
    _type = self._get_identifier_type(i_or_o)

    # if i_or_o is input of this node. then the node is not a producer.
    # if i_or_o is output of this node. then the node is a producer.
    produce = None if io_identifier == INPUT_IDENTIFIER else node.name
    # if i_or_o is input to this node, then the node is consumer.
    # if i_or_o is output of this node, then the node is not a consumer
    consume = None if io_identifier == OUTPUT_IDENTIFIER else node.name
    
    if _type == io_identifier:
      type_id = self._get_identifier(i_or_o)
      shape = convert_tensor_shape_to_array(type_id)
      identifier = type_id.name
    
    self.set_node_io(i_or_o, _type, producer=produce, consumer=consume, data={
      "shape": shape,
      "identifier": identifier
    })

  def _init(self):
    self._init_meta()
    self._init_model_graph()

  def get_io_identifier(self, n_io, is_input=True):
    io = self.model_node_io[n_io]
    check_dict = self.model_inputs if is_input else self.model_outputs
    if io["data"]["identifier"] not in check_dict:
      io["data"]["identifier"] = self._get_identifier(n_io)
    return io

  # retrieves identifier by name
  def _get_identifier(self, name):
    if name in self.model_inputs:
        return self.model.graph.input[self.model_inputs[name]]
    elif name in self.model_outputs:
        return self.model.graph.output[self.model_outputs[name]]
    elif name in self.model_initializers:
        return self.model.graph.initializer[self.model_initializers[name]]
    elif name in self.model_nodes:
        return self.model.graph.node[self.model_nodes[name]]
    elif name in self.model_sparse_initializer:
      return self.model.graph.sparse_initializer[self.model_sparse_initializer[name]]
    elif name in self.model_quantization_annotation:
      return self.model.graph.quantization_annotation[self.model_quantization_annotation[name]]
    return None

  # adds towards accumulated footprint for an operation
  def add_footprint(self, op_type, footprint):
    footprint["op_type"] = op_type
    # multiply memory footprint by 4 for float-byte size
    footprint["memory"]["activations"] *= 4
    footprint["memory"]["parameters"] *= 4
    footprint["memory"]["other_memory"] *= 4
    self.layer_statistics.append(footprint)


  def count(self):
    for n in self.model.graph.node:
      hook_fn = hook.get(n.op_type, None)
      assert hook_fn != None, f"Unsupported operation from op: {n}"

      _inputs = []
      _outputs = []
      _attributes = {}
      for n_inp in n.input:
        io = self.get_io_identifier(n_inp, True)
        _inputs.append(io)
      for n_out in n.output:
        io = self.get_io_identifier(n_out, False)
        _outputs.append(io)
      for a in n.attribute:
        _attributes[a.name] = helper.get_attribute_value(a)
      
      # the _outputs, will get updated!
      # meaning the stored node io in self.model_node_io will get updated.
      footprint = hook[n.op_type](n, _inputs, _outputs, _attributes)
      self.add_footprint(n.op_type, footprint)
  
  def _prep_stats(self):
    for l in self.layer_statistics:
      if l["op_type"] not in self.operation_statistics:
        self.operation_statistics[l["op_type"]] = flatten_dict({k:v for k,v in l.items() if k != "op_type"})
      else:
        layer = flatten_dict({k:v for k,v in l.items() if k != "op_type"})
        self.operation_statistics[l["op_type"]]["operations_flops"] += layer["operations_flops"] 
        self.operation_statistics[l["op_type"]]["operations_multiply_adds"] += layer["operations_multiply_adds"]
        self.operation_statistics[l["op_type"]]["operations_additions"] += layer["operations_additions"]
        self.operation_statistics[l["op_type"]]["operations_comparisons"] += layer["operations_comparisons"]
        self.operation_statistics[l["op_type"]]["operations_divisions"] += layer["operations_divisions"]
        self.operation_statistics[l["op_type"]]["operations_exponentials"] += layer["operations_exponentials"] 
        self.operation_statistics[l["op_type"]]["memory_activations"] += layer["memory_activations"] 
        self.operation_statistics[l["op_type"]]["memory_parameters"] += layer["memory_parameters"] 
        self.operation_statistics[l["op_type"]]["memory_other_memory"] += layer["memory_other_memory"]

    self.total_statistics = {
      "operations_flops": 0, "operations_multiply_adds": 0,
      "operations_additions": 0, "operations_comparisons": 0, 
      "operations_divisions": 0, "operations_exponentials": 0, 
      "memory_activations": 0, "memory_parameters": 0, "memory_other_memory": 0
    }

    for k,v in self.operation_statistics.items():
      self.total_statistics["operations_flops"] += v["operations_flops"] 
      self.total_statistics["operations_multiply_adds"] += v["operations_multiply_adds"]
      self.total_statistics["operations_additions"] += v["operations_additions"]
      self.total_statistics["operations_comparisons"] += v["operations_comparisons"]
      self.total_statistics["operations_divisions"] += v["operations_divisions"]
      self.total_statistics["operations_exponentials"] += v["operations_exponentials"] 
      self.total_statistics["memory_activations"] += v["memory_activations"] 
      self.total_statistics["memory_parameters"] += v["memory_parameters"] 
      self.total_statistics["memory_other_memory"] += v["memory_other_memory"]


  def export_stats(self, path, export_node_only=False, export_op_only=False):
    self._prep_stats()

    filename = os.path.basename(path)
    filename_noext = os.path.splitext(filename)[0]
    dir_path = os.path.dirname(path)
    producer_name = self.metadata["producer_name"]
    producer_ver = self.metadata["producer_ver"]
    domain = self.metadata["domain"]
    graph_name = self.metadata["name"]
    model_name = self.metadata["model_name"]
    if export_node_only:
      node_csv_path = f"{os.path.join(dir_path, filename_noext)}_node_wise.csv"
      with open(node_csv_path, "w+") as nf:
         nf.write("Model Producer Name, Model Producer Ver., Model Domain, Model Name, Graph Name\n")
         nf.write(producer_name + "," + producer_ver + "," + domain + "," + model_name + "," + graph_name + "\n")
         nf.write(",,,,\n")
         writer = csv.DictWriter(nf, fieldnames=FIELDS)
         writer.writeheader()
         for i in self.layer_statistics:
          writer.writerow(flatten_dict(i))

    if export_op_only:
      op_csv_path = f"{os.path.join(dir_path, filename_noext)}_op_wise.csv"
      with open(op_csv_path, "w+") as of:
        of.write("Model Producer Name, Model Producer Ver., Model Domain, Model name, Graph Name\n")
        of.write(producer_name + "," + producer_ver + "," + domain + "," + model_name + "," + graph_name + "\n")
        of.write(",,,,\n")
        writer = csv.DictWriter(of, fieldnames=FIELDS)
        writer.writeheader()
        for k,v in self.operation_statistics.items():
            v["op_type"] = k
            writer.writerow(v)
    
    # total.
    with open(path, "w+") as tf:
      tf.write("Model Producer Name, Model Producer Ver., Model Domain, Model name, Graph Name\n")
      tf.write(producer_name + "," + producer_ver + "," + domain + "," + model_name + "," + graph_name + "\n")
      tf.write(",,,,\n")
      writer = csv.DictWriter(tf, fieldnames=FIELDS[1:])
      writer.writeheader()
      writer.writerow(self.total_statistics)

        