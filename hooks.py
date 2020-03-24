from pprint import pprint
from onnx import numpy_helper
import numpy as np

# [batch, channels, height, width] - N x C x H x W

def batch_norm(inputs, outputs, attributes):
    # Based on https://arxiv.org/abs/1502.03167.

    # prepare footprint counters
    comp_additions = 0
    comp_divisions = 0
    mem_parameters = 0
    mem_activations = 0

    # collect input information
    input_dims = inputs[0]["data"]["shape"]
    if input_dims == None or len(input_dims) < 1:
        print("No input dimensions specified for Conv layer")
    else:
        # data dimension doesn't change
        outputs[0]["data"]["shape"] = input_dims
        
        # calculate footprint
        comp_additions = np.prod(input_dims[2:]) * input_dims[1] * input_dims[0]
        comp_divisions = comp_additions

        mem_parameters = input_dims[1] * 2
        mem_activations = comp_divisions

    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "comps": 0,
            "additions": comp_additions,
            "divisions": comp_divisions,
            "exponentials": 0
        },
        "memory": {
            "parameters": mem_parameters,
            "activations": mem_activations,
            "other_memory": 0
        }
    }

def clip(inputs, outputs, attributes):
    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "comps": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory": {
            "parameters": 0,
            "activations": 0,
            "other_memory": 0
        }
    }

def conv(inputs, outputs, attributes):
    # convolution output size based on the following publication: https://arxiv.org/pdf/1603.07285.pdf

    # prepare footprint counters
    comp_multiply_adds = 0
    comp_flops = 0
    mem_parameters = 0
    mem_activations = 0

    # collect input dimensions
    bias_dims = []
    input_dims = inputs[0]["data"]["shape"]
    if input_dims == None or len(input_dims) < 1:
        # TODO: this needs to check whether the identifier does not exist - if does, convert dims into input_dims as python array and save to input
        print("No input dimensions specified for Conv layer")
    else:
        weights_dims = inputs[1]["data"]["identifier"].dims if inputs[1]["data"]["identifier"] != None else None
        if weights_dims == None:
            raise Error("Undefined weigths input for Conv layer")
        if len(inputs) > 2:
            bias_dims = inputs[2]["data"]["identifier"].dims

        # should be available, otherwise take weights and return anything after the first 2 dimensions
        kernel_shape = attributes["kernel_shape"] if "kernel_shape" in attributes else weights_dims[2:]

        # collect other attributes
        dilations = attributes["dilations"] if "dilations" in attributes else [1 for k in kernel_shape]
        group = attributes["group"] if "group" in attributes else 1
        strides = attributes["strides"] if "strides" in attributes else [1 for k in kernel_shape]

        # collect pad information
        # default auto_pad is NOTSET
        # pads[] will only be defined if this is NOTSET
        pads = []
        auto_pad = attributes["auto_pad"].decode("utf-8") if "auto_pad" in attributes else "NOTSET"
        if auto_pad == "VALID":
            pads = [0 for k in kernel_shape]
        elif auto_pad == "SAME_UPPER" or auto_pad == "SAME_LOWER":
            pads = np.floor([k / 2 for k in kernel_shape])
        elif auto_pad == "NOTSET" and "pads" in attributes:
            begin_p = attributes["pads"][:len(kernel_shape)]
            end_p = attributes["pads"][len(kernel_shape):]
            pads = np.add(begin_p, end_p).tolist()
        else:
            pads = [0 for k in kernel_shape]

        # calculate output size
        output_size = []
        for i in range(len(input_dims[2:])):
            padded_out = input_dims[2:][i] + pads[i] - kernel_shape[i]
            dilated_out = padded_out - ((kernel_shape[i] - 1) * (dilations[i] - 1))
            strided_out = dilated_out / strides[i]
            floored_out = np.floor(strided_out)
            output_size.append(floored_out + 1) # to account for initial stride position

        output_size = np.array(output_size).astype(int).tolist()
        output_dims = [input_dims[0], weights_dims[0]] + output_size

        # update output
        outputs[0]["data"]["shape"] = output_dims

        # calculate footprint
        comp_multiply_adds = np.prod(kernel_shape) * np.prod(output_dims[2:]) * input_dims[1] * output_dims[1] * output_dims[0] / group
        comp_flops = comp_multiply_adds * 2
        mem_parameters = np.prod(kernel_shape) * input_dims[1] * output_dims[1] / group + (1 if len(bias_dims) > 0 else 0) * output_dims[1]
        mem_activations = np.prod(output_dims[2:]) * output_dims[1] * output_dims[0]

    return {
        "operations": {
            "flops": comp_flops,
            "multiply_adds": comp_multiply_adds,
            "comps": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory": {
            "parameters": mem_parameters,
            "activations": mem_activations,
            "other_memory": 0
        }
    }

def global_avg_pool(inputs, outputs, attributes):
    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "comps": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory": {
            "parameters": 0,
            "activations": 0,
            "other_memory": 0
        }
    }

def avg_pool(inputs, outputs, attributes):
    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "comps": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory": {
            "parameters": 0,
            "activations": 0,
            "other_memory": 0
        }
    }

def max_pool(inputs, outputs, attributes):
    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "comps": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory": {
            "parameters": 0,
            "activations": 0,
            "other_memory": 0
        }
    }

def identity(inputs, outputs, attributes):
    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "comps": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory": {
            "parameters": 0,
            "activations": 0,
            "other_memory": 0
        }
    }



def pad(inputs, outputs, attributes):
    # Pad follows similar structure to np.pad - https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html
    # we are interested only in the _before, _after pads for changing input shape
    # mode will not be used as the data is irrelevant
    # Pad will have a data tensor as its input[0], array of pads[1] and optional constant value[2] 
    # Pad will have a singular tensor as its output
    # Pad will most likely extend the input size by some amount
    # this will have to be allocated in memory as a separate region
    # worth considering this as a footprint
    input_shape = inputs[0]["data"]["shape"]
    out_shape = []
    # pads are in the form of [dim1_begin, ..., dimN_begin, dim1_end, ..., dimN_end]
    pads = numpy_helper.to_array(inputs[1]["data"]["identifier"])
    # number of dimensions in input
    num_inp_dims = len(input_shape)
    # split pads into respective begins and ends for each dimension
    pads_begins = pads[:num_inp_dims]
    pads_ends = pads[num_inp_dims:]
    # for each dimension, add begin+end to size of the dimension
    for i in range(0, num_inp_dims):
        both = pads_begins[i] + pads_ends[i]
        out_shape.append(input_shape[i] + both)
    outputs[0]["data"]["shape"] = out_shape
    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "comps": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory": {
            "parameters": 0,
            "activations": 0,
            "other_memory": (np.prod(out_shape) - np.prod(input_shape))
        }
    }

def reshape(inputs, outputs, attributes):
    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "comps": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory": {
            "parameters": 0,
            "activations": 0,
            "other_memory": 0
        }
    }

def softmax(inputs, outputs, attributes):
    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "comps": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory": {
            "parameters": 0,
            "activations": 0,
            "other_memory": 0
        }
    }

def transpose(inputs, outputs, attributes):
    # https://www.researchgate.net/publication/273912700_In-Place_Matrix_Transposition_on_GPUs
    # Transpose has no computation measured as it only changes the order of channels (usually in place)
    # Transpose will have a singular tensor as its input
    # Transpose will have a singular tensor as its output
    shape = inputs[0]["data"]["shape"]
    perms = attributes["perm"]
    if len(shape) == len(perms):
        out_shape = []
        for p in perms:
            out_shape.append(shape[p])
        outputs[0]["data"]["shape"] = out_shape
    else:
        print("Shape doesn't match Permutations")
    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "comps": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory": {
            "parameters": 0,
            "activations": 0,
            "other_memory": 0
        }
    }

def relu(inputs, outputs, attributes):
    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "comps": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory": {
            "parameters": 0,
            "activations": 0,
            "other_memory": 0
        }
    }

def concat(inputs, outputs, attributes):
    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "comps": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory": {
            "parameters": 0,
            "activations": 0,
            "other_memory": 0
        }
    }

def add(inputs, outputs, attributes):
    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "comps": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory": {
            "parameters": 0,
            "activations": 0,
            "other_memory": 0
        }
    }

def mul(inputs, outputs, attributes):
    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "comps": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory": {
            "parameters": 0,
            "activations": 0,
            "other_memory": 0
        }
    }

def mat_mul(inputs, outputs, attributes):
    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "comps": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory": {
            "parameters": 0,
            "activations": 0,
            "other_memory": 0
        }
    }

def _slice(inputs, outputs, attributes):
    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "comps": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory": {
            "parameters": 0,
            "activations": 0,
            "other_memory": 0
        }
    }

def flatten(inputs, outputs, attributes):
    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "comps": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory": {
            "parameters": 0,
            "activations": 0,
            "other_memory": 0
        }
    }


hook = {
    "BatchNormalization": batch_norm,
    "Clip": clip,
    "Conv": conv,
    "GlobalAveragePool": global_avg_pool,
    "AveragePool": avg_pool,
    "MaxPool": max_pool,
    "Identity": identity,
    "Pad": pad,
    "Reshape": reshape,
    "Softmax": softmax,
    "Transpose": transpose,
    "Relu": relu,
    "Concat": concat,
    "Add": add,
    "Mul": mul,
    "MatMul": mat_mul,
    "Slice": _slice,
    "Flatten": flatten
}
