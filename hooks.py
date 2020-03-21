from pprint import pprint
from onnx import numpy_helper


# [batch, channels, height, width] - N x C x H x W

def batch_norm(inputs, outputs, attributes):
    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "computations": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory_footprint": {
            "parameters": 0,
            "activations": 0
        }
    }

def clip(inputs, outputs, attributes):
    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "computations": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory_footprint": {
            "parameters": 0,
            "activations": 0
        }
    }

def conv(inputs, outputs, attributes):
    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "computations": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory_footprint": {
            "parameters": 0,
            "activations": 0
        }
    }

def global_avg_pool(inputs, outputs, attributes):
    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "computations": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory_footprint": {
            "parameters": 0,
            "activations": 0
        }
    }

def avg_pool(inputs, outputs, attributes):
    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "computations": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory_footprint": {
            "parameters": 0,
            "activations": 0
        }
    }

def max_pool(inputs, outputs, attributes):
    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "computations": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory_footprint": {
            "parameters": 0,
            "activations": 0
        }
    }

def identity(inputs, outputs, attributes):
    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "computations": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory_footprint": {
            "parameters": 0,
            "activations": 0
        }
    }


def pad(inputs, outputs, attributes):
    # Pad follows similar structure to np.pad - https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html
    # we are interested only in the _before, _after pads for changing input shape
    # need to arrange pads defined as an array into 2-tuples of before/after value extensions for each dimension
    # mode will not be used as the data is irrelevant
    # Pad will have a data tensor as its input[0], array of pads[1] and optional constant value[2] 
    # Pad will have a singular tensor as its output
    input_shape = inputs[0]["data"]["shape"]
    out_shape = []
    pads = numpy_helper.to_array(inputs[1]["data"]["identifier"])
    dim_pads = []
    for i in range(0, len(pads), 2):
        dim_pads.append(pads[i] + pads[i + 1])
    
    for i in range(0, len(input_shape), 1):
        out_shape.append(input_shape[i] + dim_pads[i])

    outputs[0]["data"]["shape"] = out_shape

    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "computations": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory_footprint": {
            "parameters": 0,
            "activations": 0
        }
    }

def reshape(inputs, outputs, attributes):
    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "computations": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory_footprint": {
            "parameters": 0,
            "activations": 0
        }
    }

def softmax(inputs, outputs, attributes):
    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "computations": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory_footprint": {
            "parameters": 0,
            "activations": 0
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
            "computations": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory_footprint": {
            "parameters": 0,
            "activations": 0
        }
    }

def relu(inputs, outputs, attributes):
    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "computations": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory_footprint": {
            "parameters": 0,
            "activations": 0
        }
    }

def concat(inputs, outputs, attributes):
    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "computations": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory_footprint": {
            "parameters": 0,
            "activations": 0
        }
    }

def add(inputs, outputs, attributes):
    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "computations": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory_footprint": {
            "parameters": 0,
            "activations": 0
        }
    }

def mul(inputs, outputs, attributes):
    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "computations": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory_footprint": {
            "parameters": 0,
            "activations": 0
        }
    }

def mat_mul(inputs, outputs, attributes):
    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "computations": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory_footprint": {
            "parameters": 0,
            "activations": 0
        }
    }

def _slice(inputs, outputs, attributes):
    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "computations": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory_footprint": {
            "parameters": 0,
            "activations": 0
        }
    }

def flatten(inputs, outputs, attributes):
    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "computations": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory_footprint": {
            "parameters": 0,
            "activations": 0
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