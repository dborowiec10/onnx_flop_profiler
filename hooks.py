from onnx import numpy_helper
import numpy as np
from pprint import pprint

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
        input_dims = inputs[0]["data"]["identifier"].dims if inputs[0]["data"]["identifier"] != None else None
        if input_dims == None:
            raise Exception("1st input for GlobalAvgPool operation not specified!")
    
    # data dimension doesn't change
    outputs[0]["data"]["shape"] = input_dims
    
    # calculate footprint
    comp_additions = np.prod(input_dims)
    comp_divisions = np.prod(input_dims)
    mem_parameters = input_dims[1] * 2
    mem_activations = np.prod(input_dims)

    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "comparisons": 0,
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
        input_dims = inputs[0]["data"]["identifier"].dims if inputs[0]["data"]["identifier"] != None else None
        if input_dims == None:
            raise Exception("1st input for GlobalAvgPool operation not specified!")
    
    weights_dims = inputs[1]["data"]["identifier"].dims if inputs[1]["data"]["identifier"] != None else None
    if weights_dims == None:
        raise Exception("Undefined weigths input for Conv layer")
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
        output_size.append(
            np.floor(
                ((input_dims[2:][i] + pads[i] - (dilations[i] * (kernel_shape[i] - 1)) - 1) / strides[i]) + 1
            )
        )

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
            "comparisons": 0,
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
    # prepare footprint counters
    comp_additions = 0
    mem_activations = 0

    # # collect input dimensions
    input_dims = inputs[0]["data"]["shape"]
    if input_dims == None or len(input_dims) < 1:
        input_dims = inputs[0]["data"]["identifier"].dims if inputs[0]["data"]["identifier"] != None else None
        if input_dims == None:
            raise Exception("1st input for GlobalAvgPool operation not specified!")

    output_dims = [input_dims[0], input_dims[1]]
    for i in range(len(input_dims[2:])):
        output_dims.append(1)

    outputs[0]["data"]["shape"] = output_dims
    comp_additions = np.prod(input_dims)
    mem_activations = np.prod(output_dims)

    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "comparisons": 0,
            "additions": comp_additions,
            "divisions": 0,
            "exponentials": 0
        },
        "memory": {
            "parameters": 0,
            "activations": mem_activations,
            "other_memory": 0
        }
    }

def avg_pool(inputs, outputs, attributes):
    # prepare footprint counters
    comp_additions = 0
    mem_activations = 0

    # # collect input dimensions
    input_dims = inputs[0]["data"]["shape"]
    if input_dims == None or len(input_dims) < 1:
        input_dims = inputs[0]["data"]["identifier"].dims if inputs[0]["data"]["identifier"] != None else None
        if input_dims == None:
            raise Exception("1st input for AvgPool operation not specified!")

    # should be available, otherwise take weights and return anything after the first 2 dimensions
    kernel_shape = attributes["kernel_shape"] if "kernel_shape" in attributes else None
    if kernel_shape == None:
        raise Exception("Undefined kernel shape for MaxPool layer")

    # collect dilations
    dilations = attributes["dilations"] if "dilations" in attributes else [1 for k in kernel_shape]

    # collect ceil mode
    ceil_mode = attributes["ceil_mode"] if "ceil_mode" in attributes else 0
    ceil_mode = np.ceil if ceil_mode == 1 else np.floor

    strides = attributes["strides"] if "strides" in attributes else [1 for k in kernel_shape]

    output_dims = []

    # collect pad information
    # default auto_pad is NOTSET
    # pads[] will only be defined if this is NOTSET
    pads = []
    auto_pad = attributes["auto_pad"].decode("utf-8") if "auto_pad" in attributes else "NOTSET"

    if auto_pad == "VALID":
        for i in range(len(input_dims[2:])):
            output_dims.append(np.ceil((input_dims[2:][i] - kernel_shape[i] + 1) / strides[i]))
    elif auto_pad == "SAME_UPPER" or auto_pad == "SAME_LOWER":
        for i in range(len(input_dims[2:])):
            output_dims.append(np.ceil(input_dims[2:][i] / strides[i]))
            
    elif auto_pad == "NOTSET" and "pads" in attributes:
        begin_p = attributes["pads"][:len(kernel_shape)]
        end_p = attributes["pads"][len(kernel_shape):]
        pads = np.add(begin_p, end_p).tolist()
        for i in range(len(input_dims[2:])):
            output_dims.append(ceil_mode((input_dims[2:][i] + pads[i] - kernel_shape[i]) / strides[i] + 1))

    output_dims = np.array(output_dims).astype(int).tolist()
    output_dims = [input_dims[0], input_dims[1]] + output_dims
    outputs[0]["data"]["shape"] = output_dims

    comp_additions = np.prod(output_dims[2:]) * np.prod(kernel_shape) * output_dims[0] * output_dims[1]
    mem_activations = np.prod(output_dims)

    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "comparisons": 0,
            "additions": comp_additions,
            "divisions": 0,
            "exponentials": 0
        },
        "memory": {
            "parameters": 0,
            "activations": mem_activations,
            "other_memory": 0
        }
    }

def max_pool(inputs, outputs, attributes):
    # prepare footprint counters
    comp_comparisons = 0
    mem_activations = 0

    # # collect input dimensions
    input_dims = inputs[0]["data"]["shape"]
    if input_dims == None or len(input_dims) < 1:
        input_dims = inputs[0]["data"]["identifier"].dims if inputs[0]["data"]["identifier"] != None else None
        if input_dims == None:
            raise Exception("1st input for MaxPool operation not specified!")
    
    # should be available, otherwise take weights and return anything after the first 2 dimensions
    kernel_shape = attributes["kernel_shape"] if "kernel_shape" in attributes else None
    if kernel_shape == None:
        raise Exception("Undefined kernel shape for MaxPool layer")

    # collect dilations
    dilations = attributes["dilations"] if "dilations" in attributes else [1 for k in kernel_shape]

    # collect ceil mode
    ceil_mode = attributes["ceil_mode"] if "ceil_mode" in attributes else 0
    ceil_mode = np.ceil if ceil_mode == 1 else np.floor

    strides = attributes["strides"] if "strides" in attributes else [1 for k in kernel_shape]

    output_dims = []

    # collect pad information
    # default auto_pad is NOTSET
    # pads[] will only be defined if this is NOTSET
    pads = []
    auto_pad = attributes["auto_pad"].decode("utf-8") if "auto_pad" in attributes else "NOTSET"

    if auto_pad == "VALID":
        for i in range(len(input_dims[2:])):
            output_dims.append(np.ceil((input_dims[2:][i] - ((kernel_shape[i] - 1) * dilations[i] + 1) + 1) / strides[i]))
    elif auto_pad == "SAME_UPPER" or auto_pad == "SAME_LOWER":
        for i in range(len(input_dims[2:])):
            output_dims.append(np.ceil(input_dims[2:][i] / strides[i]))
            
    elif auto_pad == "NOTSET" and "pads" in attributes:
        begin_p = attributes["pads"][:len(kernel_shape)]
        end_p = attributes["pads"][len(kernel_shape):]
        pads = np.add(begin_p, end_p).tolist()
        for i in range(len(input_dims[2:])):
            output_dims.append(ceil_mode((input_dims[2:][i] + pads[i] - ((kernel_shape[i] - 1) * dilations[i] + 1)) / strides[i] + 1))

    output_dims = np.array(output_dims).astype(int).tolist()
    output_dims = [input_dims[0], input_dims[1]] + output_dims
    outputs[0]["data"]["shape"] = output_dims

    comp_comparisons = np.prod(output_dims[2:]) * np.prod(kernel_shape) * output_dims[0] * output_dims[1]
    mem_activations = np.prod(output_dims)

    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "comparisons": comp_comparisons,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory": {
            "parameters": 0,
            "activations": mem_activations,
            "other_memory": 0
        }
    }

def identity(inputs, outputs, attributes):
    # prepare footprint counters
    mem_activations = 0
    
    # # collect input dimensions
    input_dims = inputs[0]["data"]["shape"]
    if input_dims == None or len(input_dims) < 1:
        input_dims = inputs[0]["data"]["identifier"].dims if inputs[0]["data"]["identifier"] != None else None
        if input_dims == None:
            raise Exception("1st input for Identity operation not specified!")
    else:
        outputs[0]["data"]["shape"] = input_dims
        mem_activations = np.prod(input_dims)

    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "comparisons": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory": {
            "parameters": 0,
            "activations": mem_activations,
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

    # # collect input dimensions
    input_shape = inputs[0]["data"]["shape"]
    if input_shape == None or len(input_shape) < 1:
        input_shape = inputs[0]["data"]["identifier"].dims if inputs[0]["data"]["identifier"] != None else None
        if input_shape == None:
            raise Exception("1st input for Pad operation not specified!")

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
            "comparisons": 0,
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
    mem_activations = 0

    # # collect input dimensions
    input_dims = inputs[0]["data"]["shape"]
    if input_dims == None or len(input_dims) < 1:
        input_dims = inputs[0]["data"]["identifier"].dims if inputs[0]["data"]["identifier"] != None else None
        if input_dims == None:
            raise Exception("1st input for Relu operation not specified!")

    alloc_input = np.zeros(input_dims)
    reshape = numpy_helper.to_array(inputs[1]["data"]["identifier"]).tolist()
    for i in range(len(reshape)):
        if reshape[i] == 0:
            reshape[i] = input_dims[i]

    reshaped = list(np.reshape(alloc_input, reshape).shape)
    outputs[0]["data"]["shape"] = reshaped
    mem_activations = np.prod(reshaped)

    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "comparisons": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory": {
            "parameters": 0,
            "activations": mem_activations,
            "other_memory": 0
        }
    }


def transpose(inputs, outputs, attributes):
    # https://www.researchgate.net/publication/273912700_In-Place_Matrix_Transposition_on_GPUs
    # Transpose has no computation measured as it only changes the order of channels (usually in place)
    # Transpose will have a singular tensor as its input
    # Transpose will have a singular tensor as its output
        # collect input information
    shape = inputs[0]["data"]["shape"]

    if shape == None or len(shape) < 1:
        shape = inputs[0]["data"]["identifier"].dims if inputs[0]["data"]["identifier"] != None else None
        if shape == None:
            raise Exception("1st input for Relu operation not specified!")

    perms = attributes["perm"]

    
    # by default reverse dimensions
    if perms == None or len(perms) < 1:
        perms = list(reversed(shape))

    if len(shape) == len(perms):
        out_shape = []
        for p in perms:
            out_shape.append(shape[p])
        outputs[0]["data"]["shape"] = out_shape

    else:
        raise Exception("Shape doesn't match Permutations for Transpose operation")
        
    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "comparisons": 0,
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
    # prepare footprint counters
    comp_comparisons = 0
    mem_activations = 0

    # collect input information
    input_dims = inputs[0]["data"]["shape"]
    if input_dims == None or len(input_dims) < 1:
        input_dims = inputs[0]["data"]["identifier"].dims if inputs[0]["data"]["identifier"] != None else None
        if input_dims == None:
            raise Exception("1st input for Relu operation not specified!")

    # data dimension doesn't change
    outputs[0]["data"]["shape"] = input_dims
    
    # calculate footprint
    comp_comparisons = np.prod(input_dims)
    mem_activations = comp_comparisons

    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "comparisons": comp_comparisons,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory": {
            "parameters": 0,
            "activations": mem_activations,
            "other_memory": 0
        }
    }

def concat(inputs, outputs, attributes):
    mem_activations = 0
    inputs_shapes = []

    # # collect input dimensions
    for i in range(len(inputs)):
        id = inputs[i]["data"]["shape"]
        if id == None or len(id) < 1:
            id = inputs[i]["data"]["identifier"].dims if inputs[i]["data"]["identifier"] != None else None
            if id == None:
                raise Exception("1st input for Concat operation not specified!")
        else:
            inputs_shapes.append(id)

    # check on which axis to concatenate
    axis = attributes["axis"] if "axis" in attributes else 1


    # check if dimensions after the axis are the same
    for i in range(len(inputs_shapes)):
        for j in range(len(inputs_shapes[i][axis + 1:])):
            if i == 0:
                continue
            elif inputs_shapes[i][axis + 1:][j] != inputs_shapes[i - 1][axis + 1:][j]:
                pprint(inputs)
                print(axis)
                print(inputs_shapes)
                raise Exception("Not matching dimensions after the Concat axis!")
        
        for j in range(len(inputs_shapes[i][:axis])):
            if inputs_shapes[i][j] != inputs_shapes[i - 1][j]:
                raise Exception("Not matching dimensions before the Concat axis!")
    
    sum = 0
    for i in range(len(inputs_shapes)):
        sum += inputs_shapes[i][axis]

    output_dims = inputs_shapes[0]
    output_dims[axis] = sum
    
    outputs[0]["data"]["shape"] = output_dims
    mem_activations = np.prod(output_dims)

    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "comparisons": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory": {
            "parameters": 0,
            "activations": mem_activations,
            "other_memory": 0
        }
    }

def add(inputs, outputs, attributes):
    comp_additions = 0
    # collect input information
    a = inputs[0]["data"]["shape"]
    if a == None or len(a) < 1:
        a = inputs[0]["data"]["identifier"].dims if inputs[0]["data"]["identifier"] != None else None
        if a == None:
            raise Exception("1st input for Add operation not specified!")

    b = inputs[1]["data"]["shape"]
    if b == None or len(b) < 1:
        b = inputs[1]["data"]["identifier"].dims if inputs[1]["data"]["identifier"] != None else None
        if b == None:
            raise Exception("2st input for Add operation not specified!")
    
    

    max_dims = max([a, b], key=len)
    min_dims = min([a, b], key=len)

    r_a = list(reversed(a))
    r_b = list(reversed(b))

    for i in range(len(min_dims)):
        if r_a[i] != r_b[i] and (r_a[i] != 1 and r_b[i] != 1):
            raise Exception("Incorrect dimensions for N-dimensional matrix multiplication")

    outputs[0]["data"]["shape"] = max_dims
    comp_additions = np.prod(max_dims)

    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "comparisons": 0,
            "additions": comp_additions,
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
    comp_multiply_adds = 0
    comp_flops = 0
    # collect input information
    a = inputs[0]["data"]["shape"]
    if a == None or len(a) < 1:
        a = inputs[0]["data"]["identifier"].dims if inputs[0]["data"]["identifier"] != None else None
        if a == None:
            raise Exception("1st input for Mul operation not specified!")

    b = inputs[1]["data"]["identifier"].dims

    max_dims = max([a, b], key=len)
    min_dims = min([a, b], key=len)

    r_a = list(reversed(a))
    r_b = list(reversed(b))

    for i in range(len(min_dims)):
        if r_a[i] != r_b[i] and (r_a[i] != 1 and r_b[i] != 1):
            raise Exception("Incorrect dimensions for N-dimensional matrix multiplication")

    outputs[0]["data"]["shape"] = max_dims
    comp_multiply_adds = np.prod(max_dims)
    comp_flops = comp_multiply_adds * 2

    return {
        "operations": {
            "flops": comp_flops,
            "multiply_adds": comp_multiply_adds,
            "comparisons": 0,
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
    comp_flops = 0
    comp_multiply_adds = 0
    mem_activations = 0
    # collect input information
    a = inputs[0]["data"]["shape"]
    if a == None or len(a) < 1:
        a = inputs[0]["data"]["identifier"].dims if inputs[0]["data"]["identifier"] != None else None
        if a == None:
            raise Exception("1st input for MatMul operation not specified!")

    b = inputs[1]["data"]["identifier"].dims
    
    a_matmul = a[len(a) - 2:]
    b_matmul = b[len(b) - 2:]

    if a_matmul[1] != b_matmul[0]:
        raise Exception("Incorrect dimensions for N-dimensional matrix multiplication (proper matrix link doesn't match)")

    out_dim_mat_mul = [a_matmul[0], b_matmul[1]]

    a_matmul_before = a[:len(a) - 2]
    b_matmul_before = b[:len(b) - 2]

    max_before_dims = max([a_matmul_before, b_matmul_before], key=len)
    min_before_dims = min([a_matmul_before, b_matmul_before], key=len)

    r_a_matmul_before = list(reversed(a_matmul_before))
    r_b_matmul_before = list(reversed(b_matmul_before))

    for i in range(len(min_before_dims)):
        if r_a_matmul_before[i] != r_b_matmul_before[i] and (r_a_matmul_before[i] != 1 and r_b_matmul_before[i] != 1):
            raise Exception("Incorrect dimensions for N-dimensional matrix multiplication")

    outputs[0]["data"]["shape"] = max_before_dims + out_dim_mat_mul

    comp_flops = (2 * a_matmul[0] * a_matmul[1] * b_matmul[1]) - (a_matmul[0] - b_matmul[1])
    comp_multiply_adds = np.ceil(comp_flops / 2)
    mem_activations = np.prod(max_before_dims + out_dim_mat_mul)

    return {
        "operations": {
            "flops": comp_flops,
            "multiply_adds": comp_multiply_adds,
            "comparisons": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory": {
            "parameters": 0,
            "activations": mem_activations,
            "other_memory": 0
        }
    }


def softmax(inputs, outputs, attributes):
    comp_exponentials = 0
    comp_additions = 0
    comp_divisions = 0
    mem_activations = 0

    # collect input information
    a = inputs[0]["data"]["shape"]
    if a == None or len(a) < 1:
        a = inputs[0]["data"]["identifier"].dims if inputs[0]["data"]["identifier"] != None else None
        if a == None:
            raise Exception("1st input for Softmax operation not specified!")

    # check on which axis to concatenate
    axis = attributes["axis"] if "axis" in attributes else 1
    # coerce input to 2D matrix
    b = a[:axis]
    b = np.prod(b).astype(int)
    c = a[axis:]
    c = np.prod(c).astype(int)
    out_dims = [b, c]
    comp_exponentials = comp_additions = comp_divisions = np.prod(a[1:]) * out_dims[0]
    mem_activations = np.prod(out_dims)
    outputs[0]["data"]["shape"] = out_dims

    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "comparisons": 0,
            "additions": comp_additions,
            "divisions": comp_divisions,
            "exponentials": comp_exponentials
        },
        "memory": {
            "parameters": 0,
            "activations": mem_activations,
            "other_memory": 0
        }
    }

def _slice(inputs, outputs, attributes):
    mem_activations = 0

    # collect input information
    a = inputs[0]["data"]["shape"]
    if a == None or len(a) < 1:
        a = inputs[0]["data"]["identifier"].dims if inputs[0]["data"]["identifier"] != None else None
        if a == None:
            raise Exception("1st input for Softmax operation not specified!")

    starts = numpy_helper.to_array(inputs[1]["data"]["identifier"]).tolist() if inputs[1]["data"]["identifier"] != None else None
    if starts == None:
        raise Exception("2nd input for Slice operation not specified!")

    ends = numpy_helper.to_array(inputs[2]["data"]["identifier"]).tolist() if inputs[2]["data"]["identifier"] != None else None
    if ends == None:
        raise Exception("3nd input for Slice operation not specified!")
    
    steps = [1 for x in range(len(starts))]
    if len(inputs) > 4 and inputs[4]["data"]["identifier"] != None:
        steps = numpy_helper.to_array(inputs[4]["data"]["identifier"]).tolist()

    axes = [x for x in range(len(a))]
    if len(inputs) > 3 and inputs[3]["data"]["identifier"] != None:
        axes = numpy_helper.to_array(inputs[3]["data"]["identifier"]).tolist()

    # prepare input data shape
    x = np.zeros(a)

    ses = []
    for i in range(len(starts)):
        ses.append([starts[i], ends[i], steps[i]])

    indices = {}
    for i, axis in enumerate(axes):
        # normalize negative axes
        ax = axis % len(a)
        indices[ax] = slice(ses[i][0], ses[i][1], ses[i][2])

    ix = [indices.get(dim, slice(None)) for dim in range(x.ndim)]
    sliced = x[tuple(ix)]

    outputs[0]["data"]["shape"] = list(sliced.shape)

    mem_activations = np.prod(list(sliced.shape))

    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "comparisons": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory": {
            "parameters": 0,
            "activations": mem_activations,
            "other_memory": 0
        }
    }


def flatten(inputs, outputs, attributes):
    mem_activations = 0

    # collect input information
    a = inputs[0]["data"]["shape"]
    if a == None or len(a) < 1:
        a = inputs[0]["data"]["identifier"].dims if inputs[0]["data"]["identifier"] != None else None
        if a == None:
            raise Exception("1st input for Softmax operation not specified!")
    
    axis = attributes["axis"] if "axis" in attributes else 1

    # coerce input to 2D matrix
    b = a[:axis]
    b = np.prod(b).astype(int)
    c = a[axis:]
    c = np.prod(c).astype(int)
    out_dims = [b, c]
    mem_activations = np.prod(out_dims)
    outputs[0]["data"]["shape"] = out_dims

    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "comparisons": 0,
            "additions": 0,
            "divisions": 0,
            "exponentials": 0
        },
        "memory": {
            "parameters": 0,
            "activations": mem_activations,
            "other_memory": 0
        }
    }


def clip(inputs, outputs, attributes):
    comp_comparisons = 0

    # collect input information
    a = inputs[0]["data"]["shape"]
    if a == None or len(a) < 1:
        a = inputs[0]["data"]["identifier"].dims if inputs[0]["data"]["identifier"] != None else None
        if a == None:
            raise Exception("1st input for Clip operation not specified!")
    
    outputs[0]["data"]["shape"] = a
    comp_comparisons = np.prod(a)

    return {
        "operations": {
            "flops": 0,
            "multiply_adds": 0,
            "comparisons": comp_comparisons,
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
