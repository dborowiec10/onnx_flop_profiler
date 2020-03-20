
def batch_norm(inputs, outputs, attributes):
    print("BatchNormalization")
    return 0

def clip(inputs, outputs, attributes):
    print("Clip")
    return 0

def conv(inputs, outputs, attributes):
    print("Conv")
    return 0

def global_avg_pool(inputs, outputs, attributes):
    print("GlobalAveragePool")
    return 0

def avg_pool(inputs, outputs, attributes):
    print("AveragePool")
    return 0

def max_pool(inputs, outputs, attributes):
    print("MaxPool")
    return 0

def identity(inputs, outputs, attributes):
    print("Identity")
    return 0

def pad(inputs, outputs, attributes):
    print("Pad")
    return 0

def reshape(inputs, outputs, attributes):
    print("Reshape")
    return 0

def softmax(inputs, outputs, attributes):
    print("Softmax")
    return 0

def transpose(inputs, outputs, attributes):
    print("Transpose")
    return 0

def relu(inputs, outputs, attributes):
    print("Relu")
    return 0

def concat(inputs, outputs, attributes):
    print("Concat")
    return 0

def add(inputs, outputs, attributes):
    print("Add")
    return 0

def mul(inputs, outputs, attributes):
    print("Mul")
    return 0

def mat_mul(inputs, outputs, attributes):
    print("MatMul")
    return 0

def _slice(inputs, outputs, attributes):
    print("Slice")
    return 0

def flatten(inputs, outputs, attributes):
    print("Flatten")
    return 0


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