
# retrieves identifier type by name
def convert_tensor_shape_to_array(t):
    arr = []
    for d in t.type.tensor_type.shape.dim:
        if d.dim_value != 0:
            arr.append(d.dim_value)
        else:
            arr.append(d.dim_param)
    return arr

def flatten_dict(init, lkey=''):
    ret = {}
    for rkey,val in init.items():
        key = lkey+rkey
        if isinstance(val, dict):
            ret.update(flatten_dict(val, key+'_'))
        else:
            ret[key] = val
    return ret
