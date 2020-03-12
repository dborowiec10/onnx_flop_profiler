import re

def parse_initializer(line):
    line = line.replace("%", "")
    open_idx = line.find("[")
    close_idx = line.find("]")
    left = line[:open_idx].strip()
    _type = None
    subtype = None
    tl_split = left.split("/")
    if len(tl_split) > 1:
        name = tl_split[0]
        _type = tl_split[1]
        ttypel_split = _type.split(":")
        if len(ttypel_split) > 1:
            _type = ttypel_split[0]
            subtype = ttypel_split[1]
    else:
        name = tl_split[0]
    terms = line[open_idx+1:close_idx].split(", ")
    return {
        "name": name,
        "type": _type,
        "subtype": subtype,
        "datatype": terms[0],
        "shape": tuple(terms[1].split("x"))
    }


def find_initializer(name):
    for i in initializers:
        if i["name"] == name:
            return i
    return None


def parse_main(line):
    inputs = []
    output = {
        "name": None,
        "variable": None,
        "type": None
    }
    arguments = []
    line = line.replace("%", "")
    left = line.split("=", 1)[0].strip()
    right = line.split("=", 1)[1].strip()

# do left side (outputs)
    left_terms = left.split("/")
    if len(left_terms) > 1:
        output["name"] = left_terms[0]
        output["variable"] = left_terms[1].split(":")[0]
        output["type"] = left_terms[1].split(":")[1] if len(left_terms[1].split(":")) > 1 else None
    else:
        output["name"] = left_terms[0]

# do right side (op/inputs/arguments)
    arg_start_idx = right.find("[")
    arg_end_idx = right.rfind("]")
    args = None
    op_name = None
    if arg_start_idx != -1 and arg_end_idx != -1:
        args = right[arg_start_idx+1:arg_end_idx]
        op_name = right[:arg_start_idx]

    op_inp_start_idx = right.find("(")
    op_inp_end_idx = right.rfind(")")
    op_inputs = right[op_inp_start_idx+1:op_inp_end_idx]
# do op
    if op_name == None:
        op_name = right[:op_inp_start_idx]
# do inputs
    op_inputs_splits = op_inputs.split(", ")
    for op_in in op_inputs_splits:
        if len(op_in.split("/")) == 1:
            inputs.append(op_in)
        else:
            spl = op_in.split("/")
            inputs.append({
                "name": spl[0],
                "variable": spl[1].split(":")[0],
                "type": spl[1].split(":")[1]
            })
# do arguments
    argsx = []
    prev_idx = 0
    if args != None:
        for m in re.finditer(r',\s[a-z]', args):
            ret = args[prev_idx:m.span()[0]]
            prev_idx += (2 + len(ret))
            argsx.append(ret.replace("'", "").replace("\"", ""))

    for a in argsx:
        l_arg = a.split(" = ")[0]
        r_arg = a.split(" = ")[1]

        if "[" in r_arg and "]" in r_arg:
            r_arg = r_arg[r_arg.find("[") + 1:r_arg.rfind("]")].split(", ")
            r_arg = [int(x) for x in r_arg]
            arguments.append({
                l_arg.strip(): r_arg  
            })
        else:
            arguments.append({
                l_arg.strip(): r_arg.strip()   
            })
    return {
        "output": output,
        "inputs": inputs,
        "operation": op_name,
        "arguments": arguments
    }

def get_return(line):
    return line.split(" %")[1]

def parse_graph_input(line):
    line = line.replace("%", "")
    open_idx = line.find("[")
    close_idx = line.find("]")
    input_name = line[:open_idx].strip()
    terms = line[open_idx+1:close_idx].split(", ")
    return {
        "name": input_name,
        "datatype": terms[0],
        "shape": tuple(terms[1].split("x"))
    }

def parse(filename):
    graph_lines = []

    graph_name = ""
    _input = []
    _return = None
    initializers = []
    _main = []

    inits_start_idx = -1
    inits_end_idx = -1
    with open(filename, "r") as _file:
        graph_lines = _file.readlines()

    for k, l in enumerate(graph_lines):
        line = l.rstrip()
        if not line.startswith("%"):
            if "graph" in line:
                graph_name = line.split(" ")[1]
                continue
            elif "initializers" in line:
                inits_start_idx = k
                continue
            elif ") {" in line:
                inits_end_idx = k
                inits_start_idx = -1
                continue
            elif "}" in line:
                continue
            elif "return" in line:
                _return = get_return(line)
                continue
        if k == 1:
            _input = parse_graph_input(line)
            continue
        if k > inits_start_idx and inits_end_idx == -1:
            parsed = parse_initializer(line)
            found = None
            for i in initializers:
                if i["name"] == parsed["name"]:
                    found = i
                    break
            if found:
                found["inner"].append({
                    "variable": parsed["type"],
                    "datatype": parsed["datatype"],
                    "shape": parsed["shape"],
                    "type": parsed["subtype"]
                })
            else:
                initializers.append({
                    "name": parsed["name"],
                    "inner": [{
                        "variable": parsed["type"],
                        "datatype": parsed["datatype"],
                        "shape": parsed["shape"],
                        "type": parsed["subtype"]
                    }]
                })
            continue
        if k > inits_end_idx and inits_start_idx == -1:
            _main.append(parse_main(line))
            continue
    
    return {
        "name": graph_name,
        "input": _input,
        "return": _return,
        "initializers": initializers,
        "operations": _main
    }