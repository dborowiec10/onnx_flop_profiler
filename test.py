import numpy as np
import codecs, json 
import onnx

x = np.random.randint(1, 50, [1, 9, 224, 230]).astype(np.float32)
W = np.ones([64, 3, 7, 7]).astype(np.float32)

print(x)
print(W)

node = onnx.helper.make_node(
        'Conv',
        inputs=['x','W'],
        outputs=['y'],
        kernel_shape=[7,7],
        pads=[0,0,0,0],
        group=1,
        strides=[2,2],
        dilations=[1,1]
)

print(node(x, W))



# dilations = [1, 1]
# kernel_shape = [7, 7]
# group = 1
# strides = [2, 2]




# print(np.prod([1, 3, 224, 224]))

# one = np.ones([2, 3, 10])
# padded = np.pad(one, [(0, 0), (3,3), (2,2)], mode="constant", constant_values=0.0)
# print(one.shape)
# print(padded.shape)

# b = padded.tolist()
# c = one.tolist()
# json.dump(b, codecs.open("dump_after.json", 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
# json.dump(c, codecs.open("dump_before.json", 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)


# one = np.ones([1, 3, 224, 224])
# padded = np.pad(one, [(0, 0), (3,3), (0,0), (3,3)], mode="constant", constant_values=0.0)
# print(padded.shape)
# b = padded.tolist()
# file_path = "dump.json"
# json.dump(b, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
