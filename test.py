import numpy as np
import codecs, json 

one = np.ones([2, 3, 10])
padded = np.pad(one, [(0, 0), (3,3), (2,2)], mode="constant", constant_values=0.0)
print(one.shape)
print(padded.shape)

b = padded.tolist()
c = one.tolist()
json.dump(b, codecs.open("dump_after.json", 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
json.dump(c, codecs.open("dump_before.json", 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)


# one = np.ones([1, 3, 224, 224])
# padded = np.pad(one, [(0, 0), (3,3), (0,0), (3,3)], mode="constant", constant_values=0.0)
# print(padded.shape)
# b = padded.tolist()
# file_path = "dump.json"
# json.dump(b, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)