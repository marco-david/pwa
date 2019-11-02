import numpy as np
from random import randint
# File names
in_name = 'cat.png'
out_name = 'cat2.png'

# Read data and convert to a list of bits
def read_file(filename):
    in_bytes = np.fromfile(filename, dtype = "uint8")
    in_bits = np.unpackbits(in_bytes)
    data = list(in_bits)
    print(len(data))
    return data

# Convert the list of bits back to bytes and save
def write_bits(filename,data):
    out_bits = np.array(data)
    # print(np.all(out_bits == in_bits))
    out_bytes = np.packbits(out_bits)
    # print(np.all(out_bytes == in_bytes))
    out_bytes.tofile(out_name)


def duplicate_bytes(input_data, k=2):
    new_bytes = []
    for b in input_data:
        new_bytes.append(b)
        new_bytes.append(b)
    return new_bytes


def bitflipper(b, errorpercent):
    number = randint(0, 100)
    if number < errorpercent:
        if b == 1:
            return 0
        else:
            return 1
    else:
        return b

def channel(input_data, errorpercent = 0, duplicates = 2):
    # no duplication implemented yet. 
    output_data = []
    for (index, b) in enumerate(input_data):
        if index < 64*100:
            output_data.append(b)
        else:
            output_data.append(bitflipper(b, errorpercent))
    
    write_bits("outputcat.png", output_data)
    









if __name__ == "__main__":
    data = read_file(in_name)
    channel(data)





