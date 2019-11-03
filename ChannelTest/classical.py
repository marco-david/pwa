import numpy as np
from random import randint
from statistics import mode
import os
# File names
in_name = 'cat.png'
out_name = 'cat2.png'
bw_name = 'blackcat.png'

from PIL import Image 

def convert_to_bw(filename,outname):
    image_file = Image.open(filename) # open colour image
    image_file = image_file.convert('1') # convert image to black and white
    image_file.save(outname)
# Read data and convert to a list of bits
# Convert the list of bits back to bytes and save


def handle_redundancy(input_bit, k, probability, function):
    output_bits = []
    for j in range(2*k+1):
        output_bits.append(function(input_bit, probability))
    return mode(output_bits)


def randomize_bw(input_cell, probability):
    if randint(0,100) < probability:
        if input_cell > 0:
            return 0
        else:
            return 255
    else:
        return input_cell

def bw_channel(filename, probability=20, k=2, outname="cats/traing/{k}"):
    im = Image.open(filename) # Can be many different formats.
    pix = im.load()
    dimension = im.size
    print(dimension)
    for i in range(dimension[0]):
        for j in range(dimension[1]):
            pix[i,j] = handle_redundancy(pix[i,j], k, probability,randomize_bw)

    # final = Image.fromarray(pix)
    im.save(outname)
    # Image.save('result.png')
    # pix[x,y] = value 

if __name__ == "__main__":
    for k in os.listdir('cats'):
        # convert_to_bw(f"cats/{k}", f"cats/{k}")

        for j in range(1,3):
            bw_channel(f"cats/{k}", outname=f"cats/train/{j}-{k}", k=0, probability=j)

        





    # convert_to_bw(in_name)
    # prob = 0
    # while prob <= 100:
    #     for k in range(6):

    #         bw_channel(bw_name, probability=prob, k=k)
    #      # bw_channel(bw_name)
    #     prob += 10
    # data = read_file(in_name)
    # channel(data)

