import numpy as np
from random import randint
# File names
in_name = 'cat.png'
out_name = 'cat2.png'
bw_name = 'blackcat.png'

from PIL import Image 


def convert_to_bw(filename):
    image_file = Image.open(filename) # open colour image
    image_file = image_file.convert('1') # convert image to black and white
    image_file.save('blackcat.png')
# Read data and convert to a list of bits
# Convert the list of bits back to bytes and save


def randomize_bw(input_cell, probability):
    if randint(0,100) < probability:
        if input_cell > 0:
            return 0
        else:
            return 255
    else:
        return input_cell

def bw_channel(filename, probability=45, k=2):
    im = Image.open(filename) # Can be many different formats.
    pix = im.load()
    dimension = im.size
    print(dimension)
    for i in range(dimension[0]):
        for j in range(dimension[1]):
            pix[i,j] = randomize_bw(pix[i,j], probability)

    # final = Image.fromarray(pix)
    im.save('result.png')
    # Image.save('result.png')
    # pix[x,y] = value 

if __name__ == "__main__":
    # convert_to_bw(in_name)

    bw_channel(bw_name)
    # data = read_file(in_name)
    # channel(data)





