import numpy as np
from random import randint
# File names
in_name = 'cat.png'
out_name = 'cat2.png'
bw_name = 'blackcat.png'
from PIL import Image 






def bw_channel(filename, probability=30, k=2):
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





