import numpy as np
from random import randint, random
from statistics import mode
# File names
in_name = 'cat.png'
out_name = 'cat2.png'
bw_name = 'blackcat.png'
from PIL import Image
import math as m
import cmath


def handle_redundancy(input_bit, k, entangle, function):
    output_bits = []
    for j in range(2*k+1):
        output_bits.append(function(input_bit, entangle))
    return mode(output_bits)

class State:
    def __init__(self, probs):
        # We dont care about amplitudes and we wont consider any mixes, so we consider only probabilities
        self.probs = probs
        self.observed = False
    def observe(self):
        # Returns the index of the state that the particle is found in. Eigenvalues corresponding to observables
        # must be dealt with separately.
        if self.observed:
            return self.observedvalue
        else:
            rand = random()
            summation = 0
            for (index,p) in enumerate(self.probs):
                summation += p
                if rand < summation:
                    self.observed = True
                    self.observedvalue = index
                    return index

# This is just an arithmatical truth, given that the maximimally mixed state is the only infomration passed through.
# This yeilds no information at all.
def single_M_channel(input_bit):
    zeroprob = 1 if input_bit == 0 else 0
    state = State([zeroprob, 1 - zeroprob])
    # X = State([1/3 if zeroprob == 0 else 2/3, 2/3 if zeroprob == 0 else 1/3])
    N1_Out = State([1/2,1/2]) # Max Mixed state.
    N1 = N1_Out.observe()
    if N1 == 0:
        Ent = State([2/3, 1/6, 1/6, 0])
    else:
        Ent = State([0,1/6,1/6, 2/3])

    Ent = Ent.observe()
    if Ent < 2:
        return input_bit
    else:
        return 0 if input_bit != 0 else 255


    # The single M channel. whcih will have no capacity.
def bw_M_channel(filename):
    im = Image.open(filename) # Can be many different formats.
    pix = im.load()
    dimension = im.size
    print(dimension)
    for i in range(dimension[0]):
        for j in range(dimension[1]):
            pix[i,j] = single_M_channel(pix[i,j])
    # final = Image.fromarray(pix)
    im.save('single_quantum_blackcat.png')
    # Image.save('result.png')



# We will see that for certain values of entanglement, this is conserved.
def double_M_channel(input_bit, entangle):
    # This is an emulator of the channel action on a bit.
    zeroprob = 1 if input_bit == 0 else 0
    # This statistical formulation is equivalent mathematically. (But not physically.)
    state = State([zeroprob, 1 - zeroprob])
    probfixed = State([1/2,1/2])
    maxent = State([1/4,1/4,1/4,1/4])
    Entangle2 = State([entangle, 1- entangle])
    ent2 = Entangle2.observe()
    if probfixed.observe() == 0:
        item_val = ent2
    else:
        item_val = maxent.observe() % 2
    if item_val == 0:
        return input_bit
    else:
        return 255 if input_bit == 0 else 0


def bw_2M_channel(filename, entangle=0.125, k = 0):
    im = Image.open(filename)
    pix = im.load()
    dimension = im.size
    totaleq = 0
    for i in range(dimension[0]):
        for j in range(dimension[1]):
            temp = pix[i,j]
            temp2 = handle_redundancy(pix[i,j], k, entangle, double_M_channel)
            pix[i,j] = temp2
            if temp == temp2:
                totaleq += 1
    print(totaleq/(dimension[1] * dimension[0]))
    im.save("quantum_blackcat.png")

if __name__ == "__main__":
    # bw_M_channel("blackcat.png")
    bw_2M_channel("blackcat.png")
