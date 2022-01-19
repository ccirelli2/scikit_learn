#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tutorial on Entropy
    https://machinelearningmastery.com/what-is-information-entropy/
    https://towardsdatascience.com/entropy-how-decision-trees-make-decisions-2946b9c18c8

    A cornerstone of information theory is the idea of quantifying how much 
    information is in a message. 
    
    More generally, this can be used to quantify the information in an 
    event and a random variable, called entropy, and is calculated using
    probability. 
    
    Learning Objectives:
        Information theory is concerned with data compression and transmission
        and builds on probability. 
        Information provide sa way to quantify the amount of surprise for an 
        event measured in bits. 
        Entropy provides a measure of the average amount of information
        needed to represent an event drawan from a probability distribution 
        for a random variable. 

    There is an import relationship between information theory and ML as 
    many ML models, decision trees, use entropy to fit their models. 
    
    The idea behind quantifying information is the idea of measuring how much
    surprise there is an event. Rare events have low prob while likely have high prob. 
    Rare events require more information to represent them thatn freq ones. 
    
    
    Shannon Information - used to calculate the amount of info in an event. 
    info(x) = -log(p(x)), where log is base 2 and p(x) is the probability
    of the event.  Base 2 means that the units of info measure in bits. 
    
    The negative sign ensures that the result is always positive or 0. 
    
    Information will be zero when the probability of the event is 1.0 or
    certain. 

"""
# Modules
import math as m
from matplotlib import pyplot
from math import log2

# Example of Taking a Log 
def ex_log():
    print(m.log(100, 10))
    print(m.log2(100))
    print(m.log2(16))
    
# Developing the Idea that Low Prob Events have more information

# Calculate the information of flipping a coin
p_x = 0.5
h_x = -m.log2(p_x)
#print('Probability of an even => {}, Information => {}'.format(p_x, h_x))


# What if the coin wasn't fair and had a prob of heads =10%?
'Note: Now the bits increased to 3.3'
p_x2 = 0.1
h_x2 = -m.log2(p_x2)
#print('Information = {}'.format(h_x2))  


# Relationship between probability and information
def plot_rel_prob_info():
    # list of probabilities
    probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # calculate information
    info = [-m.log2(p) for p in probs]
    # plot probability vs information
    pyplot.plot(probs, info, marker='.')
    pyplot.title('Probability vs Information')
    pyplot.xlabel('Probability')
    pyplot.ylabel('Information')
    pyplot.show()


# Calculate the Entropy for a Random Variable
''' Called Shannon Entropy or information entropy
    It is the average number of bits requried to represent or
    transmit an event drawn from the probability distribution for the
    random variable. 
    
    Entropy can be calculated for a random variable X with k in K discrete
    states
    H(X) = -sum(each k in K p(k) * log(p(k)))
    
    That is, the negative of the sum of the probability of each event multiplied
    by the log of the probability of each event. 

    Note:  the lowest entropy is calculated for a random variable that has a
    single event with a probability of 1.0, or a certainty. 
    
    The largest entropy for a random variable will be if all events are equally
    likely. 

'''


n = 6               # number of events
p = 1.0 / n         # probability of one event
entropy = -sum([p * log2(p) for m in range(n)])
print('Entropy => {}'.format(entropy))

'''Note:  This is the same entropy that we obtained from a single event w/ 
          a probability of 1/6. See below 
'''
def entropy_single_event():
    p = 1/6
    print(-log2(p))
    [print(p * log2(p)) for m in range(n)]



# Scipy Entropy Function
'Used when we know the probability of each event'
from scipy.stats import entropy
# discrete probabilities
p = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
# calculate entropy
e = entropy(p, base=2)
# print the result
print('entropy: %.3f bits' % e)




# CALCULATING ENTROPY FOR TWO CLASSES
''' Say we had two classes and 100 data points. 
    p(c1) = 0.3 and p(c2) = 0.7
'''

def entropy_two_vars():
    entropy = (-0.3 * log2(0.3) -0.7*log2(0.7))
    print(entropy)

'''We get a very high entropy, which means that we have a significant
   level of impurity or disorder in our data. 
   Note: entropy is always measured between 0 and 1. 
         Entropy is lowest at the extremes, i.e. when a set of data
         comprises 100 percent of one class or another. 
'''
def graph_log2():
    log2_p = []
    p      = []
    for i in np.arange(0.1, 1, 0.1):
        #entropy = round(-1 * i * log2(i), 2)
        #print('Probability => {}, Entropy => {}'.format(i, entropy))
        log2_p.append(log2(i))
        p.append(i)
        
    plt.plot(log2_p)
    plt.plot(p)
    plt.show()


def graph_entropy_two_vars():
    entropy_list = []
    prob = []
    for i in np.arange(0.1, 1, 0.1):
        p1 = i
        p2 = round(1-i,2)
        
        entropy = round(-p1 * log2(p1) -p2 * log2(p2), 2)
        entropy_list.append(entropy)
        prob.append(i)
        
        print('P1 => {}, P2 => {}, H => {}'.format(p1, p2, entropy))
    plt.plot(entropy_list)
    plt.plot(prob)
        
    
# HOW TO MEASURE REDUCTION IN DISORDER IN OUR TARGET CLASS GIVEN ADDITIONAL INFO
'''Calculation 
        IG(Y, X) = E(Y) - E(Y|X)
        
        Subtract the entropy of Y given X from the entropy of Y, which
        calculates the reduction in the uncertainty about Y. This is 
        called information gain. 
        The greater the reduction in uncertainty, the more information is
        gained about Y from X. 
'''
        
        
    
    
    