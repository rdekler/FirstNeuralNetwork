

import numpy
import scipy.special

# class for a three layer neural network
# first layer input nodes
# second layer hidden nodes
# third layer output nodes

class NeuralNetwork :

    def __init__(self , innodes , hidnodes , outnodes , alpha):
        self.innodes = innodes
        self.hidnodes = hidnodes
        self.outnodes = outnodes

        self.alpha = alpha

        # weight initialization
        # w_ij - i = rows , j = columns
        # ih = input to hidden
        # ho = hidden to output
        self.wih = numpy.random.normal(0.0 , pow(self.outnodes , -0.5) , (self.hidnodes , self.innodes))
        self.who = numpy.random.normal(0.0 , pow(self.hidnodes , -0.5) , (self.outnodes , self.hidnodes))

        self.activation_function = lambda x : scipy.special.expit(x)

    def train(self , inlist , target):
        # inputs to 2d array
        inputs = numpy.array(inlist , ndmin=2).T
        targets = numpy.array(target , ndmin=2).T

        # signals to hidden layer
        hidin = numpy.dot(self.wih , inputs)
        # signals from hidden layer
        hidout = self.activation_function(hidin)

        # signals to final output
        finin = numpy.dot(self.who , hidout)
        # signals from final output
        finout = self.activation_function(finin)

        outerr = targets - finout
        hiderr = numpy.dot(self.who.T , outerr)

        # update weights
        self.wih += self.alpha * numpy.dot((hiderr * hidout * (1.0 - hidout)), numpy.transpose(inputs))
        self.who += self.alpha * numpy.dot((outerr * finout * (1.0 - finout)) , numpy.transpose(hidout))


    def query(self , inlist):
        # list to 2d array
        inputs = numpy.array(inlist , ndmin=2).T

        # signals to hidden layer
        hidin = numpy.dot(self.wih , inputs)
        # signals from hidden layer
        hidout = self.activation_function(hidin)

        # signals to final output
        finin = numpy.dot(self.who , hidout)
        # signals from final output
        finout = self.activation_function(finin)

        return finout
