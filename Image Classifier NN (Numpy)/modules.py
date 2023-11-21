################################################################################
# MIT License
#
# Copyright (c) 2023 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2023
# Date Created: 2023-11-01
################################################################################
"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.

        TODO:
        Initialize weight parameters using Kaiming initialization. 
        Initialize biases with zeros.
        Hint: the input_layer argument might be needed for the initialization

        Also, initialize gradients with zeros.
        """

        # Note: For the sake of this assignment, please store the parameters
        # and gradients in this format, otherwise some unit tests might fail.
        self.params = {'weight': None, 'bias': None} # Model parameters
        self.grads = {'weight': None, 'bias': None} # Gradients

        
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        if input_layer:
            #self.params['weight'] = (0, 1/np.sqrt(in_features))
            self.params['weight'] = np.random.randn(in_features, out_features) * (1 / np.sqrt(in_features))

        else:
            #self.params['weight'] = (0, np.sqrt(2)/np.sqrt(in_features))
            self.params['weight'] = np.random.randn(in_features, out_features) * (np.sqrt(2) / np.sqrt(in_features))

        self.params['bias'] = np.zeros((1, out_features))

        self.grads['weight'] = np.zeros_like(self.params['weight'])
        self.grads['bias'] = np.zeros_like(self.params['bias'])
      
        self.cache = None
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        #self.params['weight'] has shape (8, 52)
        #x has shape (52, 7)

        #print(x)
        #print(x.shape)
        #print()
        out = np.dot(x, self.params['weight']) + self.params['bias']
        self.cache = x

        #for o in out:
        #    print(o)
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        grad_out = self.params['weight']

        #self.grads['weight'] = np.dot(dout.T, self.cache)
        self.grads['weight'] = np.dot(self.cache.T, dout)
        self.grads['bias'] = np.sum(dout, axis=0, keepdims=True)

        dx = np.dot(dout, grad_out.T)
        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.cache = None
        #######################
        # END OF YOUR CODE    #
        #######################


class ELUModule(object): 
    """
    ELU activation module.
    """

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        out = np.where(x > 0, x, 1.0 * (np.exp(x) - 1))
        self.cache = x
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout): #correct
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        dx = dout * np.where(self.cache > 0, 1, np.exp(self.cache))
        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.cache = None
        #######################
        # END OF YOUR CODE    #
        #######################


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        #out = np.exp(x)  / np.sum(np.exp(x))
        
        #x_max = np.max(x)
        #e_x = np.exp(x - x_max)
        #out = e_x / np.sum(e_x)
        
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims = True))
        out = exp_x / np.sum(exp_x, axis=-1, keepdims = True)
        self.cache = out
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        dx = self.cache *(dout - np.sum(dout*self.cache, axis = 1, keepdims = True))
        #######################
        # END OF YOUR CODE    #
        #######################

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.cache = None
        #######################
        # END OF YOUR CODE    #
        #######################


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        exp_x = np.exp(x - np.max(x, axis=1, keepdims = True))
        probs = exp_x / np.sum(exp_x, axis=1, keepdims = True)

        m = y.shape[0]
        log_like = -np.log(probs[range(m), y])
        out = np.sum(log_like)/ m #mean / y.shape[0] is correct

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        m = y.shape[0]
        #print(y.shape[0])
        one_hot_y = np.zeros_like(x)
        one_hot_y[range(m), y] = 1

        exp_values = np.exp(x - np.max(x, axis=-1, keepdims = True))
        probs = exp_values / np.sum(exp_values, axis =1, keepdims = True)

        dx = (probs - one_hot_y) / m

        #######################
        # END OF YOUR CODE    #
        #######################

        return dx