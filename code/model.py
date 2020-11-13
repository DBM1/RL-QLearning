import numpy as np
import keras
from keras import Model
from keras import layers, optimizers, activations, losses, utils




class NNModel(Model):
    def __init__(self):
        super(NNModel, self).__init__()

    def call(self, inputs, training=None, mask=None):
        pass
