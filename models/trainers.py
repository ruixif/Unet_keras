import tensorflow as tf
from random import shuffle
import numpy as np

class Trainer:
    def __init__(self,
                 output_shape,
                 model,
                 lr=0.0001,
                 pretrained=None):


        self.model = model
        self.weights = model.weights
        self.inputs, = model.inputs
        self.outputs, = model.outputs
        self.label = tf.placeholder(tf.float32, shape=output_shape)
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.outputs)
        self.batch_loss = tf.reduce_mean(self.loss)
        self.train = tf.train.AdamOptimizer(lr).minimize(self.batch_loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if pretrained is not None:
            self.model.load_weights(pretrained)


    def fitmodel(self, generator, maxepoch):

        # train and valid
        for epoch in range(maxepoch):
            i = 0
            for this_X, this_y in generator:
                _, current_loss = self.sess.run([self.train, self.batch_loss],
                                                feed_dict={self.inputs: this_X,
                                                           self.label: this_y,
                                                           })
                i = i + 1
                print("Running the sample pair {} of the epoch {}...".format(i, epoch+1))
                print("Current loss: {}".format(current_loss))

            self.model.save_weights('saved_weights.h5')






