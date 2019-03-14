from tensorflow.python.keras import layers, models
from tensorflow.contrib.layers import xavier_initializer_conv2d


class Unet:
    def __init__(self, input_shape, output_shape):
        inputs = layers.Input(shape=input_shape)

        convd0c, convd1c, convd2c, convd3c, convd4c = self._encoder(inputs)
        convu3d = self._decode_layer(vertical_input=convd4c, horizontal_input=convd3c, filter_num=512,
                                     filter_size=(3,3))
        convu2d = self._decode_layer(vertical_input=convu3d, horizontal_input=convd2c, filter_num=256,
                                     filter_size=(3, 3))
        convu1d = self._decode_layer(vertical_input=convu2d, horizontal_input=convd1c, filter_num=128,
                                     filter_size=(3, 3))
        convu0d = self._decode_layer(vertical_input=convu1d, horizontal_input=convd0c, filter_num=64,
                                     filter_size=(3, 3))
        #output layer
        conv_score = layers.Conv2D(2, (1,1))(convu0d)
        conv_label = layers.Conv2D(1, (1,1), activation="sigmoid")(conv_score)

        self.model = models.Model(inputs=inputs,outputs=conv_label)

    def get_model(self):
        return self.model

    def _encode_layer(self, input, maxpool, dropout, filter_num, filter_size):
        if maxpool:
            poold1a = layers.MaxPooling2D(pool_size=(2, 2))(input)
        else:
            poold1a = input

        convd1b = layers.Conv2D(filter_num, filter_size, activation='relu', padding="same",
                                    kernel_initializer=xavier_initializer_conv2d())(poold1a)
        convd1c = layers.Conv2D(filter_num, filter_size, activation='relu', padding="same",
                                    kernel_initializer=xavier_initializer_conv2d())(convd1b)

        if dropout:
            output = layers.Dropout(rate=0.5)(convd1c)
        else:
            output = convd1c

        return output


    def _decode_layer(self, vertical_input, horizontal_input, filter_num, filter_size):
        upu3a = layers.UpSampling2D(size=(2,2))(vertical_input)
        convu3a = layers.Conv2DTranspose(filter_num, (2,2), strides=1, padding="same", activation='relu', kernel_initializer=xavier_initializer_conv2d())(upu3a)
        convu3b = layers.concatenate(inputs=[horizontal_input, convu3a], axis=3)
        convu3c = layers.Conv2D(filter_num, filter_size, activation='relu', padding="same", kernel_initializer=xavier_initializer_conv2d())(convu3b)
        convu3d = layers.Conv2D(filter_num, filter_size, activation='relu', padding="same", kernel_initializer=xavier_initializer_conv2d())(convu3c)
        return convu3d

    def _encoder(self, inputs):
        convd0c = self._encode_layer(input=inputs, maxpool = False, dropout=False, filter_num=64, filter_size=(3, 3))
        convd1c = self._encode_layer(input=convd0c, maxpool=True, dropout=False, filter_num=128, filter_size=(3, 3))
        convd2c = self._encode_layer(input=convd1c, maxpool=True, dropout=False, filter_num=256, filter_size=(3, 3))
        convd3c = self._encode_layer(input=convd2c, maxpool=True, dropout=True, filter_num=512, filter_size=(3, 3))
        convd4c = self._encode_layer(input=convd3c, maxpool=True, dropout=True, filter_num=1024, filter_size=(3, 3))

        return convd0c, convd1c, convd2c, convd3c, convd4c


class smallUnet(Unet):
    def __init__(self, input_shape, output_shape):
        inputs = layers.Input(shape=input_shape)

        convd0c, convd1c = self._encoder(inputs)
        convu0d = self._decode_layer(vertical_input=convd1c, horizontal_input=convd0c, filter_num=64,
                                     filter_size=(3, 3))
        #output layer
        conv_score = layers.Conv2D(2, (1,1))(convu0d)
        conv_label = layers.Conv2D(1, (1,1), activation="sigmoid")(conv_score)

        self.model = models.Model(inputs=inputs,outputs=conv_label)

    def _encoder(self, inputs):
        convd0c = self._encode_layer(input=inputs, maxpool = False, dropout=False, filter_num=64, filter_size=(3, 3))
        convd1c = self._encode_layer(input=convd0c, maxpool=True, dropout=False, filter_num=128, filter_size=(3, 3))

        return convd0c, convd1c

    def _decode_layer(self, vertical_input, horizontal_input, filter_num, filter_size):
        upu3a = layers.UpSampling2D(size=(2,2))(vertical_input)
        convu3a = layers.Conv2DTranspose(filter_num, (2,2), strides=1, padding="same", activation='relu', kernel_initializer=xavier_initializer_conv2d())(upu3a)
        convu3b = layers.concatenate(inputs=[horizontal_input, convu3a], axis=3)
        convu3c = layers.Conv2D(filter_num, filter_size, activation='relu', padding="same", kernel_initializer=xavier_initializer_conv2d())(convu3b)
        convu3d = layers.Conv2D(filter_num, filter_size, activation='relu', padding="same", kernel_initializer=xavier_initializer_conv2d())(convu3c)
        return convu3d










