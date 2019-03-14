from keras_preprocessing.image import ImageDataGenerator
from skimage import io
import os
import numpy as np

def convert_images(data_path):
    all_images = []
    for filename in sorted(os.listdir(data_path)):
        img = io.imread(data_path + filename, as_gray=True)
        all_images.append(img)

    X = np.expand_dims(np.array(all_images), axis=3)
    return X


def training_set_generator(X_train, Y_train):
    data_gen_args = dict(featurewise_center=True,
                         featurewise_std_normalization=True,
                         rotation_range=90,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2,
                         rescale=1./255)
    # Provide the same seeda and keyword arguments to the fit and flow methods
    seed = 1
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    image_datagen.fit(X_train, augment=True, seed=seed)
    mask_datagen.fit(Y_train, augment=True, seed=seed)

    image_generator = image_datagen.flow(X_train, seed=seed, batch_size=1)
    mask_generator = mask_datagen.flow(Y_train, seed=seed, batch_size=1)

    for (img, mask) in zip(image_generator, mask_generator):
        yield (img, mask)





