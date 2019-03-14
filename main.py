from models.nets import Unet, smallUnet
from models.utils import convert_images, training_set_generator
import argparse
from models.trainers import Trainer

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--training_image_path", type=str, default="data/train/image/")
    parser.add_argument("--training_mask_path", type=str, default="data/train/label/")
    parser.add_argument("--trainer_option", type=str, default="keras")

    args = parser.parse_args()

    training_image_path = args.training_image_path
    training_mask_path = args.training_mask_path
    trainer_option = args.trainer_option

    unet = Unet(input_shape=(512,512,1), output_shape=(512,512,1))
    model = unet.get_model()
    model.summary()

    #obtain data from raw images

    X_train = convert_images(training_image_path)
    Y_train = convert_images(training_mask_path)

    #get the training data generator
    train_generator = training_set_generator(X_train, Y_train)


    if (trainer_option == "keras"):
        model.compile(optimizer='adam', loss="binary_crossentropy")
        model.fit_generator(train_generator, steps_per_epoch=1000, epochs=50)
    else:
        trainer = Trainer(output_shape=(None, 512,512,1), model=model)
        trainer.fitmodel(train_generator, maxepoch=50)











