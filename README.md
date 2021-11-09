# vector_ml_proj
### Part 1

First of all, we will gather all the dasatet from Keras. It will need a little bit of pre-processing. 
Images are given as (28, 28) matrix, we will need to reshape them to (28, 28, 1), since it's greyscale image channel will be 1. Also, each pixel value is in range (0, 255), so we will divide them by 255 so that they are in (0, 1) range. We have test set of size 10K, but we will still split training set to have the same size (10K) validation set. And finally, for labels, we will use keras `to_categorical` function to generate "dymmies".
Network will have two Conv layers with AveragePooling, two dense layers and output layer. This model did quite well with loss and accuracy of `[0.3641086518764496, 0.909500002861023]` on test set. But, during training, while training loss was going down I noticed validation loss was increasing. This means there is a chance of model overfitting. To try to improve this let's add dropout layers. After doing this, we can see that, now both validation and training loss are decreasing and we have final score of `[0.2450890690088272, 0.9122999906539917]` on test set. It did improve a little bit! We will save this model in `models/saved` directory, for future use.

Everything described above can be found in `notebooks/cnn.ipynb`.

In `models` directory we can find `cnn_model.py` file. Here in CnnModel() class we describe our model. Class constructer waits for img_width, img_height, channels and num_classes. Defaults will be (28, 28, 1, 10), since we've built this model for fashion-mnist.
This information is used to build model architecture (for inputs and outputs). After that we can call `.train(...)` function to train model. Let's deep dive in parameters.

- fashion=True - boolean, to train on fashion-mnist or own dataset. If True: train_path, test_path are ignored.
- train_path=None - path to train.csv file. This should be csv file with two columns - first for absolute paths to images, second for label.  Examples can be found in `models/data/train.csv`.
- test_path=None - path to train.csv file. Same as train.csv but for test dataset. Examples can be found in `models/data/test.csv`.

        please note that, when training,    train data will be split into train and valid, where 
        validation dataset size will be the same as test. validation will be used for 
        evaluating model at every epoch and test set will be used for evaluating final model.

- batch_size=64 - batch_size for model
- epochs=20 - how many epochs it should train
- out_path='models/saved/model.h5' - where should final model be saved.

Everything above can also be done from console:

    python models/cnn_model.py fashion
        or
    python models/cnn_model.py train --img_width 255 --img_height 255 --channels 3 --num_classes 10 --train_path path/to/train.csv --test_path path/to/test.csv --batch_size 64 --epochs 20 --out_path path/to/save/model

If fashion is used, than --img_width, --img_height and --channels parameters will be ignored and (28, 28, 1) will be used.

This class will also have classmethod `from_file(path)` that will return keras Sequential object, on which we can use `predict()` or other methods.

Future improvements:
- probably it's good idea to not return keras model but rather CnnModel object, and define predict function to first resize images and then ask model about prediction. This way, after building new model or importing from file, user won't have to think about inputs and simply give our object images to predict.
