import numpy as np
import pandas as pd

from tensorflow.keras.datasets import fashion_mnist

from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, AveragePooling2D, Dropout
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from PIL import Image

class CnnModel():
    def __init__(self, img_width=28, img_height=28, channels=1, num_classes=10):
        self.img_width, self.img_height, self.channels = img_width, img_height, channels
        self.num_classes = num_classes

        self.input_shape = (self.img_width, self.img_height, self.channels)
        
        self.model = self._build_model()

    @staticmethod
    def from_file(path):
        model = load_model(path)
        return model

    def _build_model(self):
        model = Sequential()

        model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=self.input_shape ))
        model.add(AveragePooling2D())
        model.add(Dropout(0.25))

        model.add(Conv2D(32, kernel_size=3, activation='relu'))
        model.add(AveragePooling2D())
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.25))

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.25))

        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def train(self, fashion=True, train_path=None, test_path=None, batch_size=64, epochs=20, out_path='models/saved/model.h5'):
        if fashion:
            (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        else:
            (x_train, y_train), (x_test, y_test) = self._get_data(train_path, test_path)

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=x_test.shape[0])

        # we need to reshape data according to channels. fashion mnist dataset is (28,28) images with one channel.
        # we need to reshape them to (28, 28, 1)
        if len(x_train[0].shape) < 3:
            x_train = x_train.reshape(x_train.shape + (self.channels, ) )
            x_val = x_val.reshape(x_val.shape + (self.channels, ) )
            x_test = x_test.reshape(x_test.shape + (self.channels, ) )

        # normilize so that values are in (0, 1) range instead of (0, 255)
        x_train = x_train.astype('float32') / 255
        x_val = x_val.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        y_train = to_categorical(y_train)
        y_val = to_categorical(y_val)
        y_test = to_categorical(y_test)

        self.model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=epochs)

        test_loss, test_accuracy = self.model.evaluate(x_test, y_test)

        print(f"Training done with loss {test_loss} and accuracy {test_accuracy} on test set")

        self.model.save(out_path)

    def _get_data(self, train_path, test_path):
        train_df = pd.read_csv(train_path, header=None)
        test_df = pd.read_csv(test_path, header=None)
        
        read_img = lambda x: np.asarray(Image.open(x).resize( (self.img_width, self.img_height) ))

        def read(df):
            x = np.array(df.iloc[:,0].apply(read_img).to_list())
            y = df.iloc[:, 1].astype('category').cat.codes.to_list()

            return x, y

        x_train, y_train = read(train_df)
        x_test, y_test = read(test_df)

        return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("operation", type=str, choices=['fashion', 'train'], default="fashion",
                            help='fashion to train on fashion_mnist dataset, train to train on own dataset')

    parser.add_argument("--img_width", type=int, default=28)
    parser.add_argument("--img_height", type=int, default=28)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=10)

    parser.add_argument("--train_path", type=str, default='')
    parser.add_argument("--test_path", type=str, default='')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--out_path", type=str, default='models/saved/model.h5')

    args = parser.parse_args()

    if args.operation == 'fashion':
        model = CnnModel()
    else:
        model = CnnModel(img_height=args.img_height,
                         img_width=args.img_width,
                         channels=args.channels,
                         num_classes=args.num_classes)

    model.train(fashion=args.operation == 'fashion',
                train_path=args.train_path,
                test_path=args.test_path,
                batch_size=args.batch_size,
                epochs=args.epochs,
                out_path=args.out_path)