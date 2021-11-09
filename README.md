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

### Part 2

For this part we will write factory class. It is located in `PubSub/PubSub.py`. The idea is to give the factory class vendor and configuration dict. And from there, user will be able to create as many producers and consumers as they want, using that vendor and that config. Implemented vendors for now are `['gcp', 'kafka']` for Google Pub/Sub or Apache kafka. Configurations can be found in `cfg.py`. Usually, it's not good idea to have configs in .py file, but it's easier for our demonstration.

Configs are as follows:
`kafka`

    {
        "kafka_servers": [
            "localhost:9092"
            ]
    }
    
`Google Pub/Sub`

    {
        "project_id" : "",
    
        "credentials" : {
            "type": ",
            "project_id": "",
            "private_key_id": "",
            "private_key": "",
            "client_email": "",
            "client_id": "",
            "auth_uri": "",
            "token_uri": "",
            "auth_provider_x509_cert_url": "",
            "client_x509_cert_url": ""
        }
    }

After creating PubSub object, it has two methods: `create_producer(topic_id)` and `create_consumer(topic_subscription_id, callback)`. 

In `create_producer(topic_id)` topic_id identifies on what topic messages should be pushed. This method will return Producer object located in `PubSub/prod_cons.py`. Producer object has only one method `.push(key, value)`, which  will push message value with key to server. key can be None.

In case of `create_consumer(topic_subscription_id, callback)` we can be waiting for topic_id in case kafka is used, or subscription_id if google pub/sub if used. Hence, the name. Also, user has to write callback function, that will be passed to this function. Arguments of this callback funtion are (key, value). So, after create_consumer is called, it will start new thread in the background, that will be waiting on any messages on that topic/subscription and will call callback function when it recieves any message.

Usage:
```` 
from PubSub.PubSub import PubSub
from cfg import kafka_cfg
import json
from time import sleep

def callback(key, value):
    message = json.loads(value)
    print(message)

pubsub = PubSub('kafka', kafka_cfg)

# runs backgound thread that listens to app topic 
# and runs callback function on recieved messages
pubsub.create_consumer("app", callback)

# returns Producer object, that has method .push to push messages to topic app
producer = pubsub.create_producer("app")

# to make sure consumer thread started
sleep(.3)

for i in range(5):
    message = {'data': str(i)}

    producer.push(value=json.dumps(message).encode('utf-8'), key=None)
`````

Only thing that will change in case we want to use Google pub/sub, apart from topic/subscription names is :
```` 
pubsub = PubSub('gcp', gcp_cfg)
`````

Client classes, which PubSub is using to create producers and consumers are located in `PubSub/clients.py`. In future, to add more vendors, their client classes should be added here.
