from PubSub.PubSub import PubSub
from tensorflow.keras.datasets import fashion_mnist

import json
import uuid

from time import sleep

from cfg import kafka_cfg, gcp_cfg

def callback(key, value):
    message = json.loads(value)
    request_id = message['request_id']
    print("** CONSUMER: Received prediction {} for request id {}".format(message['prediction'], request_id))

# For kafka
pubsub = PubSub('kafka', kafka_cfg)
producer = pubsub.create_producer("app")
pubsub.create_consumer("model_result", callback)

# For google pub/sub
# pubsub = PubSub('gcp', gcp_cfg)
# producer = pubsub.create_producer("app")
# pubsub.create_consumer("model_result-sub", callback)

(_, _), (x_test, y_test) = fashion_mnist.load_data()

for i in x_test:
    message_id = str(uuid.uuid4())
    x = i.reshape((1, ) + i.shape + (1, ) )
    message = {'request_id': message_id, 'data': x.tolist()}

    producer.push(json.dumps(message).encode('utf-8'))

    print("-- PRODUCER: Sent message with id {}".format(message_id))

    sleep(.1)