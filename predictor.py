import numpy as np

from models.cnn_model import CnnModel
from PubSub.PubSub import PubSub

import json

from cfg import kafka_cfg, gcp_cfg

model_path = "models/saved/model.h5"
model = CnnModel.from_file(model_path)

def callback(key, value):
    message = json.loads(value)

    x = np.array(message['data'])
    prediction = np.argmax(model.predict(x))

    message = {'request_id': message['request_id'], 'prediction': str(prediction)}
    print(message)
    producer.push(json.dumps(message).encode('utf-8'))

# For kafka
pubsub = PubSub('kafka', kafka_cfg)
producer = pubsub.create_producer("model_result")
pubsub.create_consumer("app", callback)

# For google pub/sub
# pubsub = PubSub('gcp', gcp_cfg)
# producer = pubsub.create_producer("model_result")
# pubsub.create_consumer("app-sub", callback)