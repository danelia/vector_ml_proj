from threading import Thread
import multiprocessing
from .prod_cons import Producer, Consumer

from kafka import KafkaConsumer, KafkaProducer
from google.cloud import pubsub_v1, pubsub
from google.auth import jwt

from time import sleep

PUBLISHER_AUDIENCE = "https://pubsub.googleapis.com/google.pubsub.v1.Publisher"
AUDIENCE = "https://pubsub.googleapis.com/google.pubsub.v1.Subscriber"

class Client():
    def __init__(self, cfg, *args, **kwargs):
        self.cfg = cfg

    def make_producer(sel, topic):
        pass

    def make_consumer(self, topic_subscription, callback):
        pass

    def push(self, value, key=None):
        pass

    def pull(self):
        pass

class KafkaClient(Client):
    def __init__(self, cfg):
        self.kafka_servers = cfg['kafka_servers']
        
        self.topic = None
        self.consumer = None
        self.producer = None
    
    def make_producer(self, topic):
        self.topic = topic
        self.producer = KafkaProducer(bootstrap_servers=self.kafka_servers)
        return Producer(self)

    def make_consumer(self, topic_subscription, callback):
        self.topic = topic_subscription
        self.consumer = KafkaConsumer(self.topic, bootstrap_servers=self.kafka_servers)

        Consumer(self, callback)

    def push(self, value, key=None):
        if self.topic is None or self.producer is None:
            raise Exception("call make_producer first to push")

        self.producer.send(self.topic, key=key, value=value)
        self.producer.flush()

    def pull(self):
        if self.topic is None or self.consumer is None:
            raise Exception("call make_consumer first to pull")

        for msg in self.consumer:
            key = msg.key.decode() if msg.key else None
            return key, msg.value

class GooglePubSubClient(Client):
    def __init__(self, cfg):
        self.project_id = cfg['project_id']

        # auth
        self.credentials = jwt.Credentials.from_service_account_info(
            cfg['credentials'], audience=AUDIENCE
        )
        self.credentials_pub = self.credentials.with_claims(audience=PUBLISHER_AUDIENCE)
        
        self.topic = None
        self.subscription = None
        self.producer = None
        self.consumer = None

    def make_producer(self, topic):
        self.topic = f'projects/{self.project_id}/topics/{topic}'
        self.producer = pubsub_v1.PublisherClient(credentials=self.credentials_pub)
        return Producer(self)

    def make_consumer(self, topic_subscription, callback):
        self.subscription = f'projects/{self.project_id}/subscriptions/{topic_subscription}'
        self.consumer = pubsub.SubscriberClient(credentials=self.credentials)
        self.consumer.subscribe(self.subscription, ack_msg(callback))

        runner_th = Thread(target=runner)
        runner_th.start()

    def push(self, value, key=None):
        if self.topic is None or self.producer is None:
            raise Exception("call make_producer to push")
        
        if key is None:
            key = ''

        self.producer.publish(self.topic, value, key=key)

    def pull(self):
        pass

def ack_msg(func):
    def wrapper(msg):
        func(msg.attributes.get('key', None), msg.data)
        msg.ack()
    return wrapper

def runner():
    while True:
        sleep(.1)