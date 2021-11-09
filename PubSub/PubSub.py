from .clients import KafkaClient, GooglePubSubClient

class PubSub():
    def __init__(self, vendor, cfg):
        if vendor not in ['gcp', 'kafka']:
            print("only supporting 'gcp' or 'kafka', kafka will be used")
            vendor = 'kafka'

        self.vendor = vendor
        self.cfg = cfg

        self.backend = None
        if self.vendor == 'kafka':
            self.backend = KafkaClient
        elif self.vendor == 'gcp':
            self.backend = GooglePubSubClient

    def create_producer(self, topic_id):
        return self.backend(self.cfg).make_producer(topic_id)

    def create_consumer(self, topic_subscription_id, callback):
        self.backend(self.cfg).make_consumer(topic_subscription_id, callback)
