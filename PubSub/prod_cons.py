from threading import Thread
from time import sleep

# Producer class for pushing messages to kafka or gpc
class Producer():
    def __init__(self, client):
        self.backend = client

    def push(self, value, key=None):
        self.backend.push(key=key, value=value)

# this will be used only for kafka
# to listen for messages and run callback function on them
class Consumer():
    def __init__(self, client, callback):
        self.backend = client
        self.callback = callback
        listener_thread = Thread(target=self.listener)
        listener_thread.start()

    def listener(self):
        while True:
            sleep(.1)
            key, value = self.backend.pull()
            self.callback(key=key, value=value)