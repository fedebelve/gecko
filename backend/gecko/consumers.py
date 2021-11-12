from channels.generic.websocket import WebsocketConsumer
from random import randint
from time import sleep
import json

class WSConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()
        print(self.scope['client'])

        for i in range(100):
            self.send(json.dumps({'message': randint(1,100)}))
            sleep(1)