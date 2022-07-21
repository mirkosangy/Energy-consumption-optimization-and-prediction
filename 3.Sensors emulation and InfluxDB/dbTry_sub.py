from influxdb import InfluxDBClient
import random

from datetime import datetime
import os
import certifi
import datetime
import csv
import numpy as np
import time
import paho.mqtt.client as PahoMQTT
import pandas as pd

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

class MySubscriber:
		def __init__(self, clientID):
			self.clientID = clientID
			# create an instance of paho.mqtt.client
			self._paho_mqtt = PahoMQTT.Client(clientID, False) 

			# register the callback
			self._paho_mqtt.on_connect = self.myOnConnect
			self._paho_mqtt.on_message = self.myOnMessageReceived

			self.topic = "buildingDesign/groupwork/sensor"
			self.messageBroker = "test.mosquitto.org"


		def start (self):
			#manage connection to broker
			self._paho_mqtt.connect(self.messageBroker, 1883)
			self._paho_mqtt.loop_start()
			# subscribe for a topic
			self._paho_mqtt.subscribe(self.topic, 2)

		def stop (self):
			self._paho_mqtt.unsubscribe(self.topic)
			self._paho_mqtt.loop_stop()
			self._paho_mqtt.disconnect()

		def myOnConnect (self, paho_mqtt, userdata, flags, rc):
			print ("Connected to %s with result code: %d" % (self.messageBroker, rc))

		def myOnMessageReceived (self, paho_mqtt , userdata, msg):
			# A new message is received
			print ("Topic:'" + msg.topic+"', QoS: '"+str(msg.qos)+"' Message: '"+str(msg.payload) + "'")
			#print(msg.payload.decode())
			with InfluxDBClient(url="https://ap-southeast-2-1.aws.cloud2.influxdata.com", token="TCdOUvV3Sa0-bXUL31oBj-p6Lmy2rnIVUfOYTVB1dpKcDc_0ACvO6ljJm69-YcMp0qxdEbO5-qUj9hUjQ1TJVw==", org="reuive037125@yahoo.com", ssl_CA_cert=certifi.where()) as client:
				write_api = client.write_api(write_options=SYNCHRONOUS, ssl_CA_cert=certifi.where())
				write_api.write("buildingDesign", "reuive037125@yahoo.com", msg.payload.decode())
				print("data successfully upload to influxDB")


if __name__ == "__main__":
	test = MySubscriber("Sensor Subscriber")
	test.start()
	
	while (True):
		pass






