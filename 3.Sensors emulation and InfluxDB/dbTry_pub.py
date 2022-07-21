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

class MyPublisher:
	def __init__(self, clientID,topic, broker, port):
		self.clientID = clientID

		# create an instance of paho.mqtt.client
		self._paho_mqtt = PahoMQTT.Client(self.clientID, False) 
		# register the callback
		self._paho_mqtt.on_connect = self.myOnConnect

		self.messageBroker = broker
		#'192.168.1.5'

	def start (self,port):
		#manage connection to broker
		self._paho_mqtt.connect(self.messageBroker,port)
		self._paho_mqtt.loop_start()

	def stop (self):
		self._paho_mqtt.loop_stop()
		self._paho_mqtt.disconnect()

	def myPublish(self, topic, message):
		# publish a message with a certain topic
		self._paho_mqtt.publish(topic, message, 2)

	def myOnConnect (self, paho_mqtt, userdata, flags, rc):
		print ("Connected to %s with result code: %d" % (self.messageBroker, rc))

org = "reuive037125@yahoo.com"
bucket = "buildingDesign"

cata_topic="buildingDesign/groupwork/sensor"
broker="test.mosquitto.org"
port=1883


while True:

	path = os.getcwd()+'/sample_2.csv'
	csv_reader = csv.reader(open(path))

	#row = "row"
	count = 0
	lineArray=[]
	rowArray=[]
	#print(row+str(count))

	for line in csv_reader:
		#row+str(count) = line
		lineArray.append(line)

	#print(lineArray[1])

	rowLine = lineArray[1]

	timehour = '2022/'+rowLine[0].strip()
	if timehour[-8:-6] == '24':
		(date, time) = timehour.split()
		#print(date,'----', time)
		time = time[:-8] + '00' + time[-6:]
		#print(time)
		date = datetime.datetime.strptime(date, '%Y/%m/%d') + datetime.timedelta(days=1)
		timestamp1 = str(date.strftime('%Y/%m/%d'))+' '+time
		#print(timestamp1)
	else:
		timestamp1='2022/'+rowLine[0].strip()

	timestamp2=datetime.datetime.strptime(timestamp1, '%Y/%m/%d  %H:%M:%S')
	#timestamp3=timestamp2.replace(tzinfo=datetime.timezone.utc).timestamp()
	timestamp3 = int(round(timestamp2.timestamp()))

	hostTime = time.strftime("%Y%m%d%H%M%S", time.localtime()) 
	#print(hostTime)
	tableName = "eplusout2022,host="+str(hostTime)+' '
	finalString= tableName +'DateTime='+str(timestamp3)+','
	finalString = finalString + 'DrybulbTemperatureHourly='+str(rowLine[1])+','
	finalString = finalString + 'ZONE1OperativeTemperature='+str(rowLine[285])+','
	finalString = finalString + 'ZONE5OperativeTemperature='+str(rowLine[293])+','
	finalString = finalString + 'ZONE4OperativeTemperature='+str(rowLine[301])+','
	finalString = finalString + 'ZONE10OperativeTemperature='+str(rowLine[309])+','
	finalString = finalString + 'ZONE6OperativeTemperature='+str(rowLine[317])+','
	finalString = finalString + 'ZONE3OperativeTemperature='+str(rowLine[325])+','
	finalString = finalString + 'ZONE2OperativeTemperature='+str(rowLine[333])+','
	finalString = finalString + 'ZONE7OperativeTemperature='+str(rowLine[341])+','
	finalString = finalString + 'ZONE8OperativeTemperature='+str(rowLine[349])+','
	finalString = finalString + 'ElectricityFacility='+str(rowLine[973])+','
	finalString = finalString + 'DistrictCoolingFacility='+str(rowLine[1009])+','
	finalString = finalString + 'DistrictHeatingFacility='+str(rowLine[1010])
	finalString = finalString.rstrip()
	print(finalString)

	'''
	with InfluxDBClient(url="https://europe-west1-1.gcp.cloud2.influxdata.com", token="QqN2R_-4S4tymSTzN-tg75etmmM2H7qJIH2tYjULmoE2YBFnKy1joGkb1s5rv1LdVOWq-JENE0rC6j1JfjLenA==", org=org, ssl_CA_cert=certifi.where()) as client:
		write_api = client.write_api(write_options=SYNCHRONOUS, ssl_CA_cert=certifi.where())
		#finalString = "mem,host=host1 used_percent=23.43234543"
		#finalString = "eplusout2005,host=005 used_percent=64.2377,times=1385672008"
		write_api.write(bucket, org, finalString)

	client.close()
	'''

	cata_pub = MyPublisher("sensor publisher", cata_topic, broker, port)
	cata_pub.start(port)
	cata_pub.myPublish(cata_topic, finalString)


	time.sleep(6)




