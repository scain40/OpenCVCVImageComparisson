import numpy as np
import cv2 as cv
import os
import sys


class ObjectDetector:
	"""
	Object Detector is the class model for using YOLOv2 and gathering results
	"""
	def __init__(self):
		self.network_loading()

	def network_loading(self):
		# Loading in the trained darknet models labels
		self.LABELS = open(os.getcwd() + "\coco.names").read().strip().split("\n")
		self.readingNetwork = cv.dnn.readNetFromDarknet(os.getcwd() + "\yolov3.cfg", os.getcwd() + "\yolov3.weights")

	def read_image(self, image_name):
		# Reading in a specific image from the files that exist in the input folder
		working_image = cv.imread(image_name)
		self.labelNames = self.readingNetwork.getLayerNames()
		self.labelNames = [self.labelNames[i[0] - 1] for i in self.readingNetwork.getUnconnectedOutLayers()]
		imageInputBlob = cv.dnn.blobFromImage(working_image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
		self.readingNetwork.setInput(imageInputBlob)
		layerOutputs = self.readingNetwork.forward(self.labelNames)
		return self.processReading(layerOutputs)


	def processReading(self, processingResults):
		# Takes in the results from a reading and proceses them to check for valid objects
		classIDs = []
		for objects in processingResults:
			# loop over each of the detections
			for detection in objects:
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]
				if confidence > 0.9:
					# Appending the names of all the objects to be sorted later
					classIDs.append(self.LABELS[classID])
		# Just returning class names as it's only thing relevant to OSR later
		return classIDs

