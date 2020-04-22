# import the necessary packages
import numpy as np
import argparse
import random
import time
import cv2 as cv
import os
import config
import CommonComponents

class MaskRCNN:

	def __init__(self):
		self.default_network_loading()

	def default_network_loading(self):
		self.LABELS = open(os.getcwd() + "\\MaskNetwork\\" + config.MASK_LABELS).read().strip().split("\n")
		np.random.seed(42)
		self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3), dtype="uint8")
		self.NETWORK_CONFIG = os.getcwd() + "\\MaskNetwork\\" + config.MASK_CONFIG
		self.WEIGHTS_CONFIG = os.getcwd() + "\\MaskNetwork\\" + config.MASK_WEIGHTS
		self.network = cv.dnn.readNetFromTensorflow(self.WEIGHTS_CONFIG, self.NETWORK_CONFIG)

	def read_image(self, image_name, show_results=False):
		image = cv.imread(image_name)
		(imageHeight, imageWidth) = image.shape[:2]
		blob = cv.dnn.blobFromImage(image, swapRB=True, crop=False)
		self.network.setInput(blob)
		(boxes, masks) = self.network.forward(["detection_out_final", "detection_masks"])
		segments = []
		regions_of_interest = []
		maskings = []
		classIDs = []
		confidences = []
		coordinates = []
		for i in range(0, boxes.shape[2]):
			# extract the class ID of the detection along with the confidence
			# (i.e., probability) associated with the prediction
			classID = int(boxes[0, 0, i, 1])
			confidence = boxes[0, 0, i, 2]
			# filter out weak predictions by ensuring the detected probability
			# is greater than the minimum probability
			if confidence > config.MASK_CONFIDENCE:
				classIDs.append(self.LABELS[classID])
				confidences.append(confidence)
				# clone our original image so it can be modified
				clone = image.copy()
				box = boxes[0, 0, i, 3:7] * np.array([imageWidth, imageHeight, imageWidth, imageHeight])
				(startX, startY, endX, endY) = box.astype("int")
				boxW = endX - startX
				boxH = endY - startY
				# extract the pixel-wise segmentation for the object, resize
				# the mask such that it's the same dimensions of the bounding
				# box, and then finally threshold to create a *binary* mask
				mask = masks[i, classID]
				mask = cv.resize(mask, (boxW, boxH), interpolation=cv.INTER_NEAREST)
				mask = (mask > config.MASK_THRESHOLD)
				# extracting the Region of Interest marking out our object of the image
				roi = clone[startY:endY, startX:endX]
				visMask = (mask * 255).astype("uint8")
				instance = cv.bitwise_and(roi, roi, mask=visMask)
				# show the extracted ROI, the mask, along with the
				# segmented instance
				segments.append(instance)
				regions_of_interest.append(roi)
				maskings.append(visMask)
				# now, extract *only* the masked region of the ROI by passing
				# in the boolean mask array as our slice condition
				roi = roi[mask]
				color = random.choice(self.COLORS)
				blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")
				# store the blended ROI in the original image
				coordinates.append([startX, startY, endX, endY])
				if show_results:
					clone[startY:endY, startX:endX][mask] = blended
					# Drawing the bounding of a detected object
					color = [int(c) for c in color]
					cv.rectangle(clone, (startX, startY), (endX, endY), color, 2)
					# draw the predicted label and associated probability of the entity existing
					text = "{}: {:.4f}".format(self.LABELS[classID], confidence)
					cv.putText(clone, text, (startX, startY - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
					cv.imshow("Output", clone)
					cv.waitKey(0)
		# Returning all relevant images and the class in the detection area
		return [segments, regions_of_interest, maskings, classIDs, confidences, coordinates]


