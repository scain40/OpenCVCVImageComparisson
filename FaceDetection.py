# https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81
import cv2 as cv
import os
import CommonComponents
import face_recognition
from PIL import Image
# Could have this be done from side-on and front-on to try and determine the direction of each face in a picture
# Detector from this technique does not work, gonna have to take another approach


class FaceDetection:

	def __init__(self):
		self.initialise_detector()

	def initialise_detector(self):
		self.front_face_detector = cv.CascadeClassifier('FaceDetectionDatasets\\haarcascade_frontalface_default.xml')
		self.profile_face_detector = cv.CascadeClassifier('FaceDetectionDatasets\\haarcascade_profileface.xml')

	def read_image(self, image_name, debug_images=False):
		reading_image = cv.imread(image_name)
		image_front_faces = []
		image_side_faces = []
		# Steps, turn image grey, use cascade detector, split up faces from images
		greyscale = cv.cvtColor(reading_image, cv.COLOR_BGR2GRAY)
		frontal_faces = self.front_face_detector.detectMultiScale(greyscale, 1.1, 4)
		side_faces = self.profile_face_detector.detectMultiScale(greyscale, 1.1, 5)
		for face in frontal_faces:
			image_front_faces.append(face)
		for face in side_faces:
			image_side_faces.append(face)
		return [image_front_faces, image_side_faces]


	def facial_recognition_library_read_image(self, image_name, debug_images = False):
		print('Processing {x}'.format(x=image_name.split("\\")[-1]))
		detecting_image = face_recognition.load_image_file(image_name)
		detected_faces = face_recognition.face_locations(detecting_image, number_of_times_to_upsample=1, model="cnn")
		sorted_return_faces = []
		for x in detected_faces:
			top, right, bottom, left = x
			width = right - left
			height = bottom - top
			sorted_return_faces.append([left, top, width, height])
		return sorted_return_faces

