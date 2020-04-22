import config
import cv2 as cv
from math import sqrt


class DataStorage:

	def __init__(self, image_name):
		self.image_name = image_name
		self.yolo = []
		self.mask_results = []
		self.faces = []
		self.poses = []

	def mask_results_comparissons(self, mask_comparing_set):
		# Need to compare objects found by a mask to see how they weigh up to each other
		matching_results = [0 for x in range(mask_comparing_set.__len__())]
		for self_entry in self.mask_results:
			for comparing_entry_number in range(mask_comparing_set.__len__()):
				if self_entry.classID == mask_comparing_set[comparing_entry_number].classID and matching_results[comparing_entry_number == 0]:
					matching_results[comparing_entry_number] = 1
		total_matched = 0
		for x in matching_results:
			if x == 1:
				total_matched = total_matched + 1
		return (1 / matching_results.__len__()) * total_matched


class Face:

	def __init__(self, coordinates, side):
		self.x = coordinates[0]
		self.y = coordinates[1]
		self.width = coordinates[2]
		self.height = coordinates[3]
		self.side = side

	def average_colour_comparison(self, secondary_average):
		# B comparison
		if self.average_colour[0] > secondary_average[0] - config.AVERAGE_FACE_COLOUR_DRIFT  and self.average_colour[0] < secondary_average[0] + config.AVERAGE_FACE_COLOUR_DRIFT:
			if self.average_colour[1] > secondary_average[1] - config.AVERAGE_FACE_COLOUR_DRIFT and self.average_colour[1] < secondary_average[1] + config.AVERAGE_FACE_COLOUR_DRIFT:
				if self.average_colour[2] > secondary_average[2] - config.AVERAGE_FACE_COLOUR_DRIFT and self.average_colour[2] < secondary_average[2] + config.AVERAGE_FACE_COLOUR_DRIFT:
					return True
		return False

	def masked_average_colour_comparison_scoring(self, secondary_average):
		"""Performs maskless rgb face average comparisson"""
		# B comparison
		divisions = config.MASK_AVERAGE_FACE_COLOUR_DRIFT / config.MASK_FACE_COLOUR_DRIFT_STAGES
		totalTesting = divisions
		totalScore = 0
		for x in range(config.MASK_FACE_COLOUR_DRIFT_STAGES):
			if self.mask_average_bgr[0] > secondary_average[0] - totalTesting  and self.mask_average_bgr[0] < secondary_average[0] + totalTesting:
				if self.mask_average_bgr[1] > secondary_average[1] - totalTesting and self.mask_average_bgr[1] < secondary_average[1] + totalTesting:
					if self.mask_average_bgr[2] > secondary_average[2] - totalTesting and self.mask_average_bgr[2] < secondary_average[2] + totalTesting:
						totalScore = totalScore + 1 / config.MASK_FACE_COLOUR_DRIFT_STAGES
			totalTesting = totalTesting + divisions
		return totalScore

	def average_colour_comparison_scoring(self, secondary_average):
		"""Performs maskless rgb face average comparisson"""
		# B comparison
		divisions = config.AVERAGE_FACE_COLOUR_DRIFT / config.FACE_COLOUR_DRIFT_STAGES
		totalTesting = divisions
		totalScore = 0
		for x in range(config.FACE_COLOUR_DRIFT_STAGES):
			if self.average_colour[0] > secondary_average[0] - totalTesting  and self.average_colour[0] < secondary_average[0] + totalTesting:
				if self.average_colour[1] > secondary_average[1] - totalTesting and self.average_colour[1] < secondary_average[1] + totalTesting:
					if self.average_colour[2] > secondary_average[2] - totalTesting and self.average_colour[2] < secondary_average[2] + totalTesting:
						totalScore = totalScore + 1 / config.FACE_COLOUR_DRIFT_STAGES
			totalTesting = totalTesting + divisions
		return totalScore

	def hsv_face_colour_comparisson(self, secondary_average):
		"""Performs maskless hsv colour face comparison"""
		# Calculating the magnitude of the hues combined
		hue_distance = min(abs(secondary_average[0] - self.average_hsv[0]), 360-abs(secondary_average[1] - self.average_hsv[0])) / 180
		saturation_distance = abs(secondary_average[1] - self.average_hsv[1])
		value_distance = abs(secondary_average[2] - self.average_hsv[2])
		# Getting the vector magnitude of this
		distancing_total = sqrt((hue_distance*hue_distance) + (saturation_distance * saturation_distance) + (value_distance * value_distance))
		# Test to see how close the distance is to the maximum allowed distancing
		divisions = config.HSV_FACE_COLOUR_DRIFT / config.HSV_FACE_COLOUR_DRIFT_STAGES
		totalTesting = divisions
		totalScore = 0
		# The closer the distance is to 0, the closer the colours were in testing
		for x in range(config.HSV_FACE_COLOUR_DRIFT_STAGES):
			if distancing_total < totalTesting:
				totalScore = totalScore + (1 / config.HSV_FACE_COLOUR_DRIFT_STAGES)
			totalTesting = totalTesting + divisions
		return totalScore

	def masked_hsv_face_colour_comparisson(self, secondary_average):
		"""Performs masked hsv colour face comparison"""
		# Calculating the magnitude of the hues combined
		hue_distance = min(abs(secondary_average[0] - self.mask_average_hsv[0]), 360 - abs(secondary_average[1] - self.mask_average_hsv[0])) / 180
		saturation_distance = abs(secondary_average[1] - self.mask_average_hsv[1])
		value_distance = abs(secondary_average[2] - self.mask_average_hsv[2])
		# Getting the vector magnitude of this
		distancing_total = sqrt((hue_distance * hue_distance) + (saturation_distance * saturation_distance) + (
					value_distance * value_distance))
		# Test to see how close the distance is to the maximum allowed distancing
		divisions = config.HSV_FACE_COLOUR_DRIFT / config.MASK_HSV_FACE_COLOUR_DRIFT_STAGES
		totalTesting = divisions
		totalScore = 0
		# The closer the distance is to 0, the closer the colours were in testing
		for x in range(config.MASK_HSV_FACE_COLOUR_DRIFT_STAGES):
			if distancing_total < totalTesting:
				totalScore = totalScore + (1 / config.MASK_HSV_FACE_COLOUR_DRIFT_STAGES)
			totalTesting = totalTesting + divisions
		return totalScore

	def draw_face(self, image, colour):
		# Put in methods to draw a face on an image
		cv.rectangle(image, (self.x, self.y), (self.x + self.width, self.y + self.height), 2)
		try:
			cv.putText(image, "Average Colour: (" + str(self.average_colour)[1:-1] + ")", (self.x, self.y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color=colour)
		except AttributeError:
			pass

class Yolo:

	def __init__(self, objectName):
		self.classID = objectName


class Mask:

	def __init__(self, segment, roi, mask, classID, confidence, coordinates):
		self.segment = segment
		self.roi = roi
		self.mask = mask
		self.classID = classID
		self.confidence = confidence
		self.x1 = coordinates[0]
		self.y1 = coordinates[1]
		self.x2 = coordinates[2]
		self.y2 = coordinates[3]

	def draw_mask(self, image, colour):
		cv.rectangle(image, (self.x1, self.y1), (self.x2, self.y2), 2)
		cv.putText(image, self.classID, (self.x1, self.y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color=colour)


class Pose:

	def __init__(self):
		self.nose = None
		self.neck = None
		self.r_shoulder = None
		self.r_elbow = None
		self.r_wrist = None
		self.r_hip = None
		self.r_knee = None
		self.r_ankle = None
		self.r_eye = None
		self.r_ear = None
		self.l_shoulder = None
		self.l_elbow = None
		self.l_wrist = None
		self.l_hip = None
		self.l_knee = None
		self.l_ankle = None
		self.l_eye = None
		self.l_ear = None

	def draw_pose(self, image):
		"""Draws out all existing points of a pose onto an image"""
		if self.nose:
			cv.circle(image, (self.nose[0], self.nose[1]), 5, [0, 0, 255], -1, cv.LINE_AA)
		if self.neck:
			cv.circle(image, (self.neck[0], self.neck[1]), 5, [0, 0, 255], -1, cv.LINE_AA)
		if self.r_shoulder:
			cv.circle(image, (self.r_shoulder[0], self.r_shoulder[1]), 5, [0, 0, 255], -1, cv.LINE_AA)
		if self.r_elbow:
			cv.circle(image, (self.r_elbow[0], self.r_elbow[1]), 5, [0, 0, 255], -1, cv.LINE_AA)
		if self.r_wrist:
			cv.circle(image, (self.r_wrist[0], self.r_wrist[1]), 5, [0, 0, 255], -1, cv.LINE_AA)
		if self.r_hip:
			cv.circle(image, (self.r_hip[0], self.r_hip[1]), 5, [0, 0, 255], -1, cv.LINE_AA)
		if self.r_knee:
			cv.circle(image, (self.r_knee[0], self.r_knee[1]), 5, [0, 0, 255], -1, cv.LINE_AA)
		if self.r_eye:
			cv.circle(image, (self.r_eye[0], self.r_eye[1]), 5, [0, 0, 255], -1, cv.LINE_AA)
		if self.r_ear:
			cv.circle(image, (self.r_ear[0], self.r_ear[1]), 5, [0, 0, 255], -1, cv.LINE_AA)
		if self.l_shoulder:
			cv.circle(image, (self.l_shoulder[0], self.l_shoulder[1]), 5, [0, 0, 255], -1, cv.LINE_AA)
		if self.l_elbow:
			cv.circle(image, (self.l_elbow[0], self.l_elbow[1]), 5, [0, 0, 255], -1, cv.LINE_AA)
		if self.l_wrist:
			cv.circle(image, (self.l_wrist[0], self.l_wrist[1]), 5, [0, 0, 255], -1, cv.LINE_AA)
		if self.l_hip:
			cv.circle(image, (self.l_hip[0], self.l_hip[1]), 5, [0, 0, 255], -1, cv.LINE_AA)
		if self.l_knee:
			cv.circle(image, (self.l_knee[0], self.l_knee[1]), 5, [0, 0, 255], -1, cv.LINE_AA)
		if self.l_eye:
			cv.circle(image, (self.l_eye[0], self.l_eye[1]), 5, [0, 0, 255], -1, cv.LINE_AA)
		if self.l_ear:
			cv.circle(image, (self.l_ear[0], self.l_ear[1]), 5, [0, 0, 255], -1, cv.LINE_AA)

	def pose_colour_comparisson(self, body_part_colour, body_part):
		divisions = config.AVERAGE_POSE_COLOUR_DRIFT / config.POSE_COLOUR_DRIFT_STAGES
		totalTesting = divisions
		totalScore = 0
		working_colour = None
		if body_part == "Chest":
			working_colour = self.chest_average_bgr
		else:
			working_colour = self.leg_average_bgr
		if type(working_colour) == type(None):
			return 0
		for x in range(config.POSE_COLOUR_DRIFT_STAGES):
			if working_colour[0] > body_part_colour[0] - totalTesting and working_colour[0] < body_part_colour[0] + totalTesting:
				if working_colour[1] > body_part_colour[1] - totalTesting and working_colour[1] < body_part_colour[1] + totalTesting:
					if working_colour[2] > body_part_colour[2] - totalTesting and working_colour[2] < body_part_colour[2] + totalTesting:
						totalScore = totalScore + (1 / config.POSE_COLOUR_DRIFT_STAGES)
			totalTesting = totalTesting + divisions
		return totalScore


	def hsv_pose_colour_comparisson(self, secondary_average, body_part):
		"""Performs maskless hsv colour face comparison"""
		if body_part == "Chest":
			working_colour = self.chest_average_hsv
		else:
			working_colour = self.leg_average_hsv
		if type(working_colour) == type(None):
			return 0
		# Calculating the magnitude of the hues combined
		hue_distance = min(abs(secondary_average[0] - working_colour[0]), 360-abs(secondary_average[1] - working_colour[0])) / 180
		saturation_distance = abs(secondary_average[1] - working_colour[1])
		value_distance = abs(secondary_average[2] - working_colour[2])
		# Getting the vector magnitude of this
		distancing_total = sqrt((hue_distance*hue_distance) + (saturation_distance * saturation_distance) + (value_distance * value_distance))
		# Test to see how close the distance is to the maximum allowed distancing
		divisions = config.POSE_HSV_DRIFT_STAGES / config.HSV_POSE_COLOUR_DRIFT_STAGES
		totalTesting = divisions
		totalScore = 0
		# The closer the distance is to 0, the closer the colours were in testing
		for x in range(config.HSV_POSE_COLOUR_DRIFT_STAGES):
			if distancing_total < totalTesting:
				totalScore = totalScore + (1 / config.HSV_POSE_COLOUR_DRIFT_STAGES)
			totalTesting = totalTesting + divisions
		return totalScore

