import YOLOv2 as yolo
import Mask_RCNN as mask
import FaceDetection as face
import PoseDetection as pose
import config
import CommonComponents
import cv2 as cv
import DataStorage
from PIL import ImageTk
from PIL import Image

class mainWindow():

	def __init__(self):
		# Should be used to store all required active components
		self.activeComponents = [True, True, False, True]

	def data_storage_run(self):
		self.image_names = CommonComponents.load_image_names(config.WORKING_FOLDER)
		self.activeComponents = [True, True, True, True]
		if self.activeComponents[0]:
			print('Starting yolo detector')
			self.yoloDetector = yolo.ObjectDetector()
		if self.activeComponents[1]:
			print('Starting mask detector')
			self.maskDetector = mask.MaskRCNN()
		if self.activeComponents[2]:
			print('Starting face detector')
			self.faceDetector = face.FaceDetection()
		if self.activeComponents[3]:
			print('Starting pose detector')
			self.poseDetector = pose.PoseDetection()

		#Creates the full set of results in an array style setup
		self.full_results = []
		# Doing a loop over all images in the folder to find all of our results
		for name in self.image_names:
			new_entry = DataStorage.DataStorage(name.name)
			#Adding the yolo results for our entry
			if config.YOLO_OBJECT_COMPARISSON:
				yolo_results = self.yoloDetector.read_image(name.name)
				for object in yolo_results:
					new_entry.yolo.append(DataStorage.Yolo(object))
			#Adding the mask results for our entry
			if config.FACE_COLOUR_COMPARISSON or config.MASK_OBJECT_COMPARISSON:
				self.maskDetector = mask.MaskRCNN()
				mask_entries = self.maskDetector.read_image(name.name)
				for masking in range(mask_entries[0].__len__()):
					new_entry.mask_results.append(DataStorage.Mask(mask_entries[0][masking], mask_entries[1][masking],
					                                               mask_entries[2][masking], mask_entries[3][masking],
					                                               mask_entries[4][masking], mask_entries[5][masking]))
			if config.FACE_COLOUR_COMPARISSON:
				if config.FACE_DETECTION_TYPE == "Cascade":
					face_entry = self.faceDetector.read_image(name.name)
					for front_face in face_entry[0]:
						new_entry.faces.append(DataStorage.Face(front_face, "Front"))
					for side_face in face_entry[1]:
						new_entry.faces.append(DataStorage.Face(side_face, "Side"))
				elif config.FACE_DETECTION_TYPE == "dlib":
					face_entries = self.faceDetector.facial_recognition_library_read_image(name.name)
					for x in face_entries:
						new_entry.faces.append(DataStorage.Face(x, "Unconfirmed"))
			if self.activeComponents[3]:
				# In here is where pose detection is done
				poses_detected = self.poseDetector.read_image(name.name)
				new_entry.poses = self.pose_processing(name.name, poses_detected)
			self.full_results.append(new_entry)
		# Showing off all the face results here
		if config.DRAW_DISPLAY_IMAGES:
			for entry in self.full_results:
				testing_image = cv.imread(entry.image_name)
				if config.DRAW_FACE_RESULTS:
					for face_entry in entry.faces:
						face_entry.draw_face(testing_image, [255, 255, 255])
				if config.DRAW_MASK_RESULTS:
					for mask_entry in entry.mask_results:
						mask_entry.draw_mask(testing_image, [255, 255, 255])
				if config.DRAW_POSE_RESULTS:
					for pose_entry in entry.poses:
						pose_entry.draw_pose(testing_image)
				if config.FACE_SHADOW_REMOVAL:
					testing_image = CommonComponents.retinex_shadow_removal(testing_image, config.FACE_SHADOW_REMOVAL_TYPE)
				cv.imshow("Testing image", testing_image)
				cv.waitKey(0)

		# Object similarity scoring methods
		# Yolo scoring
		if config.YOLO_OBJECT_COMPARISSON:
			self.yolo_object_comparisson()
		# Mask scoring
		if config.MASK_OBJECT_COMPARISSON:
			self.mask_object_comparisson()

		# Running a maskless face-colour analysis
		if config.FACE_COLOUR_COMPARISSON:
			self.maskless_face_colour_analysis()
			# Calculating average maskless hsv face colours
			self.maskless_hsv_average_calculation()
			# Running a mask-based face_colour analysis
			self.face_average_colour_detection()
			# Making the comparissons of the face-based colour detections
			self.face_average_colour_comparissons()

		# Running a mask-based clothing colour analysis
		if config.CLOTHING_COLOUR_COMPARISSON:
			self.pose_average_colour_detection_open_pose()
			# Running a face-sized base clothing colour analysis
			self.average_clothes_detection_results_comparing()


	def pose_processing(self, image_name, full_pose_detection):
		# NEED TO ROOT OUT DUPLICATED POINTS
		processed_poses = []
		for testing_pose in full_pose_detection:
			new_pose = DataStorage.Pose()
			test_image = cv.imread(image_name)
			# input('This is the legnth of our testing pose')
			for x in testing_pose:
				# input('What is at this level?')
				cv.circle(test_image, (x[0].astype(int), x[1].astype(int)), 5, (0, 0, 0), -1, cv.LINE_AA)
				# Gonna have to use a massive if statement
				if x[2] == 'Nose':
					new_pose.nose = [x[0].astype(int), x[1].astype(int)]
				elif x[2] == 'Neck':
					new_pose.neck = [x[0].astype(int), x[1].astype(int)]
				elif x[2] == 'R-Sho':
					new_pose.r_shoulder = [x[0].astype(int), x[1].astype(int)]
				elif x[2] == 'R-Elb':
					new_pose.r_elbow = [x[0].astype(int), x[1].astype(int)]
				elif x[2] == 'R-Wr':
					new_pose.r_wrist = [x[0].astype(int), x[1].astype(int)]
				elif x[2] == 'L-Sho':
					new_pose.l_shoulder = [x[0].astype(int), x[1].astype(int)]
				elif x[2] == 'L-Elb':
					new_pose.l_elbow = [x[0].astype(int), x[1].astype(int)]
				elif x[2] == 'L-Wr':
					new_pose.l_wrist = [x[0].astype(int), x[1].astype(int)]
				elif x[2] == 'R-Hip':
					new_pose.r_hip = [x[0].astype(int), x[1].astype(int)]
				elif x[2] == 'R-Knee':
					new_pose.r_knee = [x[0].astype(int), x[1].astype(int)]
				elif x[2] == 'R-Ank':
					new_pose.r_ankle = [x[0].astype(int), x[1].astype(int)]
				elif x[2] == 'L-Hip':
					new_pose.l_hip = [x[0].astype(int), x[1].astype(int)]
				elif x[2] == 'L-Knee':
					new_pose.l_knee = [x[0].astype(int), x[1].astype(int)]
				elif x[2] == 'L-Ank':
					new_pose.l_ankle = [x[0].astype(int), x[1].astype(int)]
				elif x[2] == 'R-Eye':
					new_pose.r_eye = [x[0].astype(int), x[1].astype(int)]
				elif x[2] == 'L-Eye':
					new_pose.l_eye = [x[0].astype(int), x[1].astype(int)]
				elif x[2] == 'R-Ear':
					new_pose.r_eat = [x[0].astype(int), x[1].astype(int)]
				elif x[2] == 'L-Ear':
					new_pose.l_ear = [x[0].astype(int), x[1].astype(int)]
			processed_poses.append(new_pose)
			#cv.imshow("DetectedPose", test_image)
			#cv.waitKey(0)
		return processed_poses

	def yolo_object_comparisson(self):
		for results_set in self.full_results:
			for comparing_set in self.full_results:
				if results_set.image_name == comparing_set.image_name:
					continue # Skipping comparing an object to itself
				compared_objects = []
				found_comparissons = 0
				for objects in comparing_set.yolo:
					# No objects in the current image exist in this image if all are still false at the end
					compared_objects.append(objects.classID)
				for object in results_set.yolo:
					for x in range(compared_objects.__len__()):
						if compared_objects[x] == object.classID:
							try:
								compared_objects.remove(object.classID)
								found_comparissons = found_comparissons + 1
								break
							except ValueError:
								pass
				testing_image = results_set.image_name.split("\\")[-1]
				comparing_image_name = comparing_set.image_name.split("\\")[-1]
				print("{x} has this Yolo Object Similarity Rating {y}".format(x=testing_image, y=comparing_image_name))
				print("{x} Objects detected in first image, {y} Objects Detected in second".format(x=results_set.yolo.__len__(), y=comparing_set.yolo.__len__()))
				print("{x} out of {y} Objects of image one detected in image two".format(x=found_comparissons, y=comparing_set.yolo.__len__()))
				print("{x} is the total object similarity rating".format(x=found_comparissons/comparing_set.yolo.__len__()))
				input('Press enter to continue')

	def mask_object_comparisson(self):
		for results_set in self.full_results:
			for comparing_set in self.full_results:
				if results_set.image_name == comparing_set.image_name:
					continue # Skipping comparing an object to itself
				compared_objects = []
				found_comparissons = 0
				for objects in comparing_set.mask_results:
					# No objects in the current image exist in this image if all are still false at the end
					compared_objects.append(objects.classID)
				for object in results_set.mask_results:
					for x in range(compared_objects.__len__()):
						if compared_objects[x] == object.classID:
							compared_objects.remove(object.classID)
							found_comparissons = found_comparissons + 1
							break
				testing_image = results_set.image_name.split("\\")[-1]
				comparing_image_name = comparing_set.image_name.split("\\")[-1]
				print("{x} has this Mask Object Similarity Rating {y}".format(x=testing_image, y=comparing_image_name))
				print("{x} Objects detected in first image, {y} Objects Detected in second".format(x=results_set.mask_results.__len__(), y=comparing_set.mask_results.__len__()))
				print("{x} out of {y} Objects of image one detected in image two".format(x=found_comparissons, y=comparing_set.mask_results.__len__()))
				print("{x} is the total object similarity rating".format(x=found_comparissons/comparing_set.mask_results.__len__()))
				input('Press enter to continue')


	#BEYOND HERE IS THE SEGMENT FOR LOOKING AT POSE DETECTION BASED AVERAGE COLOURS
	def pose_average_colour_detection_open_pose(self):
		for results_set in self.full_results:
			for pose in results_set.poses:
				# Finding the mask that contains the most points for this pose
				selected_mask = None
				selected_contained = 0
				for mask in results_set.mask_results:
					mask_contained = 0
					# This is checking how many pose components are located within a mask box
					if pose.nose and CommonComponents.contained_point(pose.nose, mask):
						mask_contained = mask_contained + 1
					if pose.neck and CommonComponents.contained_point(pose.neck, mask):
						mask_contained = mask_contained + 1
					if pose.r_shoulder and CommonComponents.contained_point(pose.r_shoulder, mask):
						mask_contained = mask_contained + 1
					if pose.r_elbow and CommonComponents.contained_point(pose.r_elbow, mask):
						mask_contained = mask_contained + 1
					if pose.r_wrist and CommonComponents.contained_point(pose.r_wrist, mask):
						mask_contained = mask_contained + 1
					if pose.r_hip and CommonComponents.contained_point(pose.r_hip, mask):
						mask_contained = mask_contained + 1
					if pose.r_knee and CommonComponents.contained_point(pose.r_knee, mask):
						mask_contained = mask_contained + 1
					if pose.r_eye and CommonComponents.contained_point(pose.r_eye, mask):
						mask_contained = mask_contained + 1
					if pose.r_ear and CommonComponents.contained_point(pose.r_ear, mask):
						mask_contained = mask_contained + 1
					if pose.l_shoulder and CommonComponents.contained_point(pose.l_shoulder, mask):
						mask_contained = mask_contained + 1
					if pose.l_elbow and CommonComponents.contained_point(pose.l_elbow, mask):
						mask_contained = mask_contained + 1
					if pose.l_wrist and CommonComponents.contained_point(pose.l_wrist, mask):
						mask_contained = mask_contained + 1
					if pose.l_hip and CommonComponents.contained_point(pose.l_hip, mask):
						mask_contained = mask_contained + 1
					if pose.l_knee and CommonComponents.contained_point(pose.l_knee, mask):
						mask_contained = mask_contained + 1
					if pose.l_eye and CommonComponents.contained_point(pose.l_eye, mask):
						mask_contained = mask_contained + 1
					if pose.l_ear and CommonComponents.contained_point(pose.l_ear, mask):
						mask_contained = mask_contained + 1
					if mask_contained > selected_contained:
						selected_mask = mask
						selected_contained = mask_contained
				if type(selected_mask) == type(None):
					print('Pose found in')
					print(results_set.image_name)
					print(results_set.mask_results.__len__())
					print("None of this number of masks surrounds it's location, skipping colour detection")
					continue
				# Torso components
				chest_average_bgr = [0, 0, 0]
				chest_average_hsv = [0, 0, 0]
				chest_total_hsv = [0, 0, 0]
				chest_total_bgr = [0, 0, 0]
				chest_pixels = 0
				# Shoulders
				chest_total_bgr, chest_total_hsv, chest_pixels = self.add_pose_totals(chest_total_bgr, chest_total_hsv, chest_pixels, self.pose_location_analysis(selected_mask, pose.r_shoulder))
				chest_total_bgr, chest_total_hsv, chest_pixels = self.add_pose_totals(chest_total_bgr, chest_total_hsv, chest_pixels, self.pose_location_analysis(selected_mask, pose.l_shoulder))
				# Wrists
				chest_total_bgr, chest_total_hsv, chest_pixels = self.add_pose_totals(chest_total_bgr, chest_total_hsv, chest_pixels, self.pose_location_analysis(selected_mask, pose.r_wrist))
				chest_total_bgr, chest_total_hsv, chest_pixels = self.add_pose_totals(chest_total_bgr, chest_total_hsv, chest_pixels, self.pose_location_analysis(selected_mask, pose.l_wrist))
				# Neck - Neck always seems to be still on the shirt location
				chest_total_bgr, chest_total_hsv, chest_pixels = self.add_pose_totals(chest_total_bgr, chest_total_hsv, chest_pixels, self.pose_location_analysis(selected_mask, pose.neck))
				# Elbow
				chest_total_bgr, chest_total_hsv, chest_pixels = self.add_pose_totals(chest_total_bgr, chest_total_hsv, chest_pixels, self.pose_location_analysis(selected_mask, pose.l_elbow))
				chest_total_bgr, chest_total_hsv, chest_pixels = self.add_pose_totals(chest_total_bgr, chest_total_hsv, chest_pixels, self.pose_location_analysis(selected_mask, pose.r_elbow))
				# Converting to integer to keep value clarity away from floats
				if chest_pixels > 0:
					chest_average_bgr[0] = (int)(chest_total_bgr[0] / chest_pixels)
					chest_average_bgr[1] = (int)(chest_total_bgr[1] / chest_pixels)
					chest_average_bgr[2] = (int)(chest_total_bgr[2] / chest_pixels)
					chest_average_hsv[0] = (int)(chest_total_hsv[0] / chest_pixels)
					chest_average_hsv[1] = (int)(chest_total_hsv[1] / chest_pixels)
					chest_average_hsv[2] = (int)(chest_total_hsv[2] / chest_pixels)
				# Trouser components
				leg_average_bgr = [0, 0, 0]
				leg_total_bgr = [0, 0, 0]
				leg_total_hsv = [0, 0, 0]
				leg_average_hsv = [0, 0, 0]
				leg_pixels = 0
				# Hip
				leg_total_bgr, leg_total_hsv, leg_pixels = self.add_pose_totals(leg_total_bgr, leg_total_hsv, leg_pixels, self.pose_location_analysis(selected_mask, pose.r_hip))
				leg_total_bgr, leg_total_hsv, leg_pixels = self.add_pose_totals(leg_total_bgr, leg_total_hsv, leg_pixels, self.pose_location_analysis(selected_mask, pose.l_hip))
				# Knee
				leg_total_bgr, leg_total_hsv, leg_pixels = self.add_pose_totals(leg_total_bgr, leg_total_hsv, leg_pixels, self.pose_location_analysis(selected_mask, pose.r_knee))
				leg_total_bgr, leg_total_hsv, leg_pixels = self.add_pose_totals(leg_total_bgr, leg_total_hsv, leg_pixels, self.pose_location_analysis(selected_mask, pose.l_knee))
				# Ankle
				leg_total_bgr, leg_total_hsv, leg_pixels = self.add_pose_totals(leg_total_bgr, leg_total_hsv, leg_pixels, self.pose_location_analysis(selected_mask, pose.r_ankle))
				leg_total_bgr, leg_total_hsv, leg_pixels = self.add_pose_totals(leg_total_bgr, leg_total_hsv, leg_pixels, self.pose_location_analysis(selected_mask, pose.l_ankle))
				# Converting to integer to keep value clarity away from floats
				if leg_pixels < 0:
					leg_average_bgr[0] = (int)(leg_total_bgr[0] / leg_pixels)
					leg_average_bgr[1] = (int)(leg_total_bgr[1] / leg_pixels)
					leg_average_bgr[2] = (int)(leg_total_bgr[2] / leg_pixels)
					leg_average_hsv[0] = (int)(leg_total_hsv[0] / leg_pixels)
					leg_average_hsv[1] = (int)(leg_total_hsv[1] / leg_pixels)
					leg_average_hsv[2] = (int)(leg_total_hsv[2] / leg_pixels)
				pose.leg_average_bgr = leg_average_bgr
				pose.leg_average_hsv = leg_average_hsv
				pose.chest_average_bgr = chest_average_bgr
				pose.chest_average_hsv = chest_average_hsv

	def pose_location_analysis(self, checking_mask, limb_node):
		try:
			detected_total_colour = [0, 0, 0]
			detected_total_hsv = [0, 0, 0]
			checked_pixels = 0
			hsv_roi = cv.cvtColor(checking_mask.roi, cv.COLOR_BGR2HSV)
			# CONVERT THIS STUFF TO NEW CO-ORDINATES TO GET WORKING SET
			start_x = (limb_node[0] - config.POSE_DETECTION_RADIUS) - checking_mask.x1
			start_y = (limb_node[1] - config.POSE_DETECTION_RADIUS) - checking_mask.y1
			end_x = (limb_node[0] + config.POSE_DETECTION_RADIUS) - checking_mask.x1
			end_y = (limb_node[1] + config.POSE_DETECTION_RADIUS) - checking_mask.y1
			for x in range(start_x, end_x):
				for y in range(start_y, end_y):
					try:
					# CHECKING OF THE MASK IS DONE IN A Y BY X BASIS
						if checking_mask.mask[y][x] == 255:
							detected_total_colour[0] = detected_total_colour[0] + checking_mask.roi[y][x][0]
							detected_total_colour[1] = detected_total_colour[1] + checking_mask.roi[y][x][1]
							detected_total_colour[2] = detected_total_colour[2] + checking_mask.roi[y][x][2]
							detected_total_hsv[0] = detected_total_hsv[0] + hsv_roi[y][x][0]
							detected_total_hsv[1] = detected_total_hsv[1] + hsv_roi[y][x][1]
							detected_total_hsv[2] = detected_total_hsv[2] + hsv_roi[y][x][2]
							checked_pixels = checked_pixels + 1
					except IndexError:
						# Occurs if we check outside the expected mask, skips over
						pass
			return [checked_pixels, detected_total_colour, detected_total_hsv]
		except TypeError:
			# This will get returned if the limb does not exist
			return [0, [0, 0, 0], [0, 0, 0]]

	def add_pose_totals(self, total, total_hsv, count, location_analysis_result):
		total[0] = total[0] + location_analysis_result[1][0]
		total[1] = total[1] + location_analysis_result[1][1]
		total[2] = total[2] + location_analysis_result[1][2]
		total_hsv[0] = total_hsv[0] + location_analysis_result[2][0]
		total_hsv[1] = total_hsv[1] + location_analysis_result[2][1]
		total_hsv[2] = total_hsv[2] + location_analysis_result[2][2]
		count = count + location_analysis_result[0]
		return [total, total_hsv, count]


	def average_clothes_detection_results_comparing(self):
		for results_set in self.full_results:
			for comparing_set in self.full_results:
				if results_set.image_name == comparing_set.image_name:
					# Skipping over compaing somethings values to itself
					continue
				legs_tested = 0
				legs_total_score = 0
				legs_total_hsv_score = 0
				torsos_total_score = 0
				torsos_total_hsv_score = 0
				torsos_tested = 0
				for pose in results_set.poses:
					for comparing_poses in comparing_set.poses:
						try:
							legs_total_score = legs_total_score + pose.pose_colour_comparisson(comparing_poses.leg_average_bgr, "Leg")
							legs_total_hsv_score = legs_total_hsv_score + pose.hsv_pose_colour_comparisson(comparing_poses.leg_average_hsv, "Leg")
							legs_tested = legs_tested + 1
						except AttributeError:
							pass # Happens if leg is not detected
						try:
							torsos_total_score = torsos_total_score + pose.pose_colour_comparisson(comparing_poses.chest_average_bgr, "Chest")
							torsos_total_hsv_score = torsos_total_hsv_score + pose.hsv_pose_colour_comparisson(comparing_poses.chest_average_hsv, "Chest")
							torsos_tested = torsos_tested + 1
						except AttributeError:
							pass # Happens if a torso is not detected
				average_leg_score = 0
				average_torso_score = 0
				average_leg_hsv_score = 0
				average_torso_hsv_score = 0
				if legs_total_score != 0.0 and legs_total_score > 0.0:
					average_leg_score = legs_total_score / legs_tested
				if legs_total_hsv_score != 0.0 and legs_total_hsv_score > 0.0:
					average_leg_hsv_score = legs_total_hsv_score / legs_tested
				if torsos_total_score != 0.0 and torsos_total_score > 0.0:
					average_torso_score = torsos_total_score / torsos_tested
				if torsos_total_hsv_score != 0.0 and torsos_total_hsv_score > 0.0:
					average_torso_hsv_score = torsos_total_hsv_score / torsos_tested
				testing_image = results_set.image_name.split("\\")[-1]
				comparing_image_name = comparing_set.image_name.split("\\")[-1]
				print("{x} has these average pose colour comparissons to {y}".format(x=testing_image, y=comparing_image_name))
				print("{x} Poses detected in first image, {y} detected in second".format(x=results_set.poses.__len__(), y=comparing_set.poses.__len__()))
				print("Leg RGB: {x} \nChest RGB: {y}".format(x=average_leg_score, y=average_torso_score))
				#print("Leg HSV: {x} \nChest HSV: {y}".format(x=average_leg_hsv_score, y=average_torso_hsv_score))
				input('Press enter to continue')

	#BEYOND HERE ARE THE CALCULATIONS FOR FINDING AVERAGE FACE COLOUR BASED ON A MASK
	def face_average_colour_detection(self):
		for results_set in self.full_results:
			for face in results_set.faces:
				# Checking if the center of the face is in a detected mask
				center_x = face.x + (face.width/2)
				center_y = face.y + (face.height/2)
				located_in_mask = None
				for mask in results_set.mask_results:
					# Switch these if a face is not within a detected mask
					# Switch round center_y and center_x if they do not work
					if CommonComponents.contained_point([center_y, center_x], mask):
						located_in_mask = mask
						break
				if not located_in_mask:
					print('Face found in')
					print(results_set.image_name)
					print(results_set.mask_results.__len__())
					print("There is no mask that contains it's location, skipping colour detection")
					continue
				face_average_bgr = self.face_location_analysis(located_in_mask, face)

	def face_location_analysis(self, checking_mask, face_node):
		# Calculating the co-ordinates of the face in the image
		start_x = face_node.x - checking_mask.x1
		start_y = face_node.y - checking_mask.y1
		end_x = face_node.x + face_node.width - checking_mask.x1
		end_y = face_node.y + face_node.height - checking_mask.y1

		# Printing out the stats of the location we are looking at
		detected_total_bgr = [0, 0, 0]
		detected_total_hsv = [0, 0, 0]
		checked_pixels = 0
		cv.rectangle(checking_mask.roi, (start_x, start_y), (end_x, end_y), 2)
		hsv_mask = cv.cvtColor(checking_mask.roi, cv.COLOR_BGR2HSV)
		standard_mask = checking_mask.roi
		if config.FACE_SHADOW_REMOVAL:
			hsv_mask = CommonComponents.retinex_shadow_removal(hsv_mask, config.FACE_SHADOW_REMOVAL_TYPE)
			standard_mask = CommonComponents.retinex_shadow_removal(standard_mask, config.FACE_SHADOW_REMOVAL_TYPE)
		for x in range(start_x, end_x):
			for y in range(start_y, end_y):
				try:
					if checking_mask.mask[y][x] == 255:
						detected_total_bgr[0] = detected_total_bgr[0] + standard_mask[y][x][0]
						detected_total_bgr[1] = detected_total_bgr[1] + standard_mask[y][x][1]
						detected_total_bgr[2] = detected_total_bgr[2] + standard_mask[y][x][2]
						detected_total_hsv[0] = detected_total_hsv[0] + hsv_mask[y][x][0]
						detected_total_hsv[1] = detected_total_hsv[1] + hsv_mask[y][x][1]
						detected_total_hsv[2] = detected_total_hsv[2] + hsv_mask[y][x][2]
						checked_pixels = checked_pixels + 1
				except IndexError:
					# Skipping locations not contained within the mask
					pass
		average_bgr = [0, 0, 0]
		average_hsv = [0, 0, 0]
		if checked_pixels != 0:
			average_bgr[0] = (int)(detected_total_bgr[0] / checked_pixels)
			average_bgr[1] = (int)(detected_total_bgr[1] / checked_pixels)
			average_bgr[2] = (int)(detected_total_bgr[2] / checked_pixels)
			average_hsv[0] = (int)(detected_total_hsv[0] / checked_pixels)
			average_hsv[1] = (int)(detected_total_hsv[1] / checked_pixels)
			average_hsv[2] = (int)(detected_total_hsv[2] / checked_pixels)
		face_node.mask_average_bgr = average_bgr
		face_node.mask_average_hsv = average_hsv


	def maskless_face_colour_analysis(self):
		for results_set in self.full_results:
			testingImage = cv.imread(results_set.image_name)
			if config.FACE_SHADOW_REMOVAL:
				testingImage = CommonComponents.retinex_shadow_removal(testingImage, config.FACE_SHADOW_REMOVAL_TYPE)
			for face in results_set.faces:
				averageBGR = [0, 0, 0]
				totalBGR = [0, 0, 0]
				countedPixels = 0
				for x in range(face.x, face.x + face.width):
					for y in range(face.y, face.y + face.height):
						totalBGR[0] = totalBGR[0] + testingImage[y][x][0]
						totalBGR[1] = totalBGR[1] + testingImage[y][x][1]
						totalBGR[2] = totalBGR[2] + testingImage[y][x][2]
						countedPixels = countedPixels + 1
				if countedPixels != 0:
					averageBGR[0] = round(totalBGR[0] / countedPixels)
					averageBGR[1] = round(totalBGR[1] / countedPixels)
					averageBGR[2] = round(totalBGR[2] / countedPixels)
				face.average_colour = averageBGR

	def maskless_hsv_average_calculation(self):
		for results_set in self.full_results:
			testing_image = cv.imread(results_set.image_name)
			if config.FACE_SHADOW_REMOVAL:
				testing_image = CommonComponents.retinex_shadow_removal(testing_image, config.FACE_SHADOW_REMOVAL_TYPE)
			testing_image = cv.cvtColor(testing_image, cv.COLOR_BGR2HSV)
			for face in results_set.faces:
				totalHSV = [0, 0, 0]
				averageHSV = [0, 0, 0]
				countedPixels = 0
				for x in range(face.x, face.x + face.width):
					for y in range(face.y, face.y + face.height):
						totalHSV[0] = totalHSV[0] + testing_image[y][x][0]
						totalHSV[1] = totalHSV[1] + testing_image[y][x][1]
						totalHSV[2] = totalHSV[2] + testing_image[y][x][2]
						countedPixels = countedPixels + 1
				if countedPixels != 0 :
					averageHSV[0] = (int)(totalHSV[0] / countedPixels)
					averageHSV[1] = (int)(totalHSV[1] / countedPixels)
					averageHSV[2] = (int)(totalHSV[2] / countedPixels)
					face.average_hsv = averageHSV
				else:
					face.average_hsv = [0, 0, 0]

	def face_colour_comparissons(self, base_image):
		# Need to select an image and calculate a total similarity based on an allowed difference level
		workingEntry = 0
		comparisson = []
		for entry in self.full_results:
			if entry.image_name == base_image:
				workingEntry = entry
		if workingEntry != 0:
			# Compare the averages of all faces in here
			for comparisson_entry in self.full_results:
				if comparisson_entry.image_name != entry.image_name:
					matching_faces = 0
					different_faces = 0
					for base_face in workingEntry.faces:
						for comparisson_face in comparisson_entry.faces:
							try:
								if base_face.average_colour_comparison(comparisson_face.average_colour):
									matching_faces = matching_faces + 1
								else:
									different_faces = different_faces + 1
							except AttributeError:
								pass # If average face colour not detected properly
					if different_faces != 0 or matching_faces != 0:
						average_similarity = (1 / (matching_faces + different_faces)) * matching_faces
						if average_similarity > 0.0:
							print("This is our average similarity")
							print(average_similarity)
							# input('Do we think this is good for comparing?')
		else:
			input('We have no image of the base_image name here')


	def face_colour_comparissons_scored(self, base_image):
		# Need to select an image and calculate a total similarity based on an allowed difference level
		workingEntry = 0
		comparisson = []
		for entry in self.full_results:
			if entry.image_name == base_image:
				workingEntry = entry
		if workingEntry != 0:
			# Compare the averages of all faces in here
			for comparisson_entry in self.full_results:
				if comparisson_entry.image_name != entry.image_name:
					faces_tested = 0
					faces_score = 0
					hsv_score = 0
					for base_face in workingEntry.faces:
						for comparisson_face in comparisson_entry.faces:
							try:
								faces_score = base_face.average_colour_comparison_scoring(comparisson_face.average_colour)
								hsv_score = base_face.hsv_face_colour_comparisson(comparisson_face.average_hsv)
								faces_tested = faces_tested + 1
							except AttributeError:
								pass # If average face colour not detected properly

					if faces_score != 0.0 and faces_score > 0.0:
						faces_score = faces_score/faces_tested
						print("This is our average similarity")
						print(faces_score)
					if hsv_score != 0.0 and hsv_score > 0.0:
						hsv_score = hsv_score/faces_tested
						print("This is our average hsv distance similarity")
						print(hsv_score)
				comparisson.append([comparisson_entry.image_name, faces_score, hsv_score])
		else:
			input('We have no image of the base_image name here')
		return comparisson

	def face_average_colour_comparissons(self):
		"""This method compares all hsv and bgr face colour comparissons and prints the result as a text format"""
		for results_set in self.full_results:
			for comparing_set in self.full_results:
				if results_set.image_name == comparing_set.image_name:
					# Skipping comparing an image to itself
					continue
				faces_tested = 0
				faces_score = 0
				faces_score_maskless = 0
				hsv_score_maskless = 0
				masked_faces_tested = 0
				hsv_score = 0
				for base_face in results_set.faces:
					for comparisson_face in comparing_set.faces:
						#try:
						# Calculating maskless comparissons
						try:
							faces_score_maskless = faces_score_maskless + base_face.average_colour_comparison_scoring(comparisson_face.average_colour)
							hsv_score_maskless = hsv_score_maskless + base_face.hsv_face_colour_comparisson(comparisson_face.average_hsv)
							faces_tested = faces_tested + 1
						except IndexError:
							print('A maskless score does not exist in this image test')
						# Calculating mask based comparissons
						try:
							faces_score = faces_score + base_face.masked_average_colour_comparison_scoring(comparisson_face.mask_average_bgr)
							hsv_score = hsv_score + base_face.masked_hsv_face_colour_comparisson(comparisson_face.mask_average_hsv)
							masked_faces_tested = masked_faces_tested + 1
						except AttributeError:
							print('A score in here does not have a mask based average')
				# Getting the true number of faces tested since we seem to end up with dobule the value
				# Resetting is done afterwards if the value detected is above 1 to prevent rounding errors being a problem
				if results_set.faces.__len__() != 0 and comparing_set.faces.__len__() != 0:
					faces_tested = faces_tested / ((results_set.faces.__len__() + comparing_set.faces.__len__()) / 2)
					masked_faces_tested = masked_faces_tested / ((results_set.faces.__len__() + comparing_set.faces.__len__()) / 2)
				if faces_score_maskless != 0.0 and faces_score_maskless > 0.0:
					faces_score_maskless = faces_score_maskless/faces_tested
					if faces_score_maskless > 1:
						faces_score_maskless = 1
				if hsv_score_maskless != 0.0 and hsv_score_maskless > 0.0:
					hsv_score_maskless = hsv_score_maskless/faces_tested
					if hsv_score_maskless > 1:
						hsv_score_maskless = 1
				# Dividing masked scores by faces detected
				if faces_score != 0.0 and faces_score > 0.0:
					faces_score = faces_score/masked_faces_tested
					if faces_score > 1:
						faces_score = 1
				if hsv_score != 0.0 and hsv_score > 0.0:
					hsv_score = hsv_score/masked_faces_tested
					if hsv_score > 1:
						hsv_score = 1
				# Final adjustments to get scores to proper locations
				testing_image = results_set.image_name.split("\\")[-1]
				comparing_image_name = comparing_set.image_name.split("\\")[-1]
				print("{x} has these average face colour relations to {y}".format(x=testing_image, y=comparing_image_name))
				print("{x} is the number of faces detected in {y}".format(x=testing_image, y=results_set.faces.__len__()))
				print("Maskless RGB: {x} \nMaskless HSV: {y}".format(x=faces_score_maskless, y=hsv_score_maskless))
				print("{x} is the number of masks detected in".format(x=results_set.mask_results.__len__()))
				print("Masked RGB: {x} \nMasked HSV: {y}".format(x=faces_score, y=hsv_score))
				input('Hit enter to continue\n')


	def draw_all_results(self, image_number):
		showing_image = cv.imread(self.full_results[image_number].image_name)
		for face in self.full_results[image_number].faces:
			face.draw_face(showing_image, (0, 0, 0))
		for mask in self.full_results[image_number].mask_results:
			mask.draw_mask(showing_image, (0, 255, 0))
		cv.imshow("Image Detections", showing_image)
		cv.waitKey(0)
		return showing_image






main = mainWindow()
# main.start()
main.data_storage_run()

