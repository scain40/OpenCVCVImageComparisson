# Based on https://www.learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/

import cv2 as cv
import numpy as np
import config


# Formerly keypointMapping
JOINT_NAMES = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

# Defining pairings to be made between peoples joints
POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
						[1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
						[1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
						[2, 17], [5, 16]]

mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44],
          [19,20], [21,22], [23,24], [25,26], [27,28], [29,30],
          [47,48], [49,50], [53,54], [51,52], [55,56],
          [37,38], [45,46]]

class PoseDetection:

	def __init__(self):
		self.default_network_loading()

	def default_network_loading(self):
		self.network = cv.dnn.readNetFromCaffe(config.POSE_WEIGHTS, config.POSE_CONFIG)
		self.expected_points = 18

	def read_image(self, image_name):
		reading_image = cv.imread(image_name)

		self.image_height, self.image_width = reading_image.shape[:2]

		input_height = 368
		input_width = int((input_height/self.image_height) * self.image_width)

		reading_blob = cv.dnn.blobFromImage(reading_image, 1.0 / 255, (input_width, input_height), (0, 0, 0), swapRB=False, crop=False)

		# Reading the passed in image here
		self.network.setInput(reading_blob)
		output = self.network.forward()

		# Processing the limbs that have been detected by the program in here
		self.detected_keypoints = []
		self.keypoints_list = np.zeros((0, 3))
		keypoint_id = 0
		self.named_ungrouped_locations = []
		for component in range(self.expected_points):
			probability_map = output[0, component, :, :]
			probability_map = cv.resize(probability_map, (reading_image.shape[1], reading_image.shape[0]))
			processing_keypoints = self.find_keypoints(probability_map)
			for x in processing_keypoints:
				self.named_ungrouped_locations.append([x, JOINT_NAMES[component]])
			identified_keypoints = []

			for point in range(len(processing_keypoints)):
				identified_keypoints.append(processing_keypoints[point] + (keypoint_id,))
				self.keypoints_list = np.vstack([self.keypoints_list, processing_keypoints[point]])
				keypoint_id += 1

			self.detected_keypoints.append(identified_keypoints)
		valid_pairs, invalid_pairs = self.process_valid_points(output)
		seperate_people_points = self.seperate_people(valid_pairs, invalid_pairs)

		# This section here is dedicated to drawing the pose detection results
		people_point_sets = []
		for x in range(len(seperate_people_points)):
			# Adding a set for everyone that exists
			people_point_sets.append([])
		for x in range(17):
			# This number represents the number of people that exist, could seperat them with this
			for count in range(len(seperate_people_points)):
				index = seperate_people_points[count][np.array(POSE_PAIRS[x])]
				if -1 in index:
					continue # Filtering out unmatched points
				# A and B are just co-ordinate sets mixed between eachother
				B = np.int32(self.keypoints_list[index.astype(int), 0])
				A = np.int32(self.keypoints_list[index.astype(int), 1])
				# For some reason here we have B[0] A[0] making up one location, B[1], A[1] making another
				actual_point1 = [self.keypoints_list[index.astype(int), 0][0], self.keypoints_list[index.astype(int), 1][0]]
				actual_point2 = [self.keypoints_list[index.astype(int), 0][1], self.keypoints_list[index.astype(int), 1][1]]
				# Checking if we already have the points existing in the skeleton
				point1 = False
				point2 = False
				for existing_point in people_point_sets[count]:
					if existing_point[0] == actual_point1[0] and existing_point[1] == actual_point1[0]:
						point1 = True
					if  existing_point[0] == actual_point2[0] and existing_point[1] == actual_point2[1]:
						point2 = True
				# Adding in points that do not exist
				if not point1:
					people_point_sets[count].append(actual_point1)
				if not point2:
					people_point_sets[count].append(actual_point2)
				people_point_sets.append(actual_point1)
				people_point_sets.append(actual_point2)
		# Need to make a new set, rooting out the scalars
		final_people_set = []
		# Creating a final set for everyone that exists
		for x in range(len(seperate_people_points)):
			final_people_set.append([])
		# Looking for what co-ordinate matches which joint, adding that information to the joint
		person_count = 0
		for person in people_point_sets:
			for coords in person:
				# In here compare to the keypoint list
				for named_entry in self.named_ungrouped_locations:
					try:
						if coords[0] == named_entry[0][0] and coords[1] == named_entry[0][1]:
							coords.append(named_entry[1])
							final_people_set[person_count].append(coords)
					except IndexError:
						# This is here to passover the scalar values that get passed through coords
						pass
			person_count = person_count + 1
		return final_people_set

	def find_keypoints(self, probabilities_map):
		# Blurring to make probabilities more likely to be detected
		smooth_map = cv.GaussianBlur(probabilities_map, (3, 3), 0, 0)

		masking_map = np.uint8(smooth_map>config.POSE_DETECTION_THRESHOLD)
		keypoints = []

		contours, _ = cv.findContours(masking_map, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

		for cnt in contours:
			blobing_Mask = np.zeros(masking_map.shape)
			blobing_Mask = cv.fillConvexPoly(blobing_Mask, cnt, 1)
			maskedProbMap = smooth_map * blobing_Mask
			_, maxVal, _, maxLoc = cv.minMaxLoc(maskedProbMap)
			keypoints.append(maxLoc + (probabilities_map[maxLoc[1], maxLoc[0]],))
		return keypoints


	def process_valid_points(self, network_output):
		valid_pairs = []
		invalid_pairs = []
		n_interp_samples = 10
		paf_score_th = 0.1  # Located in config
		conf_th = 0.7
		# loop for every POSE_PAIR
		for k in range(len(mapIdx)):
			# A->B constitute a limb
			pafA = network_output[0, mapIdx[k][0], :, :]
			pafB = network_output[0, mapIdx[k][1], :, :]
			pafA = cv.resize(pafA, (self.image_width, self.image_height))
			pafB = cv.resize(pafB, (self.image_width, self.image_height))

			# Find the keypoints for the first and second limb
			candA = self.detected_keypoints[POSE_PAIRS[k][0]]
			candB = self.detected_keypoints[POSE_PAIRS[k][1]]
			nA = len(candA)
			nB = len(candB)

			if (nA != 0 and nB != 0):
				valid_pair = np.zeros((0, 3))
				for i in range(nA):
					max_j = -1
					maxScore = -1
					found = 0
					for j in range(nB):
						# Find d_ij
						d_ij = np.subtract(candB[j][:2], candA[i][:2])
						norm = np.linalg.norm(d_ij)
						if norm:
							d_ij = d_ij / norm
						else:
							continue
						interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
						                        np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
						paf_interp = []
						for k in range(len(interp_coord)):
							paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
							                   pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))]])
						paf_scores = np.dot(paf_interp, d_ij)
						avg_paf_score = sum(paf_scores) / len(paf_scores)

						# Check if the connection is valid
						if (len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples) > conf_th:
							if avg_paf_score > maxScore:
								max_j = j
								maxScore = avg_paf_score
								found = 1
					# Append the connection to the list
					if found:
						valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

				# Append the detected connections to the global list
				valid_pairs.append(valid_pair)
			else:  # If no keypoints are detected
				# No connections have been found between points when this is reached
				invalid_pairs.append(k)
				valid_pairs.append([])
		return valid_pairs, invalid_pairs


	def seperate_people(self, valid_pairs, invalid_pairs):
		# the last number in each row is the overall score
		personwiseKeypoints = -1 * np.ones((0, 19))

		for pair in range(len(mapIdx)):
			if pair not in invalid_pairs:
				partAs = valid_pairs[pair][:, 0]
				partBs = valid_pairs[pair][:, 1]
				indexA, indexB = np.array(POSE_PAIRS[pair])
				# k = expected point   i = x    j = y
				for x in range(len(valid_pairs[pair])):
					found = 0
					person_idx = -1
					for j in range(len(personwiseKeypoints)):
						if personwiseKeypoints[j][indexA] == partAs[x]:
							person_idx = j
							found = 1
							break

					if found:
						personwiseKeypoints[person_idx][indexB] = partBs[x]
						personwiseKeypoints[person_idx][-1] += self.keypoints_list[partBs[x].astype(int), 2] + \
						                                       valid_pairs[pair][x][2]

					# if find no partA in the subset, create a new subset
					elif not found and pair < 17:
						row = -1 * np.ones(19)
						row[indexA] = partAs[x]
						row[indexB] = partBs[x]
						# add the keypoint_scores for the two keypoints and the paf_score
						row[-1] = sum(self.keypoints_list[valid_pairs[pair][x, :2].astype(int), 2]) + valid_pairs[pair][x][2]
						personwiseKeypoints = np.vstack([personwiseKeypoints, row])
		return personwiseKeypoints
