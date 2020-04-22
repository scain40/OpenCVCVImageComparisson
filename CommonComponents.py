import os
import sys
import retinex
import json
import DataStorage


def load_image_names(image_folder):
	image_names = []
	for fileName in os.listdir(os.getcwd() + "\\" + image_folder):
		image_names.append(open(os.getcwd() + "\\" + image_folder + "\\" + fileName))
	return image_names


def retinex_shadow_removal(image, retinex_type):
	with open('config.json', 'r') as f:
		config = json.load(f)
	if retinex_type == 'automatedMSRCR':
		return retinex.automatedMSRCR( image, config['sigma_list'])
	elif retinex_type == 'MSRCRP':
		return retinex.MSRCP(image, config['sigma_list'], config['low_clip'], config['high_clip'])
	elif retinex_type == 'MSRCR':
		return retinex.MSRCR(image, config['sigma_list'], config['G'], config['b'], config['alpha'], config['beta'], config['low_clip'], config['high_clip'])
	else:
		return image


def aabb_location_comparisson(one, two):
	if one[0] < two[2] and one[2] > two[0]:
		if one[1] < two[3] and one[3] > two[1]:
			return True
	return False


def mask_face_aabb(mask, face):
	if face.x < mask.x2 and (face.x + face.width) > mask.x1:
		if face.y < mask.y2 and (face.y + face.height) > mask.y1:
			return True
	return False


def masking_check(pixelMasking, wantedColour):
	if pixelMasking == wantedColour[0] and pixelMasking == wantedColour[1] and pixelMasking == wantedColour[2]:
		return True
	else:
		return False

def masking_(pixelMasking, wantedColour):
	if pixelMasking == wantedColour[0] and pixelMasking == wantedColour[1] and pixelMasking == wantedColour[2]:
		return True
	else:
		return False


def contained_point(point, mask):
	"""Checks to see if a point is contained within the coordinates of a mask"""
	if point[0] < mask.x2 and point[0] > mask.x1 and point[1] < mask.y2 and point[1] > mask.y1:
		return True
	return False


def contained_point2(point, mask):
	"""Checks to see if a point is contained within the coordinates of a mask"""
	if point[1] < mask.x2 and point[1] > mask.x1 and point[0] < mask.y2 and point[0] > mask.y1:
		return True
	return False

