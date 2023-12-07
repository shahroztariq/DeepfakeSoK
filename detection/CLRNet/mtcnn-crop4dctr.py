#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from skimage import io
import os
import glob
from PIL import Image

import cv2
from skimage.transform import rotate
from mtcnn.mtcnn import MTCNN
import shutil
import numpy as np


image_format = "*.*"
rect_expand_ratio = 1.2		# ratio expanding face region
# rect_expand_ratio = 1.4		# ratio expanding face region
ignore_ratio = 1.732	# ignore small faces where the w+h < the largest face's (w+h)/(this ratio)

def get_images(image_folder):
	images = []
	for image_path in sorted(glob.glob(os.path.join(image_folder, image_format))):
		filename = os.path.basename(image_path)
		images.append((filename, image_path))
	return images

if __name__ == "__main__":

	if len(sys.argv) < 2:
		print("Usage: python mtcnn-crop4dctr.py 'image folder'")
		sys.exit()

	if not "result" in os.listdir(sys.argv[1]):
		os.mkdir(sys.argv[1]+"/result")

	if not "done_mtcnn" in os.listdir(sys.argv[1]):
		os.mkdir(sys.argv[1]+"/done_mtcnn")

	images = get_images(sys.argv[1])

	detector = MTCNN()

	for (filename, image_path) in images:
		print("loading an image from,", image_path)
		image = io.imread(image_path)
		img_width = image.shape[1]
		img_height = image.shape[0]

		print("detecting faces on image:"+filename)
		result = detector.detect_faces(image)

		max_wh = 0
		for n in range(len(result)):
			if result[n]['box'][2] + result[n]['box'][3] > max_wh:
				max_wh = result[n]['box'][2] + result[n]['box'][3]

		for n in range(len(result)):
			bounding_box = result[n]['box']
			landmark = result[n]['keypoints']
				
			x_center = bounding_box[0] + int(bounding_box[2] / 2)
			y_center = bounding_box[1] + int(bounding_box[3] / 2)

			if bounding_box[2] + bounding_box[3] < max_wh / ignore_ratio:
				continue

			width = int(bounding_box[2] * rect_expand_ratio)
			height = int(bounding_box[3] * rect_expand_ratio)

			left_eye = landmark['left_eye']		# (x, y)
			right_eye = landmark['right_eye']	# (x, y)

			dY = right_eye[1] - left_eye[1]
			dX = right_eye[0] - left_eye[0]
			angle = np.degrees(np.arctan2(dY, dX))

			M = cv2.getRotationMatrix2D((x_center, y_center), angle, 1)
			output = cv2.warpAffine(image, M, (img_width, img_height), flags=cv2.INTER_CUBIC)

			# gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)	# Convert Color to Gray scale
			# gray = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)  # Convert Color to RGB
			gray = output

			face_rect = (x_center - int(width / 2),		#Left 
				y_center - int(height / 2), 		#Top
				x_center + int(width / 2), 		#Right
				y_center + int(height / 2))		#Bottom

			face = Image.fromarray(gray).crop(face_rect)

			min_size = min(width, height)
			image_resize = 8
			while (image_resize + 8) < min_size:
				image_resize += 8

			face = face.resize((image_resize, image_resize), Image.NEAREST)

			f_name, _ = os.path.splitext(filename)
			saved_name = "result/"+f_name+"_"+str(n)+".jpg"
			saved_name = os.path.join(sys.argv[1], saved_name)
			print("saving extracted face image as", saved_name)
			face.save(saved_name)

		print("moving the image file to 'done_mtcnn/'")
		shutil.move(image_path, sys.argv[1]+"/done_mtcnn/")
