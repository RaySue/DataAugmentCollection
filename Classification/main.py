# coding:utf-8 
# @Time : 05/03/2018 18:17
# @Author : SuRui

# from keras.preprocessing.image import ImageDataGenerator
import os
import math
import time
import random
import logging
import threading
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageEnhance, ImageOps, ImageFile

# logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True

"""
Deep learning image augmentation
cited from https://scottontechnology.com/flip-image-opencv-DL_framework/
http://augmentor.readthedocs.io/en/master/userguide/mainfeatures.html
"""
"""数据增强
   1. 翻转变换 flip
   2. 随机修剪 random crop
   3. 色彩抖动 color jittering
   4. 平移变换 shift
   5. 尺度变换 scale
   6. 对比度变换 contrast
   7. 噪声扰动 noise
   8. 旋转变换/反射变换 Rotation/reflection
   9. 模糊 Guassian Blur
   10.拉伸变换(Current only for OCR)
"""


class DataAugmentation:
	"""
	包含数据增强的八种方式
	"""
	
	# TODO: 新方法，可以把图片以一定的正弦变换来处理从而得到更多的样本。比如y=sin(x)这个曲线，可以让图片的每个像素的
	#       i + sin(x)
	#
	#
	
	
	"""
	对单张图片进行笛卡尔积式的生成
	1. 先对原图进行各种扭曲变换
	2. 然后对所有扭曲变换的图片进行颜色的随机
	3. 然后对所有生成图片进行随机高斯模糊
	
	对多张图片采取随机抽样然后少量笛卡尔积式样的生成
	"""
	
	def __init__(self):
		
		self.timeMap = {
			"randomRotation": 3,
			"randomWarp": 2,
			# "randomGaussianBlur": 5
		}
		
		self.funcMap = {
			"randomRotation": DataAugmentation.randomRotation,
			# 	"randomCrop": DataAugmentation.randomCrop,
			"randomWarp": DataAugmentation.randomWarp,
			# "randomGaussianBlur": DataAugmentation.randomGaussianBlur,
			# "randomColor": DataAugmentation.randomColor
			# "randomGaussian": DataAugmentation.randomGaussian
		}
		
		self.opsList = self.timeMap.keys()
	
	@property
	def get_fun_map(self):
		return self.funcMap
	
	@property
	def getTimeMap(self):
		return self.timeMap
	
	@property
	def getOpsList(self):
		return self.opsList
	
	@staticmethod
	def openImage(image):
		# return cv2.imread(image)
		return Image.open(image, mode="r")
	
	@staticmethod
	def saveImage(image, path):
		image.save(path)
	
	@staticmethod
	def randomRotation0(image, mode=Image.BICUBIC):
		"""
		 对图像进行随机任意角度(0~360度)旋转
		:param mode 邻近插值,双线性插值,双三次B样条插值(default)
		:param image PIL的图像image
		:return: 旋转转之后的图像
		"""
		random_angle = np.random.randint(-10, 10)
		return image.rotate(random_angle, mode)
	
	# 仿射变换(旋转) type 0表示角度
	@staticmethod
	def randomRotation(image):
		src = np.asarray(image)
		w = src.shape[1]
		h = src.shape[0]
		scale = 1.0
		type = 0
		def_color = 255
		angle_src = np.random.randint(-20, 20)
		if type == 0:
			rangle = np.deg2rad(angle_src)  # angle in radians
			angle = angle_src
		else:
			rangle = angle_src
			angle = np.rad2deg(angle_src)
		# now calculate new image width and height
		nw = w  # (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
		nh = h  # (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
		# ask OpenCV for the rotation matrix
		rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
		# calculate the move from the old center to the new center combined
		# with the rotation
		rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
		# the move only affects the translation, so update the translation
		# part of the transform
		rot_mat[0, 2] += rot_move[0]
		rot_mat[1, 2] += rot_move[1]
		img_out = cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4,
		                         borderMode=cv2.BORDER_CONSTANT, borderValue=def_color)
		return Image.fromarray(np.uint8(img_out))
	
	@staticmethod
	def randomCrop(image):
		"""
		对图像随意剪切,考虑到图像大小范围(68,68),使用一个一个大于(36*36)的窗口进行截图
		:param image: PIL的图像image
		:return: 剪切之后的图像

		"""
		image_width = image.size[0]
		image_height = image.size[1]
		crop_win_size = np.random.randint(40, 68)
		random_region = (
			(image_width - crop_win_size) >> 1, (image_height - crop_win_size) >> 1, (image_width + crop_win_size) >> 1,
			(image_height + crop_win_size) >> 1)
		return image.crop(random_region)
	
	@staticmethod
	def randomColor(image):
		"""
		对图像进行颜色抖动
		:param image: PIL的图像image
		:return: 有颜色色差的图像image
		"""
		random_factor = np.random.randint(0, 62) / 10.  # 随机因子
		color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
		#
		random_factor = np.random.randint(10, 21) / 10.  # 随机因子
		brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
		#
		random_factor = np.random.randint(10, 42) / 10.  # 随机因子
		contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
		#
		random_factor = np.random.randint(0, 62) / 10.  # 随机因子
		return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度
	
	@staticmethod
	def randomWarp(image):
		"""
		OCR data augmentation
		对图像进行拉伸变换
		:param image:
		:return:
		"""
		img = np.asarray(image)
		color = np.min(img) + (np.max(img) - np.min(img))
		def_color = [int(color)]
		# random.choice
		out_height, out_width = img.shape[:2]
		offset_list = [-1, 0, 1]
		pts_in = [[0 + random.choice(offset_list), 0 + random.choice(offset_list)],
		          [out_width + random.choice(offset_list), 2 + random.choice(offset_list)],
		          [2 + random.choice(offset_list), out_height + random.choice(offset_list)],
		          [out_width + random.choice(offset_list), out_height + random.choice(offset_list)]]
		
		pts_out = [[0.0, 0.0], [out_width, 0.0], [0.0, out_height], [out_width, out_height]]  # 左上、右上、左下、右下
		M_perspective = cv2.getPerspectiveTransform(np.float32(pts_in), np.float32(pts_out))
		img_out = cv2.warpPerspective(img, M_perspective, (out_width, out_height), None, 2,
		                              cv2.BORDER_CONSTANT, def_color)
		return Image.fromarray(np.uint8(img_out))
	
	@staticmethod
	def randomGaussianBlur(image):
		"""
		对图像进行高斯滤波，使图像变得模糊
		:param image:
		:return:
		"""
		kernal_size_list = [3, 5, 7, 9, 1]
		kernal_size = random.choice(kernal_size_list)
		sigmaX = random.choice(range(10, 130, 15))
		sigma = sigmaX * 1.0 / 100
		# print kernal_size
		# print sigma
		img = np.asarray(image)
		img = cv2.GaussianBlur(img, (kernal_size, kernal_size), sigma)
		# return imgs
		return Image.fromarray(np.uint8(img))
	
	@staticmethod
	def randomGaussianRGB(image, mean=0.2, sigma=0.3):
		"""
		For rgb image
		对图像进行高斯噪声处理
		:param image:
		:return:
		"""
		
		def gaussianNoisy(im, mean=0.2, sigma=0.3):
			"""
			对图像做高斯噪音处理
			:param im: 单通道图像
			:param mean: 偏移量
			:param sigma: 标准差
			:return:
			"""
			for _i in range(len(im)):
				im[_i] += random.gauss(mean, sigma)
			return im
		
		# # 将图像转化成数组
		img = np.asarray(image)
		img.flags.writeable = True  # 将数组改为读写模式
		width, height = img.shape[:2]
		img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
		img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
		img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
		img[:, :, 0] = img_r.reshape([width, height])
		img[:, :, 1] = img_g.reshape([width, height])
		img[:, :, 2] = img_b.reshape([width, height])
		return Image.fromarray(np.uint8(img))


def makeDir(path):
	try:
		if not os.path.exists(path):
			if not os.path.isfile(path):
				# os.mkdir(path)
				os.makedirs(path)
			return 0
		else:
			return 1
	except Exception as e:
		print(str(e))
		return -2


def imageOps(func_name, image, des_path, file_name, timeMap):
	dataAug = DataAugmentation()
	funcMap = dataAug.get_fun_map
	
	if funcMap.get(func_name) is None:
		print("%s is not exist", func_name)
		# logger.error("%s is not exist", func_name)
		return -1
	
	for _i in range(0, timeMap.get(func_name), 1):
		new_image = funcMap[func_name](image)
		body_name = get_name_body(file_name)  # file_name.split("_")[0]
		new_file_name = body_name + "_" + func_name + str(_i) + ".jpg"
		DataAugmentation.saveImage(new_image, os.path.join(des_path, new_file_name))


def get_name_body(basename):
	name_arr = basename.split(".")
	body_name = ""
	duan_count = len(name_arr)
	if duan_count > 2:
		for i in range(duan_count - 2):
			body_name += name_arr[i] + "."
		body_name += name_arr[duan_count - 2]
	elif duan_count == 2:
		body_name = name_arr[0]
	elif duan_count == 1:
		body_name = basename
	return body_name


def runThreadOPS(path, new_path):
	augmentor = DataAugmentation()
	# 多线程处理事务
	# :param src_path: 资源文件
	# :param des_path: 目的地文件
	# :return:
	if os.path.isdir(path):
		img_names = os.listdir(path)
	else:
		img_names = [path.split("/")[-1]]
		path = "/".join(path.split("/")[:-1])
	
	for img_name in img_names:
		print(img_name)
		tmp_img_name = os.path.join(path, img_name)
		if os.path.isdir(tmp_img_name):
			if makeDir(os.path.join(new_path, img_name)) != -1:
				runThreadOPS(tmp_img_name, os.path.join(new_path, img_name))
			else:
				print('create new dir failure')
				return -1
				# os.removedirs(tmp_img_name)
		
		elif tmp_img_name.split('.')[1] != "DS_Store":
			# 读取文件并进行操作
			image = DataAugmentation.openImage(tmp_img_name)
			
			# debug
			# im = DataAugmentation.randomWarp(image)
			# plt.figure(1)
			# plt.imshow(im, cmap='gray')
			# plt.show()
			
			threadImage = [0] * 20
			_index = 0
			for ops_name in augmentor.getOpsList:
				arg_input = (ops_name, image, new_path, img_name, augmentor.getTimeMap,)
				threadImage[_index] = threading.Thread(target=imageOps, args=arg_input)
				threadImage[_index].start()
				_index += 1
				time.sleep(0.1)


if __name__ == '__main__':
	src_dir = "./input"
	files = os.listdir(src_dir)
	for file in files:
		path_i = os.path.join(src_dir, file)
		if file[0] in ['8']:  # '1', '9'
			runThreadOPS(path_i, u"./out_dir")
