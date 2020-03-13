#coding=utf-8
import cv2
import numpy as np
import math
import random
from PIL import Image, ImageEnhance, ImageOps, ImageFile
import os
import glob
import argparse
import sys
import xml.etree.ElementTree as ET
import imgaug.augmenters as iaa

# imgpath = sys.argv[1]
# annopath = sys.argv[2]

imgpath = "D:\data\ConditionMonitor\onlybody\img" # "D:\data\pigbody\JPEGImagesblackbody" #""D:\\vmdata\origins_collect\ybtrough2"
annopath = "D:\data\ConditionMonitor\onlybody\label_check" #"D:\data\pigbody\Annotationsblackbody" #"D:\\vmdata\origins_collect\ybtroughxml2"
destimgpath = imgpath + "_aug"
destannopath = annopath + "_aug"
os.system("del %s"% destimgpath)
os.system("del %s"% destannopath)
if not os.path.exists(destimgpath):
    os.mkdir(destimgpath)

if not os.path.exists(destannopath):
    os.mkdir(destannopath)


def EqExtension(src):
    I_backup = src.copy()
    b, g, r = cv2.split(I_backup)
    b = cv2.equalizeHist(b)
    g = cv2.equalizeHist(g)
    r = cv2.equalizeHist(r)

    I_eq = cv2.merge([b, g, r])

    return I_eq

def HueExtension(src):
    img_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    cre = random.randint(90, 100)
    cre = float(cre) / 100
    img_hsv[:,:,2] = img_hsv[:,:,2] * cre

    # print(img_hsv[:,:,0])
    dst = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return dst

def flip_img(img,scale=1.0,direction=0):
    img_flip = cv2.flip(img,direction)
    l = img.shape
    img_flip = cv2.resize(img_flip,(int(l[1]*scale),int(l[0]*scale)))
    return img_flip

def flip_box(src,originbox,scale=1.0,direction=0): #todo
    l = src.shape
    boxnum = int(len(originbox)/4)
    resultbox = []
    if direction == 1:
        for i in range(boxnum):
            flip_rec = [l[1]-originbox[2+4*i],originbox[1 + 4*i],l[1]-originbox[0 + 4*i],originbox[3 + 4*i]]
            resultbox.extend(flip_rec)
        # flip_rec = [l[1]-originbox[2],originbox[1],l[1]-originbox[0],originbox[3]]
    if direction == 0:
        for i in range(boxnum):
            flip_rec = [originbox[0 + 4*i],l[0] - originbox[3 + 4*i],originbox[2 + 4*i],l[0] - originbox[1 + 4*i]]
            resultbox.extend(flip_rec)
        # flip_rec = [originbox[0],l[0] - originbox[3],originbox[2],l[0] - originbox[1]]
    resultbox = [int(i*scale) for i in resultbox]
    return resultbox


def rotate_image(img, angle, scale=1.):
    w = img.shape[1]
    h = img.shape[0]
    # 角度变弧度
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height

    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    # 仿射变换
    return cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

def rotate_xml(src,originbox , angle, scale=1.):
    boxnum = int(len(originbox) /4)
    resultbox = []
    for i in range(boxnum):
        xmin, ymin, xmax, ymax = originbox[4*i:4*i+4]
        w = src.shape[1]
        h = src.shape[0]
        rangle = np.deg2rad(angle)  # angle in radians
        # now calculate new image width and height
        # 获取旋转后图像的长和宽
        nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
        nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0,2] += rot_move[0]
        rot_mat[1,2] += rot_move[1]
        # rot_mat是最终的旋转矩阵
        # 获取原始矩形的四个中点，然后将这四个点转换到旋转后的坐标系下
        point1 = np.dot(rot_mat, np.array([xmin , ymin, 1]))
        point2 = np.dot(rot_mat, np.array([xmax,  ymax, 1]))
        point3 = np.dot(rot_mat, np.array([xmin , ymax, 1]))
        point4 = np.dot(rot_mat, np.array([xmax, ymin , 1]))
        # point1 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymin, 1]))
        # point2 = np.dot(rot_mat, np.array([xmax, (ymin + ymax) / 2, 1]))
        # point3 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymax, 1]))
        # point4 = np.dot(rot_mat, np.array([xmin, (ymin + ymax) / 2, 1]))
        # 合并np.array
        concat = np.vstack((point1, point2, point3, point4))
        # 改变array类型
        concat = concat.astype(np.int32)
        #print concat
        rx, ry, rw, rh = cv2.boundingRect(concat)
        lx = rx + rw
        ly = ry + rh
        resultbox.append(rx)
        resultbox.append(ry)
        resultbox.append(lx)
        resultbox.append(ly)
    return resultbox

def randomGaussianBlur(img):
    """
    对图像进行高斯滤波，使图像变得模糊
    :param image:
    :return:
    """
    kernal_size_list = [5, 7, 9]  # [3, 5, 7, 9, 10]
    kernal_size = random.choice(kernal_size_list)
    sigmaX = random.choice(range(1, 8, 2))  #range(10, 130, 15)
    sigma = sigmaX
    #sigma = sigmaX * 1.0 / 10 #  100
    # print kernal_size
    # print sigma
    #img = np.asarray(img)
    img_gau = cv2.GaussianBlur(img, (kernal_size, kernal_size), sigma)
    # return img
    return img_gau

def randomColor(image):
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """
    c = random.randint(0,3)

    random_factor = np.random.randint(0, 50) / 10.  # 随机因子
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    if c == 0:
        return color_image
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    if c == 1:
        return brightness_image
    #
    random_factor = np.random.randint(10, 42) / 10.  # 随机因子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    if c == 2:
        return contrast_image
    random_factor = np.random.randint(0, 62) / 10.  # 随机因子
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度

# 随机生成5000个椒盐噪声
def zaodian(img):
    height,weight,channel = img.shape
    img_zao = img.copy()
    for i in range(5000):
        x = np.random.randint(0,height)
        y = np.random.randint(0,weight)
        img_zao[x ,y ,:] = 255
    return img_zao

def translate(img,w,h):
    rows,cols,channel = img.shape
    M = np.float32([[1,0,w],[0,1,h]])
    img_ping = cv2.warpAffine(img,M,(cols,rows))
    return img_ping

def trans_rec(w,h,originrec):
    ping_rec = [originrec[0] + w, originrec[1] + h, originrec[2] + w, originrec[3] + h]
    return ping_rec

def rhgan(ganimg,srcimg):
    h,w,_ = ganimg.shape
    maxh = srcimg.shape[0] - h -1
    maxw = srcimg.shape[1] - w -1
    if maxh <0 or maxw < 0:
        return srcimg
    y = random.randint(0,maxh)
    x = random.randint(0,maxw)
    roiimg = srcimg[y:y + h,x:x+w,:]
    mask = np.ones(ganimg.shape, ganimg.dtype) * 255
    cv2.copyTo(ganimg, mask, roiimg)
    return srcimg

def superpixelsaug(img):
    images = np.expand_dims(img, axis=0)
    aug = iaa.Superpixels(p_replace=0.05)
    images_aug = aug(images=images)
    img_aug = np.squeeze(images_aug)
    return img_aug
    
def fogaug(img):
    images = np.expand_dims(img, axis=0)
    aug = iaa.Fog()
    images_aug = aug(images=images)
    img_aug = np.squeeze(images_aug)
    return img_aug

def cloudsaug(img):
    images = np.expand_dims(img, axis=0)
    aug = iaa.Clouds()
    images_aug = aug(images=images)
    img_aug = np.squeeze(images_aug)
    return img_aug

def fnaug(img):
    images = np.expand_dims(img, axis=0)
    aug = iaa.FrequencyNoiseAlpha(first=iaa.EdgeDetect(0.5))
    images_aug = aug(images=images)
    img_aug = np.squeeze(images_aug)
    return img_aug

def Coarseaug(img):
    images = np.expand_dims(img, axis=0)
    aug = iaa.CoarseDropout(0.02, size_percent=0.5)
    images_aug = aug(images=images)
    img_aug = np.squeeze(images_aug)
    return img_aug

def parsexml(annofile):
    tree = ET.parse(annofile)
    root = tree.getroot()
    boxs = []
    for object in root.findall('object'):
        for bndbox in object.findall('bndbox'):
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            boxs.append(xmin)
            boxs.append(ymin)
            boxs.append(xmax)
            boxs.append(ymax)
    return boxs

def generatexml(srcannofile,destannofile,newbox,w,h):
    print ("*******",newbox)
    tree = ET.parse(srcannofile)
    root = tree.getroot()
    newbox = np.array(newbox).reshape(-1,4)
    print (newbox)
    for size in root.findall('size'):
        size.find('width').text = str(w)
        size.find('height').text = str(h)
    for index, object in enumerate(root.findall('object')):
        # if index >= 1:
        #     continue
        for bndbox in object.findall('bndbox'):
            bndbox.find('xmin').text = newbox[index][0]
            print (newbox[index][0])
            bndbox.find('ymin').text = newbox[index][1]
            print (newbox[index][1])

            bndbox.find('xmax').text = newbox[index][2]
            print (newbox[index][2])

            bndbox.find('ymax').text = newbox[index][3]
            print (newbox[index][3])

    tree.write(destannofile)

def copyxml(srcannofile,destannofile):
    os.system("copy %s %s"%(srcannofile,destannofile))
    return 0

if __name__ == '__main__':
    imgfiles = os.listdir(imgpath)
    start = 0
    for index ,name in enumerate(imgfiles):
        if name == "vlcsnap-2019-06-12-10h11m01s081.png":
            start = index
    print (imgfiles)
    annofiles = os.listdir(annopath)
    for img_name in imgfiles[start:]:
        img = cv2.imread(imgpath + "/" + img_name)
        if img is None:
            continue
        xmlname = img_name.rstrip("jpg").rstrip("png") + "xml"
        srcannofile = os.path.join(annopath ,xmlname)
        if not  os.path.exists(srcannofile):
            continue
        rectangle = parsexml(annopath + "/" +xmlname)
        if len(rectangle) == 0 :
            continue
        rand_scale = random.randint(1, 3)
        img_flip = flip_img(img, 1.0 / rand_scale)
        flip_rec = flip_box(img, rectangle, 1.0 / rand_scale)
        flip_rec = [str(i) for i in flip_rec]
        flip_imgname = img_name.rstrip('.jpg').rstrip(".png") + '_flip' + '.jpg'
        flip_xmlname = flip_imgname.rstrip("jpg") + "xml"
        destannofile = destannopath + "/" + flip_xmlname
        cv2.imwrite(destimgpath + "/" + flip_imgname, img_flip)
        generatexml(srcannofile, destannofile, flip_rec,img_flip.shape[1],img_flip.shape[0])
        range_num = random.randint(1, 15)
        # range_num = 100
        for i in range(range_num):
            rand_num = random.randint(0, 15)
            # rand_num = 0
            if rand_num == 0 or rand_num == 13 or rand_num == 14 or rand_num == 15 :
                rand_scale = random.randint(1, 3)
                rand_angle = random.randint(0, 5)
                angles = [5, 10,  355, 350,90,270]
                # angles = [90,270]
                img_rot = rotate_image(img, angles[rand_angle], 1.0 / rand_scale)
                rot_rec = rotate_xml(img, rectangle, angles[rand_angle], 1.0 / rand_scale)
                rot_rec = [str(j) for j in rot_rec]
                rot_imgname = img_name.rstrip('.jpg').rstrip('.png') + '_rot%s' % i + '.jpg'
                rot_xmlname = rot_imgname.rstrip('jpg') + 'xml'
                destannofile = destannopath + "/" + rot_xmlname
                cv2.imwrite(destimgpath + "/" + rot_imgname, img_rot)
                generatexml(srcannofile, destannofile, rot_rec,img_rot.shape[1],img_rot.shape[0])

            elif rand_num == 1 :
                img_gau = randomGaussianBlur(img)
                gau_rec = rectangle
                gau_rec = [str(i) for i in gau_rec]
                gau_imgname = img_name.rstrip('.jpg').rstrip('.png') + '_gau%s' % i + '.jpg'
                gau_xmlname = gau_imgname.rstrip('jpg') + 'xml'
                destannofile = destannopath + "/" + gau_xmlname
                cv2.imwrite(destimgpath + "/" + gau_imgname, img_gau)
                generatexml(srcannofile, destannofile, gau_rec,img_gau.shape[1],img_gau.shape[0])
            elif rand_num == 2:
                img_zao = zaodian(img)
                zao_rec = rectangle
                zao_rec = [str(j) for j in zao_rec]
                zao_imgname = img_name.rstrip('.jpg').rstrip('.png') + '_zao%s' % i + '.jpg'
                zao_xmlname = zao_imgname.rstrip('jpg') + 'xml'
                destannofile = destannopath + "/" + zao_xmlname
                cv2.imwrite(destimgpath + "/" + zao_imgname, img_zao)
                generatexml(srcannofile, destannofile, zao_rec,img_zao.shape[1],img_zao.shape[0])

            elif rand_num == 3:
                ch = [5, 10, 15, 20]
                cw = [5, 10, 15, 20]
                w = random.choice(cw)
                h = random.choice(ch)
                img_ping = translate(img, w, h)
                print (rectangle)
                print (w,h)
                box_num = int(len(rectangle)/4)
                ping_result_rec = []
                for i in range(box_num):
                    ping_rec = [rectangle[0 + 4 *i] + w, rectangle[1 + 4*i ] + h, rectangle[2 + 4*i] + w, rectangle[3 +4*i] + h]
                    ping_result_rec.extend(ping_rec)
                # ping_rec = [rectangle[0] + w, rectangle[1] + h, rectangle[2] + w, rectangle[3] + h]
                bool_continue = 0
                for i in range(box_num):
                    if ping_result_rec[0 + 4*i] > img_ping.shape[0] or ping_result_rec[2 +4*i] >img_ping.shape[0] or ping_result_rec[1 + 4*i] >img_ping.shape[1] or ping_result_rec[3 + 4*i] >img_ping.shape[1]:
                        bool_continue = 1
                if bool_continue == 1:
                    continue
                print (ping_result_rec)
                ping_result_rec = [str(i) for i in ping_result_rec]
                ping_imgname = img_name.rstrip('.jpg').rstrip('.png') + '_ping%s' % i + '.jpg'

                ping_xmlname = ping_imgname.rstrip('jpg') + 'xml'
                destannofile = destannopath + "/" + ping_xmlname
                print (destannofile)
                cv2.imwrite(destimgpath + "/" + ping_imgname, img_ping)
                generatexml(srcannofile, destannofile, ping_result_rec,img_ping.shape[1],img_ping.shape[0])
            elif rand_num == 4:
                img_c = Image.open(imgpath + '/' + img_name, mode="r")
                img_color = randomColor(img_c)
                img_color = cv2.cvtColor(np.asarray(img_color), cv2.COLOR_RGB2BGR)
                color_rec = rectangle
                color_rec = [str(j) for j in color_rec]
                color_imgname = img_name.rstrip('.jpg').rstrip('.png') + '_color%s' % i + '.jpg'
                color_xmlname = color_imgname.rstrip('jpg') + 'xml'
                destannofile = destannopath + "/" + color_xmlname
                cv2.imwrite(destimgpath + "/" + color_imgname, img_color)
                generatexml(srcannofile, destannofile, color_rec,img_color.shape[1],img_color.shape[0])
            elif rand_num == 5:
                img_eq = EqExtension(img)
                eq_rec = rectangle
                eq_rec = [str(j) for j in eq_rec]
                eq_imgname = img_name.rstrip('.jpg').rstrip('.png') + '_eq%s' % i + '.jpg'
                eq_xmlname = eq_imgname.rstrip('jpg') + 'xml'
                destannofile = destannopath + "/" + eq_xmlname
                cv2.imwrite(destimgpath + "/" + eq_imgname, img_eq)
                generatexml(srcannofile, destannofile, eq_rec,img_eq.shape[1],img_eq.shape[0])
            elif rand_num == 6:
                img_hue = HueExtension(img)
                hue_rec = rectangle
                hue_rec = [str(j) for j in hue_rec]
                hue_imgname = img_name.rstrip('.jpg').rstrip('.png') + '_hue%s' % i + '.jpg'
                hue_xmlname = hue_imgname.rstrip('jpg') + 'xml'
                destannofile = destannopath + "/" + hue_xmlname
                cv2.imwrite(destimgpath + "/" + hue_imgname, img_hue)
                generatexml(srcannofile, destannofile, hue_rec,img_hue.shape[1],img_hue.shape[0])
            elif rand_num == 7:
                ganpath = "./gan/"
                ganfiles = os.listdir(ganpath)
                ganlen = len(ganfiles)
                gan_num = random.randint(0,ganlen-1)
                ganimg = cv2.imread(ganpath + ganfiles[gan_num])
                img_gan = rhgan(ganimg, img)
                gan_imgname = img_name.rstrip('.jpg').rstrip('.png') + '_gan%s' % i + '.jpg'
                gan_xmlname = gan_imgname.rstrip('jpg') + 'xml'
                destannofile = os.path.join(destannopath ,  gan_xmlname)

                cv2.imwrite(destimgpath + "/" + gan_imgname, img_gan)
                print(srcannofile,destannofile)
                copyxml(srcannofile,destannofile)
            elif rand_num == 8:
                img_sup = superpixelsaug(img)
                sup_imgname = img_name.rstrip('.jpg').rstrip('.png') + '_sup%s' % i + '.jpg'
                sup_xmlname = sup_imgname.rstrip('jpg') + 'xml'
                destannofile = os.path.join(destannopath, sup_xmlname)

                cv2.imwrite(destimgpath + "/" + sup_imgname, img_sup)
                print(srcannofile, destannofile)
                copyxml(srcannofile, destannofile)

            elif rand_num == 9:
                img_fog = fogaug(img)
                fog_imgname = img_name.rstrip('.jpg').rstrip('.png') + '_fog%s' % i + '.jpg'
                fog_xmlname = fog_imgname.rstrip('jpg') + 'xml'
                destannofile = os.path.join(destannopath, fog_xmlname)

                cv2.imwrite(destimgpath + "/" + fog_imgname, img_fog)
                print(srcannofile, destannofile)
                copyxml(srcannofile, destannofile)

            elif rand_num == 10:
                img_cloud = cloudsaug(img)
                cloud_imgname = img_name.rstrip('.jpg').rstrip('.png') + '_cloud%s' % i + '.jpg'
                cloud_xmlname = cloud_imgname.rstrip('jpg') + 'xml'
                destannofile = os.path.join(destannopath, cloud_xmlname)

                cv2.imwrite(destimgpath + "/" + cloud_imgname, img_cloud)
                print(srcannofile, destannofile)
                copyxml(srcannofile, destannofile)
            elif rand_num == 11:
                img_fn = fnaug(img)
                fn_imgname = img_name.rstrip('.jpg').rstrip('.png') + '_fn%s' % i + '.jpg'
                fn_xmlname = fn_imgname.rstrip('jpg') + 'xml'
                destannofile = os.path.join(destannopath, fn_xmlname)

                cv2.imwrite(destimgpath + "/" + fn_imgname, img_fn)
                print(srcannofile, destannofile)
                copyxml(srcannofile, destannofile)
            elif rand_num == 12:
                img_coarse = Coarseaug(img)
                coarse_imgname = img_name.rstrip('.jpg').rstrip('.png') + '_coarse%s' % i + '.jpg'
                coarse_xmlname = coarse_imgname.rstrip('jpg') + 'xml'
                destannofile = os.path.join(destannopath, coarse_xmlname)

                cv2.imwrite(destimgpath + "/" + coarse_imgname, img_coarse)
                print(srcannofile, destannofile)
                copyxml(srcannofile, destannofile)








