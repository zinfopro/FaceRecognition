# -*- coding: utf-8 -*-
"""
@author: ZahraَAkbari
Final Project of Bachelor Degree
Technical and Vocational University
Dr.Shariati Technical College
"""
import tensorflow as tf


tf.compat.v1.disable_v2_behavior() # سوییچ از نسخه ی 2 به نسخه ی 1 تسنورفلو
import numpy as np
import cv as cv2
from detection.mtcnn import detect_face  # کدهای ریپوی دیوید سندبرگ
from keras.preprocessing import image
from scipy import misc
import FaceToolKit as ftk
import os
import datetime

tf.reset_default_graph()
v = ftk.Verification();
v.load_model("./models/20180204-160909/")
v.initial_input_output_tensors()
image_size = 160
verification_threshhold = 1.188
default_color = (0, 255, 0)  # BGR
default_thickness = 2
file_object = open('log.txt', 'a')
base_dir = os.path.expanduser("./Cross_logs")
c = 0
counter = 0
flag = False
os.makedirs(base_dir, exist_ok=True)
now = datetime.datetime.now()
path = os.path.join(base_dir, str(now.year) + str(now.month) + str(now.day))
os.makedirs(path, exist_ok=True)
with tf.Graph().as_default():
    sess = tf.Session()
    pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    minsize = 20
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709


def image_to_encoding(img):
    img = image.load_img(img, target_size=(160, 160))
    y = image.img_to_array(img)
    return v.img_to_encoding(y, image_size)


database = {}
valid_images = [".jpg", ".png"]
valid_path = "./images"
for f in os.listdir(valid_path):
    ext = os.path.splitext(f)[1]
    name = os.path.splitext(f)[0]
    if ext.lower() not in valid_images:
        continue
    database[name] = image_to_encoding(os.path.join(valid_path, f))


def distance(emb1, emb2):
    diff = np.subtract(emb1, emb2)
    return np.sum(np.square(diff))


def who_is_it(image_path, database):
    encoding = image_to_encoding(image_path)
    min_dist = 2000
    for (name, db_enc) in database.items():
        dist = distance(encoding, db_enc)
        if min_dist > dist:
            min_dist = dist
            identity = name
    if min_dist > verification_threshhold:
        print("Not in the database.")
    else:
        print("it's " + str(identity) + ", the distance is " + str(min_dist))
        now = datetime.datetime.now()
        file_object.write("[log] " + str(identity) + " " + str(now) + "\r\n")

    return min_dist, identity


cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    bounding_boxes, points = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
    counter += 1
    for bounding_box in bounding_boxes:
        pts = bounding_box[:4].astype(np.int32)
        pt1 = (pts[0], pts[1])
        pt2 = (pts[2], pts[3])
        cv2.rectangle(frame, pt1, pt2, color=default_color, thickness=default_thickness)
        if counter >= 50:
            cropped = frame[pts[1]:pts[3], pts[0]:pts[2], :]
            scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            now = datetime.datetime.now()
            image_path = os.path.join(path, str(c) + "-" + str(now.strftime("%Y-%m-%d-%H-%M")) + ".png")
            cv2.imwrite(image_path, scaled)
            r = who_is_it(image_path, database)
            flag = True
            c += 1
            cv2.putText(frame, str(r[1]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
    if flag == True:
        counter = 0
        flag = False
    cv2.imshow('FaceRecognition', frame)
    key = cv2.waitKey(1)
    if key == 13:
        break
file_object.close()
cap.release()
cv2.destroyAllWindows()
