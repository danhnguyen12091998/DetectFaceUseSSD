"""trt_ssd.py

This script demonstrates how to do real-time object detection with
TensorRT optimized Single-Shot Multibox Detector (SSD) engine.
"""


import sys
import time
import argparse

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.ssd_classes import get_cls_dict
from utils.ssd import TrtSSD
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization

###############################################################################
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import pandas as pd
import imutils
import dlib
import os.path
from os import path
import pandas as pd
import argparse
import imutils
import time
import dlib
import cv2
import sys
import os.path
from os import path
import numpy
from pandas import DataFrame
from matplotlib import pyplot
from pandas import read_csv
#from pandas import to_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
import copy
###############################################################################

#####################################################################
# Function to compute EAR
#####################################################################
def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear

def lip_distance(shape):
	top_lip = shape[50:53]
	top_lip = np.concatenate((top_lip, shape[61:64]))

	low_lip = shape[56:59]
	low_lip = np.concatenate((low_lip, shape[65:68]))

	top_mean = np.mean(top_lip, axis=0)
	low_mean = np.mean(low_lip, axis=0)

	distance = abs(top_mean[1] - low_mean[1])
	return distance

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.15
EYE_AR_CONSEC_FRAMES = 5
YAWN_THRESH = 30


# initialize the frame counters and the total number of blinks
COUNTER = 0
ALARM_ON = False

#####################################################################
# prepare SVM module.
#####################################################################

dataset=pd.read_csv("balanced_preproc/balanced_preproc_all.csv", index_col="frame")

# Split-out validation dataset (get 20% dataset to validate)
array = dataset.values
X = array[:,:dataset.shape[1]-1].astype(float)
Y = array[:,dataset.shape[1]-1]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,test_size=validation_size, random_state=seed)

# Test options and evaluation metric
num_folds = 10
seed = 7

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold

# prepare the model
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)

print("\n")

model = SVC(C=1.7)  #choose our best model and C
model.fit(rescaledX, Y_train)

# estimate accuracy on validation dataset
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)

print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

print(roc_auc_score(Y_validation,predictions))

#####################################################################
# function to convert 
#####################################################################

def prev_to_csv(X,scaler=scaler,model=model):
	# print(X)
	rescaledX = scaler.transform(X)
	predictions = model.predict(rescaledX)
	newdata = DataFrame(predictions, index=X.index, columns=["blink"])
	return newdata

#moving avareage function
def moving_av(mylist, N):
	cumsum, moving_aves = [0], []
	for i, x in enumerate(mylist, 1):
		cumsum.append(cumsum[i-1] + x)
		if i>=N:
			moving_ave = (cumsum[i] - cumsum[i-N])/N
			#can do stuff with moving_ave here
			moving_aves.append(moving_ave)
	return moving_aves

def get_value_blink_persecond(ear_list):
 
    mov_ear_3=moving_av(ear_list,3)
    mov_ear_5=moving_av(ear_list,5)
    mov_ear_7=moving_av(ear_list,7)

    ear_list = pd.Series(ear_list, index=range(0, len(ear_list)))

    mov_ear_3=pd.Series(mov_ear_3, index=range(2, len(mov_ear_3)+2))
    mov_ear_5=pd.Series(mov_ear_5, index=range(3, len(mov_ear_5)+3))
    mov_ear_7=pd.Series(mov_ear_7, index=range(4, len(mov_ear_7)+4))

    ear_list = pd.DataFrame(ear_list)

    ear_list["mov_ear_3"] = mov_ear_3
    ear_list["mov_ear_5"] = mov_ear_5
    ear_list["mov_ear_7"] = mov_ear_7
    ear_list.columns = ["ear", "mov_ear_3","mov_ear_5","mov_ear_7"]

    ear_list.index.name="frame"
    ##########################################################
    dati=ear_list
    #######################################################
    listear=list(dati.ear)

    #normalizzo
    listear=np.array(listear)
    listear=(listear-np.nanmin(listear))/(np.nanmax(listear)-np.nanmin(listear))
    listear=list(listear)


    col=['F1',"F2","F3","F4","F5",'F6',"F7"]
    df_fin=pd.DataFrame(columns=col)

    #creo righe da 9 frame	
    for i in range(3, len(listear)-4):
        tmp_ear=listear[i-3:i+4]
        df_fin.loc[i]=tmp_ear
        
    df_fin.index.name="frame"
    df_fin.dropna(how='any', inplace=True)

    # using SVM
    previsioni=prev_to_csv(df_fin)

    #####################################################################
    DATA = previsioni
    #####################################################################
    BLINK_LIST = list(DATA.blink)

    for n in range(len(BLINK_LIST)):
        if BLINK_LIST[n]==1.0:
            i = copy.deepcopy(n)
            if sum(BLINK_LIST[i:i+6])<3.0:
                BLINK_LIST[i]=0.0
            else:
                while (sum(BLINK_LIST[i:i+6])>=3.0):
                    BLINK_LIST[i+1]=1.0
                    BLINK_LIST[i+2]=1.0
                    i+=1

    #build singles 1.0 corresponding to blink
    for n in range(len(BLINK_LIST)):
        #trovo il primo 1.0
        if BLINK_LIST[n]==1.0:
            i = copy.deepcopy(n)
            while (BLINK_LIST[i+1]==1.0):
                BLINK_LIST[i+1]=0.0
                i+=1

    #scale 1.0 by 5 frames to position it at about close
    BLINK_LIST=[0.0,0.0,0.0,0.0,0.0]+BLINK_LIST[:len(BLINK_LIST)-5]

    # Number of blinks use SVM
    number_of_blinks = 0

    for n in range(len(BLINK_LIST)):
        if BLINK_LIST[n]==1.0:
            number_of_blinks = number_of_blinks +1
            # print("\nnumber_of_blinks")
            # print(number_of_blinks)
    return number_of_blinks

def get_the_status(blinks_per_minutes_list, yawn_per_minutes):
	delta = blinks_per_minutes_list[len(blinks_per_minutes_list)-1] - blinks_per_minutes_list[len(blinks_per_minutes_list)-2]
	print("delta", delta)
	if delta < 5:
		if yawn_per_minutes < 1:
			print("normal")
		else:
			print("alert")
	if delta > 5 and delta < 15:
		if yawn_per_minutes < 3:
			print("alert")
		else:
			print("drowsiness")
	if delta > 15:
		print("drowsiness")

def get_rect(boxes, confs, clss):
    for bb, cf, cl in zip(boxes, confs, clss):
        rect = dlib.rectangle(bb[0], bb[1], bb[2], bb[3])
    return rect

#facial detector
predictor = dlib.shape_predictor('/home/jetson/Documents/git-repo/KhoaLuanTotNghiep/data/shape_predictor_68_face_landmarks.dat')

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

FRAME=0
number_of_frame=0
ear_list=list()
array_blink_threshold=list()
value_tmp = 0
result_blink_SVM = []

###############################################################################

WINDOW_NAME = 'TrtSsdDemo'
INPUT_HW = (300, 300)
SUPPORTED_MODELS = [
    'ssd_mobilenet_v1_coco',
    'ssd_mobilenet_v1_egohands',
    'ssd_mobilenet_v2_coco',
    'ssd_mobilenet_v2_egohands',
    'ssd_mobilenet_v1_face'
]


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'SSD model on Jetson Nano')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument('--model', type=str, default='ssd_mobilenet_v2_coco',
                        choices=SUPPORTED_MODELS)
    args = parser.parse_args()
    return args


def loop_and_detect(cam, trt_ssd, conf_th, vis):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_ssd: the TRT SSD object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    full_scrn = False
    fps = 0.0
    IS_YAWN = False
    yawnPerMinute = 0
    number_of_frame = 0
    ear_list = []
    tic = time.time()
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is not None:
            boxes, confs, clss = trt_ssd.detect(img, conf_th)
            # get EAR value
            if len(boxes) > 0:
                rect = get_rect(boxes, confs, clss)
                shape = predictor(img, rect)
                shape = face_utils.shape_to_np(shape)

                # extract the left and right eye coordinates, then use the
                # coordinates to compute the eye aspect ratio for both eyes
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                mount = shape[mStart:mEnd]

                # average the eye aspect ratio together for both eyes
                ear = (leftEAR + rightEAR) / 2.0
                distance = lip_distance(shape)
                # print("distance ", distance)

                ear_list.append(ear)
                # compute the convex hull for the left and right eye, then
                # visualize each of the eyes
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                mountHull = cv2.convexHull(mount)
                cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(img, [mountHull], -1, (0, 255, 0), 1)
                if ear < EYE_AR_THRESH:
                    COUNTER += 1
                    # if the eyes were closed for a sufficient number of
                    # then sound the alarm
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        # if the alarm is not on, turn it on
                        if not ALARM_ON:
                            ALARM_ON = True
                            print("DROWSINESS ALERT!")
                # otherwise, the eye aspect ratio is not below the blink
                # threshold
                else:
                    COUNTER = 0
                    ALARM_ON = False
                
                if distance > YAWN_THRESH:
                    if IS_YAWN == False:
                        IS_YAWN = True
                else:
                    if IS_YAWN == True:
                        yawnPerMinute = yawnPerMinute + 1
                        print("YAWNNNNNNNN !")
                        IS_YAWN = False

            number_of_frame +=1    
            
            # Get number of blinks per second
            if number_of_frame == 1800:
                value_blink_persecond = get_value_blink_persecond(ear_list)
                result_blink_SVM.append(value_blink_persecond)
                print("Blinks per min: ", value_blink_persecond)
                print("Yawns per min:", yawnPerMinute)
                print("blinks list:", result_blink_SVM)
                if len(result_blink_SVM) < 2:
                    print("Started")
                else:
                    get_the_status(result_blink_SVM, yawnPerMinute)
                    number_of_frame = 0
                    ear_list = []
                    yawnPerMinute = 0    

            img = vis.draw_bboxes(img, boxes, confs, clss)
            img = show_fps(img, fps)
            cv2.imshow(WINDOW_NAME, img)
            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            # calculate an exponentially decaying average of fps number
            fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
            tic = toc
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)


def main():
    args = parse_args()
    cam = Camera(args)
    cam.open()
    if not cam.is_opened:
        sys.exit('Failed to open camera!')

    cls_dict = get_cls_dict(args.model.split('_')[-1])
    trt_ssd = TrtSSD(args.model, INPUT_HW)

    cam.start()
    open_window(WINDOW_NAME, args.image_width, args.image_height,
                'Camera TensorRT SSD Demo for Jetson Nano')
    vis = BBoxVisualization(cls_dict)
    loop_and_detect(cam, trt_ssd, conf_th=0.3, vis=vis)

    cam.stop()
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
