from imutils import face_utils
import numpy as np
from scipy.spatial import distance as dist
import argparse
import imutils
import dlib
import cv2
from pymouse import PyMouse
m = PyMouse()
tmp1 = []
tmp2 = []
ANCHOR_POINT = (0, 0)

(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
EYE_AR_THRESH = 0.28
EYE_AR_CONSEC_FRAMES = 5

COUNTER = 0
TOTAL = 0
ANCHOR = 0
before_eye = (0,0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)


def eye_distance(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	eye_dist = (A + B) / (2.0 * C)

	return eye_dist

def direction(r_eye_point, before_point,h, multiple = 1):
	nx, ny = r_eye_point
	x, y = before_point

	# if ny > y + multiple * h:
	if ny > y +multiple*h :
		return 'down'
	# elif ny < y - multiple * h:
	elif ny < y - multiple*h:
		return 'up'
	#elif nx > x:
	#	return 'right'
	#elif nx < x :
	#	return 'left'


	return '-'

def extract_eye(image, left, bottom_left, bottom_right, right, upper_right, upper_left):
	lower_bound = max([left[1], right[1], bottom_left[1], bottom_right[1], upper_left[1], upper_right[1]])
	upper_bound = min([left[1], right[1], upper_left[1], upper_right[1], bottom_left[1], bottom_right[1]])

	eye = image[upper_bound-3:lower_bound+3, left[0]-3:right[0]+3]
	image[upper_bound-3:lower_bound+3, left[0]-3:right[0]+3] = eye

	return eye
def get_mouse_x():
	mouse_position = list(m.position())
	mouse_x = mouse_position[0]
	return mouse_x

def get_mouse_y():
	mouse_position = list(m.position())
	mouse_y = mouse_position[1]
	return mouse_y

def mouse_hor(shape, ANCHOR, image,nStart,nEnd):

	global dir
	global ANCHOR_POINT
	global r_eye_point
	global before_eye
	r_eye = shape[nStart:nEnd]
	r_eye_point = (r_eye[3, 0], r_eye[3, 1])








	ANCHOR_POINT = r_eye_point
	ANCHOR_HEIGHT = 3
	cv2.line(image, ANCHOR_POINT, r_eye_point, (0,0,255), 2)

	# dir = direction(r_eye_point, ANCHOR_POINT, ANCHOR_HEIGHT)
	dir = direction(r_eye_point,before_eye,ANCHOR_HEIGHT)




	while ANCHOR < 3:
		ANCHOR += 1
		ANCHOR_POINT = r_eye_point








	#before_eye = r_eye_point




	print("dir: ",dir)
	if dir == 'up':
		#m.move(get_mouse_x(),get_mouse_y()-30)
		m.move(get_mouse_x(), 100)
		tmp1.append([get_mouse_x(),get_mouse_y()])
	elif dir == 'down':
		#m.move(get_mouse_x(),get_mouse_y()+30)
		m.move(get_mouse_x(), 500)
		tmp2.append([get_mouse_x(),get_mouse_y()])
	elif dir == 'left':

		#m.move(get_mouse_x()+30,get_mouse_y())
		m.move(260, get_mouse_y())
	elif dir == 'right':
		m.move(790, get_mouse_y())
		#m.move(get_mouse_x()-30,get_mouse_y())
	elif dir == '-':
		#m.move(get_mouse_x(),get_mouse_y())
		m.move(540,300)

	return ANCHOR

def mouse_click(shape,COUNTER,TOTAL,EYE_AR_THRESH,EYE_AR_CONSEC_FRAMES,right_eye,nStart,nEnd,lStart, lEnd):
	leftEye = shape[lStart:lEnd]
	rightEye = shape[nStart:nEnd]
	leftEAR = eye_distance(leftEye)
	rightEAR = eye_distance(rightEye)

	ear = (leftEAR + rightEAR) / 2.0

	if ear < EYE_AR_THRESH:
		COUNTER += 1

	else:
		if COUNTER >= EYE_AR_CONSEC_FRAMES:
			TOTAL += 1
			m.click(get_mouse_x(),get_mouse_y(),1)

			cv2.putText(right_eye, "Click!", (10, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		COUNTER = 0

	return COUNTER



def find_pupil(right_eye):
	global gray_right_eye
	global threshold
	gray_right_eye = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
	gray_right_eye = cv2.GaussianBlur(gray_right_eye, (11, 11), 0)
	_, threshold = cv2.threshold(gray_right_eye, 65, 255, cv2.THRESH_BINARY_INV)
	contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
	return contours


def mouse_ver(contours , right_eye):
	for cnt in contours:
		(x, y, w, h) = cv2.boundingRect(cnt)
		cv2.rectangle(right_eye, (x, y), (x + w, y + h), (255, 0, 0), 2)
		cv2.line(right_eye, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
		cv2.line(right_eye, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)

		arr.append((int(x)+int(w/2)))
		if len(arr) == 20:
			a= np.array(arr)
			xx = list(m.position())
			xxx = xx[0]
			yyy = xx[1]
			#print(xx)
			avg = np.sort(a)

			if(avg[5] > 39 ):
				m.move(xxx,yyy)
			elif(avg[5] <= 35):
				m.move(xxx,yyy)
			else :
				m.move(xxx,yyy)
			arr.clear()

		break


def image_show(right_eye, image):
	cv2.imshow("Threshold", threshold)
	# cv2.imshow("gray right_eye", gray_right_eye)
	cv2.imshow("right_eye", right_eye)
	cv2.imshow("image",image)

arr = []
while(True):
	ret, image = cap.read()
	image = image[150:480, 200:500]
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	print(m.position())
	print("tmp1:  ",tmp1)
	print("tmp2:  ",tmp2)
	rects = detector(gray, 1)
	ANCHOR_check = 0

	for (i, rect) in enumerate(rects):
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		right_eye = imutils.resize(extract_eye(image, shape[36], shape[41], shape[46], shape[45], shape[44], shape[37]), width=200, height=100)

		ANCHOR = mouse_hor(shape, ANCHOR, image,nStart,nEnd)
		print(ANCHOR)
		COUNTER = mouse_click(shape, COUNTER, TOTAL, EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES, right_eye, nStart, nEnd,
							  lStart, lEnd)

		print(COUNTER)

		rows, cols, _ = right_eye.shape
		right_eye = right_eye[0:1000,0:60]

		contours = find_pupil(right_eye)
		mouse_ver(contours, right_eye)

		ANCHOR_check = 1
		image_show(right_eye, image)

	if ANCHOR_check == 0:
		ANCHOR =0

	key = cv2.waitKey(1)
	if key == 27:
		break

cap.release()
cv2.destroyAllWindows()
