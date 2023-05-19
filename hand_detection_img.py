import cv2
import mediapipe as mp

model = mp.solutions.hands.Hands(static_image_mode=True, min_detection_confidence=0.1,
	min_tracking_confidence=0.2, max_num_hands=3)

img = cv2.imread("files/HandSignals.png")

width, height = img.shape[1], img.shape[0]

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# results = model.process(img_rgb)
# print(results)
# print(results.multi_hand_landmarks) 
# print(type(results.multi_hand_landmarks)) # <class 'list'>

lms = model.process(img_rgb).multi_hand_landmarks # landmarks

if lms: # if lms was not NAN
	# print(len(lms)) # 2
	# print(type(lms[0])) # first hand # <class 'mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList'>
	# print(lms[0])
	for lm in lms: 
		# print(lm)
		# print(type(lm)) # is not a list # <class 'mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList'>
		# print(lm.landmark) # lm is an object with attribute of landmark having 21 rows and each row has x, y, z
		# print(len(lm.landmark))

		# lm0 = lm.landmark[0].x, lm.landmark[0].y
		# print(lm0)
		# cv2.circle(img, (int(lm0[0]*width), int(lm0[1]*height)), 5, (0, 0, 255), cv2.FILLED) # pixel must be an integer, not a float
		for point in lm.landmark:
			lm0 = point.x, point.y
			cv2.circle(img, (int(lm0[0]*width), int(lm0[1]*height)), 5, (0, 0, 255), cv2.FILLED) 


cv2.imshow("Hand Signals", img)

cv2.waitKey(0)

cv2.destroyWindow("Hand Signals")