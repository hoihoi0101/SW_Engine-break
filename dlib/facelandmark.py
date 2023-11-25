import dlib
import cv2 as cv
import numpy as np

# red_nose = cv.imread('C:\Users\joohy\SW_engine\SW_Engine-break\haar_test\red_nose.png', cv.IMREAD_UNCHANGED)
# rudolph = cv.imread('C:\Users\joohy\SW_engine\SW_Engine-break\haar_test\rudolph_horns.png', cv.IMREAD_UNCHANGED)
# pada_ears = cv.imread('C:\Users\joohy\SW_engine\SW_Engine-break\haar_test\panda_ear.png', cv.IMREAD_UNCHANGED)
# pada_eyes = cv.imread('C:\Users\joohy\SW_engine\SW_Engine-break\haar_test\panda_eye.png', cv.IMREAD_UNCHANGED)



 
detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


cap = cv.VideoCapture(0)


#range는 끝값이 포함안됨  
ALL = list(range(0, 68))
RIGHT_EYEBROW = list([19]) 
LEFT_EYEBROW = list([24]) 
RIGHT_EYE = list([36,39]) 
LEFT_EYE = list([42,45]) 
NOSE = list([30]) 
MOUTH_OUTLINE = list(range(48, 61)) 
MOUTH_INNER = list(range(61, 68))
JAWLINE = list(range(0, 17))

# ALL = list(range(0, 68))
# RIGHT_EYEBROW = list(range(17, 22)) 
# LEFT_EYEBROW = list(range(22, 27)) 
# RIGHT_EYE = list(range(36, 42)) 
# LEFT_EYE = list(range(42, 48)) 
# NOSE = list(range(27, 36)) 
# MOUTH_OUTLINE = list(range(48, 61)) 
# MOUTH_INNER = list(range(61, 68))
# JAWLINE = list(range(0, 17))

index = ALL

while True:

    ret, img_frame = cap.read()

    img_gray = cv.cvtColor(img_frame, cv.COLOR_BGR2GRAY)


    dets = detector(img_gray, 1)


    for face in dets:

        shape = predictor(img_frame, face) #얼굴에서 68개 점 찾기

        list_points = []
        for p in shape.parts():
            list_points.append([p.x, p.y])

        list_points = np.array(list_points)
        
        red_nose = cv.imread('C:/Users/joohy/SW_engine/SW_Engine-break/haar_test/red_nose.png', cv.IMREAD_UNCHANGED)
        
        # 스티커 크기 조절
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        image_to_display = cv.resize(red_nose, (w, h))


        for i,pt in enumerate(list_points[index]):
            # pt[0] = max(0, pt[0])
            # pt[1] = max(0, pt[1])    
            # x_end = min(pt[0] + image_to_display.shape[1], img_frame.shape[1])
            # y_end = min(pt[1] + image_to_display.shape[0], img_frame.shape[0])
            pt_pos = (pt[0], pt[1]) # 각 포인트의 x, y 좌표 같음 
            # 얼굴에 점 표시
            cv.circle(img_frame, pt_pos, 2, (0, 255, 0), -1)
            # img_frame[pt[1]:y_end, pt[0]:x_end] = image_to_display[:y_end - pt[1], :x_end - pt[0]]
            
            # # 스티커??
            # pt_pos = (pt[0] + x, pt[1] + y)
            # img_frame[y:y+h, x:x+w, 3] = 0  # 스티커 부분 투명하게 만들기
            # img_frame[y:y+h, x:x+w] += sticker_image_resized[:, :, :3]

       
        # 얼굴에 사각형 표시
        cv.rectangle(img_frame, (face.left(), face.top()), (face.right(), face.bottom()),
            (0, 0, 255), 3)


    cv.imshow('result', img_frame)

   
    key = cv.waitKey(1)

    if key == 27:
        break
   
    elif key == ord('1'):
        index = ALL
    elif key == ord('2'):
        index = LEFT_EYEBROW + RIGHT_EYEBROW
    elif key == ord('3'):
        index = LEFT_EYE + RIGHT_EYE
    elif key == ord('4'):
        index = NOSE
    elif key == ord('5'):
        index = MOUTH_OUTLINE+MOUTH_INNER
    elif key == ord('6'):
        index = JAWLINE


cap.release()