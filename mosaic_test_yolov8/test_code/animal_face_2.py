# 얼굴 마스크 테스트 바운딩 박스 중앙 좌표 계산후 그위치에 이미지 합성하기 (그나마 나은 방법)
from ultralytics import YOLO
import numpy as np
import cv2

# model = 개인 경로에 맞게 수정해야함
model = YOLO('C:/Capston/SW_Engine-break/mosaic_test_yolov8/models/yolov8n-face.pt')
nose_cascade = cv2.CascadeClassifier('C:\Capston\SW_Engine-break\mosaic_test_yolov8\models\haarcascade_mcs_nose.xml')
# 스티커 이미지 불러오기
#루돌프 스티커
sticker_nose = cv2.imread('C:/Capston/SW_Engine-break/mosaic_test_yolov8/test_data/red_nose.png', cv2.IMREAD_UNCHANGED)
sticker_face = cv2.imread('C:/Capston/SW_Engine-break/mosaic_test_yolov8/test_data/rudolph_horns.png', cv2.IMREAD_UNCHANGED)
#판다 탈 씌우기
tal_face = cv2.imread('C:/Capston/SW_Engine-break/mosaic_test_yolov8/test_data/clown.png', cv2.IMREAD_UNCHANGED)
tal_nose = None
# # clown 이미지 크기 조정. 여기서는 0.8 설정
# doll_face_img2 = cv2.resize(doll_face_img2, (0, 0), fx=0.8, fy=0.8)

# 현재 인형 얼굴 이미지
current_nose_img = sticker_nose
current_face_img = sticker_face

#크기 위치를 설정하기 위한 값
C_count = 0

# Webcam 설정
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

ds_factor = 0.5

# 인형 얼굴 씌우기 플래그
overlay_flag = False

while True:
    success, frame = cap.read()
    frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)

    # YOLO 모델로 얼굴 찾기
    faceRects = model(frame, stream=True)

    #Haar_cascade로 코찾기
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    nose_rects = nose_cascade.detectMultiScale(gray, 1.3, 5)

    # 키 입력 확인
    key = cv2.waitKey(1)
    if key == ord('t'):
        overlay_flag = not overlay_flag  # 플래그 상태 토글
    elif key == ord('c'):
        # 현재 인형 얼굴 이미지 변경
        if np.array_equal(current_face_img, sticker_face) and np.array_equal(current_nose_img, sticker_nose) :
            current_face_img = tal_face
            current_nose_img = tal_nose
            C_count = 1
        elif np.array_equal(current_face_img, tal_face) and np.array_equal(current_nose_img, tal_nose):
            current_face_img = sticker_face
            current_nose_img = sticker_nose
            C_count = 0

    elif key == ord('q'):
        break  # 'q'를 누르면 종료

    # 인형 얼굴 씌우기
    if overlay_flag:

        #Haar_Cascade번 코 스티커
        if C_count != 1:
            for (x, y, w, h) in nose_rects:
                # 좌표 조정하여 스티커 위치 조정 가능
                nose_resized_sticker = cv2.resize(current_nose_img, (w, h))

                # 알파 채널 있는 경우만 처리
                if nose_resized_sticker.shape[2] == 4:
                    alpha_sticker = nose_resized_sticker[:, :, 3] / 255.0
                    alpha_frame = 1.0 - alpha_sticker

                    for c in range(0, 3):
                        frame[y:y + h, x:x + w, c] = (alpha_sticker * nose_resized_sticker[:, :, c] +
                                                      alpha_frame * frame[y:y + h, x:x + w, c])
                else:
                    # 알파 채널이 없는 경우 처리
                    frame[y:y + h, x:x + w] = nose_resized_sticker[:, :, :3]

                break

        for face in faceRects:
            boxes = face.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w = abs(x2 - x1)
                h = abs(y2 - y1)


                #스티커 및 탈 별로 크기와 위치를 조정
                #루돌프 스티커의 크기 및 위치
                if C_count == 0 :
                    current_face_width = int(1.9 * w)
                    current_face_height = int(1.5 * h)
                    current_face_x = int((current_face_width - w) / 2)
                    current_face_y = int((current_face_height - h) * 2 + 10)
                #판다 탈의 크기 및 위치
                elif C_count == 1:
                    current_face_width = int(1.0 * w)
                    current_face_height = int(1.0 * h)
                    current_face_x = int((current_face_width - w) / 2)
                    current_face_y = int((current_face_height - h) / 2 + 10)
                    # current_face_width = int(1.9 * w)
                    # current_face_height = int(1.5 * h)
                    # current_face_x = int((current_face_width - w) / 2)
                    # current_face_y = int((current_face_height - h) / 2 + 35)



                # 머리 부분 스티커
                # 스티커 크기 조정
                sticker_width = current_face_width
                sticker_height = current_face_height
                sticker_x = x1 - current_face_x
                sticker_y = y1 - current_face_y

                # 스티커가 프레임을 벗어나지 않도록 좌표 보정
                sticker_x = max(sticker_x, 0)
                sticker_y = max(sticker_y, 0)

                # 좌표 보정 후 스티커를 프레임에 적용
                sticker_area = frame[sticker_y:sticker_y + sticker_height, sticker_x:sticker_x + sticker_width]
                resized_sticker = cv2.resize(current_face_img, (sticker_area.shape[1], sticker_area.shape[0]))

                if resized_sticker.shape[2] == 4:
                    alpha_sticker = resized_sticker[:, :, 3] / 255.0
                    alpha_frame = 1.0 - alpha_sticker

                    for c in range(0, 3):
                        sticker_area[:, :, c] = (alpha_sticker * resized_sticker[:, :, c] +
                                                 alpha_frame * sticker_area[:, :, c])
                else:
                    sticker_area[:, :] = resized_sticker[:, :, :3]

                frame[sticker_y:sticker_y + sticker_height, sticker_x:sticker_x + sticker_width] = sticker_area

                # #YOLO버전 코 부분 스티커
                #
                # # 스티커 크기 조정
                # nose_sticker_width = int(0.3 * w)
                # nose_sticker_height = int(0.3 * h)
                # nose_sticker_x =x1 - int((nose_sticker_width - w) / 2)
                # nose_sticker_y =y1 - int((nose_sticker_height - h) / 2)
                #
                # # 스티커가 프레임을 벗어나지 않도록 좌표 보정
                # nose_sticker_x = max(nose_sticker_x, 0)
                # nose_sticker_y = max(nose_sticker_y, 0)
                #
                # # 좌표 보정 후 스티커를 프레임에 적용
                # nose_sticker_area = frame[nose_sticker_y:nose_sticker_y + nose_sticker_height, nose_sticker_x:nose_sticker_x + nose_sticker_width]
                # nose_resized_sticker = cv2.resize(current_nose_img, (nose_sticker_area.shape[1], nose_sticker_area.shape[0]))
                #
                # # 알파 채널 있는 경우만 처리
                # if nose_resized_sticker.shape[2] == 4:
                #     alpha_sticker = nose_resized_sticker[:, :, 3] / 255.0
                #     alpha_frame = 1.0 - alpha_sticker
                #
                #     for c in range(0, 3):
                #         nose_sticker_area[:, :, c] = (alpha_sticker * nose_resized_sticker[:, :, c] +
                #                                  alpha_frame * nose_sticker_area[:, :, c])
                # else:
                #     nose_sticker_area[:, :] = nose_resized_sticker[:, :, :3]
                #
                # frame[nose_sticker_y:nose_sticker_y + nose_sticker_height, nose_sticker_x:nose_sticker_x + nose_sticker_width] = nose_sticker_area


    cv2.imshow('Webcam', frame)



cap.release()
cv2.destroyAllWindows()