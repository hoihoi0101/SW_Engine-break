# 얼굴 마스크 테스트 바운딩 박스안에 이미지 넣기 - 짜치는 방법


from ultralytics import YOLO
import cv2
import math



# model = 개인 경로에 맞게 수정해야함
# model = YOLO('C:/Code/PROJECT/SW_Engine-break/mosaic_test_yolov8/models/best1027_3.pt')
# model = YOLO('C:/Code/PROJECT/SW_Engine-break/mosaic_test_yolov8/models/best1027_4.pt')
# model = YOLO('C:/Code/PROJECT/SW_Engine-break/mosaic_test_yolov8/models/yolov8n.pt')
model = YOLO('C:/Code/PROJECT/SW_Engine-break/mosaic_test_yolov8/models/yolov8n-face.pt')



# 얼굴에 씌울 인형 얼굴 이미지 로드 (인형 얼굴 이미지를 개인 경로에 맞게 수정해야 합니다.)
doll_face_img = cv2.imread('C:/Code/PROJECT/SW_Engine-break/mosaic_test_yolov8/test_data/panda.png', cv2.IMREAD_UNCHANGED)



#start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, img = cap.read()

    results = model(img, stream=True)


    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # 인형 얼굴 이미지 크기 조정
            doll_face_resized = cv2.resize(doll_face_img, (x2 - x1, y2 - y1))

            # 알파 채널을 이용하여 합성
            alpha_s = doll_face_resized[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(0, 3):
                img[y1:y2, x1:x2, c] = (alpha_s * doll_face_resized[:, :, c] +
                                        alpha_l * img[y1:y2, x1:x2, c])

    cv2.imshow('Webcam', img)

    ## Q 누르면 캠 종료
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


