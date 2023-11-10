# 얼굴 마스크 테스트 바운딩 박스 중앙 좌표 계산후 그위치에 이미지 합성하기 (그나마 나은 방법)


from ultralytics import YOLO
import cv2

# model = 개인 경로에 맞게 수정해야함
model = YOLO('C:/Code/PROJECT/SW_Engine-break/mosaic_test_yolov8/models/yolov8n-face.pt')

# 얼굴에 씌울 인형 얼굴 이미지 로드 (인형 얼굴 이미지를 개인 경로에 맞게 수정해야 합니다.)
doll_face_img = cv2.imread('C:/Code/PROJECT/SW_Engine-break/mosaic_test_yolov8/test_data/panda.png', cv2.IMREAD_UNCHANGED)

# Webcam 설정
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, img = cap.read()

    # YOLO 모델로 얼굴 찾기
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

            # 얼굴 중심 좌표 계산
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # 인형 얼굴 이미지 크기 동적 조정 (예시로 2배 크기로 조정)
            doll_face_resized = cv2.resize(doll_face_img, (2 * (x2 - x1), 2 * (y2 - y1)))

            # 이미지 합성
            start_x = max(center_x - (x2 - x1), 0)
            end_x = min(center_x + (x2 - x1), img.shape[1])
            start_y = max(center_y - (y2 - y1)-30, 0)
            end_y = min(center_y + (y2 - y1)-30, img.shape[0])



            # 합성할 이미지 크기를 원본 이미지에 맞게 조정
            doll_face_resized = doll_face_resized[:end_y - start_y, :end_x - start_x, :]

            alpha_s = doll_face_resized[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(0, 3):
                img[start_y:end_y, start_x:end_x, c] = (
                    alpha_s * doll_face_resized[:, :, c] +
                    alpha_l * img[start_y:end_y, start_x:end_x, c]
                )

    cv2.imshow('Webcam', img)

    # Q 누르면 캠 종료
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
