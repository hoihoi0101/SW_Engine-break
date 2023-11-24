# 얼굴 마스크 테스트 바운딩 박스 중앙 좌표 계산후 그위치에 이미지 합성하기 (그나마 나은 방법)


from ultralytics import YOLO
import cv2

# model = 개인 경로에 맞게 수정해야함
model = YOLO('C:/Code/PROJECT/SW_Engine-break/mosaic_test_yolov8/models/yolov8n-face.pt')

# 얼굴에 씌울 인형 얼굴 이미지 로드 (인형 얼굴 이미지를 개인 경로에 맞게 수정해야 합니다.)
doll_face_img1 = cv2.imread('C:/Code/PROJECT/SW_Engine-break/mosaic_test_yolov8/test_data/panda.png', cv2.IMREAD_UNCHANGED)
doll_face_img2 = cv2.imread('C:/Code/PROJECT/SW_Engine-break/mosaic_test_yolov8/test_data/clown.png', cv2.IMREAD_UNCHANGED)


doll_face_img2 = cv2.resize(doll_face_img2, (doll_face_img2.shape[1] // 2, doll_face_img2.shape[0] // 2))
# # clown 이미지 크기 조정. 여기서는 0.8 설정
# doll_face_img2 = cv2.resize(doll_face_img2, (0, 0), fx=0.8, fy=0.8)

# 현재 인형 얼굴 이미지
current_doll_face_img = doll_face_img1

# Webcam 설정
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# 인형 얼굴 씌우기 플래그
overlay_flag = False

while True:
    success, img = cap.read()

    # YOLO 모델로 얼굴 찾기
    results = model(img, stream=True)

    # 키 입력 확인
    key = cv2.waitKey(1)
    if key == ord('t'):
        overlay_flag = not overlay_flag  # 플래그 상태 토글
    elif key == ord('c'):
        # 현재 인형 얼굴 이미지 변경
        if current_doll_face_img is doll_face_img1:
            current_doll_face_img = doll_face_img2
        else:
            current_doll_face_img = doll_face_img1
    elif key == ord('q'):
        break  # 'q'를 누르면 종료

    # 인형 얼굴 씌우기
    if overlay_flag:
        for r in results:
            boxes = r.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                doll_face_resized = cv2.resize(current_doll_face_img, (2 * (x2 - x1), 2 * (y2 - y1)))

                start_x = max(center_x - (x2 - x1), 0)
                end_x = min(center_x + (x2 - x1), img.shape[1])
                start_y = max(center_y - (y2 - y1) - 30, 0)
                end_y = min(center_y + (y2 - y1) - 30, img.shape[0])

                doll_face_resized = doll_face_resized[:end_y - start_y, :end_x - start_x, :]

                alpha_s = doll_face_resized[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s

                for c in range(0, 3):
                    img[start_y:end_y, start_x:end_x, c] = (
                            alpha_s * doll_face_resized[:, :, c] +
                            alpha_l * img[start_y:end_y, start_x:end_x, c]
                    )


    cv2.imshow('Webcam', img)



cap.release()
cv2.destroyAllWindows()
