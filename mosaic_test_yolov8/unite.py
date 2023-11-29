from ultralytics import YOLO
import cv2

model = YOLO('C:/Code/PROJECT/SW_Engine-break/mosaic_test_yolov8/models/yolov8n-face.pt')

doll_face_img1 = cv2.imread('C:/Code/PROJECT/SW_Engine-break/mosaic_test_yolov8/test_data/panda.png',
                            cv2.IMREAD_UNCHANGED)
doll_face_img2 = cv2.imread('C:/Code/PROJECT/SW_Engine-break/mosaic_test_yolov8/test_data/clown.png',
                            cv2.IMREAD_UNCHANGED)

nose_img = cv2.imread('C:/Code/PROJECT/SW_Engine-break/mosaic_test_yolov8/test_data/red_nose.png', cv2.IMREAD_UNCHANGED)
headband_img = cv2.imread('C:/Code/PROJECT/SW_Engine-break/mosaic_test_yolov8/test_data/rudolph_horns.png',
                          cv2.IMREAD_UNCHANGED)

nose_cascade = cv2.CascadeClassifier(
    'C:/Code/PROJECT/SW_Engine-break/mosaic_test_yolov8/models/haarcascade_mcs_nose.xml')

doll_face_img2 = cv2.resize(doll_face_img2, (doll_face_img2.shape[1] // 2, doll_face_img2.shape[0] // 2))

overlay_flag = [False, False, False, False]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

ds_factor = 0.5

imgs = [doll_face_img1, doll_face_img2, nose_img, headband_img]

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    key = cv2.waitKey(1)
    if key == ord('1'):
        overlay_flag[0] = not overlay_flag[0]
    elif key == ord('2'):
        overlay_flag[1] = not overlay_flag[1]
    elif key == ord('3'):
        overlay_flag[2] = not overlay_flag[2]
    elif key == ord('4'):
        overlay_flag[3] = not overlay_flag[3]
    elif key == ord('q'):
        break

    for idx, flag in enumerate(overlay_flag):
        if flag and idx != 2:  # '3'을 눌렀을 때는 실행하지 않음
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    # 키 번호에 따라 이미지를 적용할 위치 변경
                    if idx == 0:  # '1'을 눌렀을 때
                        center_x += 50  # 예시: x 좌표를 오른쪽으로 50 이동
                    elif idx == 1:  # '2'를 눌렀을 때
                        center_y += 50  # 예시: y 좌표를 아래로 50 이동
                    elif idx == 3:  # '4'를 눌렀을 때
                        start_y -= 300  # 예시: x 좌표를 왼쪽으로 50 이동

                    img_resized = cv2.resize(imgs[idx], (2 * (x2 - x1), 2 * (y2 - y1)))

                    start_x = max(center_x - (x2 - x1), 0)
                    end_x = min(center_x + (x2 - x1), img.shape[1])
                    start_y = max(center_y - (y2 - y1) - 30, 0)
                    end_y = min(center_y + (y2 - y1) - 30, img.shape[0])

                    img_resized = img_resized[:end_y - start_y, :end_x - start_x, :]

                    alpha_s = img_resized[:, :, 3] / 255.0
                    alpha_l = 1.0 - alpha_s

                    for c in range(0, 3):
                        img[start_y:end_y, start_x:end_x, c] = (
                                alpha_s * img_resized[:, :, c] +
                                alpha_l * img[start_y:end_y, start_x:end_x, c]
                        )

    if overlay_flag[2]:  # '3' 키를 눌렀을 때 Haar cascade 탐지 작동 및 루돌프 코 적용
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        nose_rects = nose_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in nose_rects:
            nose_resized_sticker = cv2.resize(nose_img, (w, h))
            if nose_resized_sticker.shape[2] == 4:
                alpha_sticker = nose_resized_sticker[:, :, 3] / 255.0
                alpha_frame = 1.0 - alpha_sticker
                for c in range(0, 3):
                    img[y:y + h, x:x + w, c] = (alpha_sticker * nose_resized_sticker[:, :, c] +
                                                alpha_frame * img[y:y + h, x:x + w, c])

    cv2.imshow('Webcam', img)

cap.release()
cv2.destroyAllWindows()
