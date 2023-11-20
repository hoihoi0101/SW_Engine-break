import cv2
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO('C:/Capston/SW_Engine-break/mosaic_test_yolov8/models/yolov8n-face.pt')

# 웹캠 설정
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    # 웹캠 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 모델로 얼굴 찾기
    faces = model(frame)

    # 찾은 얼굴 영역 출력
    for face in faces.xyxy[0]:
        x1, y1, x2, y2, conf, cls = face[:6]  # 바운딩 박스 좌표 정보
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # 화면에 얼굴 검출 결과 출력
    cv2.imshow('Face Detection', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료
cap.release()
cv2.destroyAllWindows()
