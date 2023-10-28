
######### 사람객체을 아예 다 모자이크 처리 - yolo v5 사용

import cv2
import torch
from PIL import Image

# # YOLOv5 모델을 CPU로 로드
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)


# 모델 파일 경로
model_path = 'C:/Code/mosaic_test_1/best_v5s1.pt'  # 모델 파일 경로에 맞게 수정

# YOLOv5 모델 불러오기
model = torch.hub.load('ultralytics/yolov5:master', 'custom', path=model_path)

# 모델을 CPU로 이식
model = model.to('cpu')


# 웹캠에서 비디오 스트림을 가져옵니다.
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # 이미지 크기를 모델이 예상하는 크기로 조정
    img = Image.fromarray(frame)
    results = model(img)  # YOLOv5를 사용하여 객체 감지 수행

    # 결과에서 감지된 객체 정보 가져오기
    for det in results.pred[0]:
        if det[-1] == 0:  # 클래스 인덱스 0은 '사람'을 나타냅니다.
            x1, y1, x2, y2, conf = map(int, det[:5])

            # 얼굴 영역 추출 (예: dlib 또는 Haar Cascade를 사용하여 얼굴 검출)
            # 여기에서는 감지된 '사람' 영역을 얼굴로 간주합니다.
            face = frame[y1:y2, x1:x2]

            # 얼굴 이미지 모자이크 처리
            face = cv2.GaussianBlur(face, (99, 99), 30)

            # 모자이크 처리된 얼굴을 다시 프레임에 삽입
            frame[y1:y2, x1:x2] = face

    cv2.imshow('Face Mosaic', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
