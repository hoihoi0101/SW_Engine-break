######### test1 에서 나온 결과에서 모자이크 처리
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1

# MTCNN (Multi-task Cascaded Convolutional Networks)을 사용하여 얼굴 검출
mtcnn = MTCNN()

# Facenet 모델 로드
facenet = InceptionResnetV1(pretrained='vggface2').eval()

# 웹캠에서 비디오 스트림을 가져옵니다.
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # MTCNN을 사용하여 얼굴 감지
    boxes, _ = mtcnn.detect(frame)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = frame[y1:y2, x1:x2]

            # 얼굴을 모자이크 처리
            face = cv2.GaussianBlur(face, (99, 99), 30)

            # 모자이크 처리된 얼굴을 다시 프레임에 삽입
            frame[y1:y2, x1:x2] = face

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
