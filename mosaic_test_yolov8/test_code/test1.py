
######### 사람 얼굴만 판단해서 facenet 사용해 객체 탐지

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

            # Facenet을 사용하여 얼굴의 특성 벡터 (임베딩) 추출
            face = cv2.resize(face, (160, 160))  # Facenet 모델은 160x160 크기의 얼굴 이미지를 필요로 합니다
            face = np.transpose(face, (2, 0, 1))  # 이미지 채널 순서 변경 (C, H, W)
            face = torch.Tensor(face).unsqueeze(0)  # 배치 차원 추가
            embeddings = facenet(face)  # 얼굴의 임베딩 추출

            # 여기에서 추출된 임베딩을 사용하여 얼굴 인식 또는 일치 여부 판단을 수행할 수 있습니다.

            # 얼굴 주위에 사각형 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
