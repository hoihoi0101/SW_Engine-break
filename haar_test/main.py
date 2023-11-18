import cv2

nose_cascade = cv2.CascadeClassifier('./haarcascade_mcs_nose.xml')
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')

# 스티커 이미지 불러오기
sticker_nose = cv2.imread('red_nose.png', cv2.IMREAD_UNCHANGED)
sticker_face = cv2.imread('rudolph_horns.png', cv2.IMREAD_UNCHANGED)

cap = cv2.VideoCapture(0)
# 웹캠 속성 설정 - 프레임 폭과 높이
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

ds_factor = 0.5

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #코 스티커 효과
    nose_rects = nose_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in nose_rects:
        # 좌표 조정하여 스티커 위치 조정 가능
        resized_sticker = cv2.resize(sticker_nose, (w, h))

        # 알파 채널 있는 경우만 처리
        if resized_sticker.shape[2] == 4:
            alpha_sticker = resized_sticker[:, :, 3] / 255.0
            alpha_frame = 1.0 - alpha_sticker

            for c in range(0, 3):
                frame[y:y + h, x:x + w, c] = (alpha_sticker * resized_sticker[:, :, c] +
                                              alpha_frame * frame[y:y + h, x:x + w, c])
        else:
            # 알파 채널이 없는 경우 처리
            frame[y:y + h, x:x + w] = resized_sticker[:, :, :3]

        break

    faceRects = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    #얼굴 스티커효과
    # for (x, y, w, h) in faceRects:
    #     # 얼굴 영역을 기준으로 스티커 크기 조정
    #     sticker_width = int(1.9 * w)
    #     sticker_height = int(1.5 * h)
    #     sticker_x = x - int((sticker_width - w) / 2)
    #     sticker_y = y - int((sticker_height - h) * 2 + 10)
    #
    #     # 스티커가 프레임을 벗어나지 않도록 좌표 보정
    #     sticker_x = max(sticker_x, 0)
    #     sticker_y = max(sticker_y, 0)
    #
    #     # 좌표 보정 후 스티커를 프레임에 적용
    #     sticker_area = frame[sticker_y:sticker_y + sticker_height, sticker_x:sticker_x + sticker_width]
    #     resized_sticker = cv2.resize(sticker_face, (sticker_area.shape[1], sticker_area.shape[0]))
    #
    #     # 알파 채널 있는 경우만 처리
    #     if resized_sticker.shape[2] == 4:
    #         alpha_sticker = resized_sticker[:, :, 3] / 255.0
    #         alpha_frame = 1.0 - alpha_sticker
    #
    #         for c in range(0, 3):
    #             sticker_area[:, :, c] = (alpha_sticker * resized_sticker[:, :, c] +
    #                                      alpha_frame * sticker_area[:, :, c])
    #     else:
    #         # 알파 채널이 없는 경우 처리
    #         sticker_area[:, :] = resized_sticker[:, :, :3]
    #
    #     # 프레임에 스티커 적용
    #     frame[sticker_y:sticker_y + sticker_height, sticker_x:sticker_x + sticker_width] = sticker_area
    #     break

    # 얼굴 영역에 모자이크 적용
    # for (fX, fY, fW, fH) in faceRects:
    #     roi = frame[fY:fY + fH, fX:fX + fW]
    #     roi = cv2.GaussianBlur(roi, (23, 23), 30)  # 모자이크 강도 조절
    #     frame[fY:fY + roi.shape[0], fX:fX + roi.shape[1]] = roi
    #     break

    # 얼굴 확대
    for (fX, fY, fW, fH) in faceRects:
        new_fX = max(0, fX - int(0.2 * fW))  # 왼쪽으로 확대할 만큼의 여유 공간
        new_fY = max(0, fY - int(0.2 * fH))  # 위로 확대할 만큼의 여유 공간
        new_fW = min(frame.shape[1] - new_fX, int(1.4 * fW))  # 오른쪽으로 확대할 만큼의 여유 공간
        new_fH = min(frame.shape[0] - new_fY, int(1.4 * fH))  # 아래로 확대할 만큼의 여유 공간
        # 얼굴 영역을 확대
        enlarged_face = cv2.resize(frame[new_fY:new_fY + new_fH, new_fX:new_fX + new_fW], (fW, fH), interpolation=cv2.INTER_LINEAR)
        frame[fY:fY + fH, fX:fX + fW] = enlarged_face
        break

    cv2.imshow('Face Effects', frame)

    #esc 누를시 화면 꺼짐
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()