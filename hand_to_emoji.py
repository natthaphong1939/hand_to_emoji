import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

emoji_dict = {
    "👍": cv2.imread("thumbs_up.png", cv2.IMREAD_UNCHANGED),
    "✌️": cv2.imread("peace_sign.png", cv2.IMREAD_UNCHANGED),
    "🤚": cv2.imread("open_hand.png", cv2.IMREAD_UNCHANGED),
}

def get_emoji(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    if thumb_tip.y < landmarks[3].y and index_tip.y > landmarks[6].y and middle_tip.y > landmarks[10].y:
        return "👍"

    if (
        index_tip.y < landmarks[6].y 
        and middle_tip.y < landmarks[10].y 
        and ring_tip.y > landmarks[14].y 
        and pinky_tip.y > landmarks[18].y 
    ):
        return "✌️"

    return "🤚"

def overlay_image(bg_image, fg_image, x, y):

    fg_h, fg_w, fg_c = fg_image.shape
    bg_h, bg_w, bg_c = bg_image.shape

    if y + fg_h > bg_h or x + fg_w > bg_w:
        return bg_image

    alpha = fg_image[:, :, 3] / 255.0 
    for c in range(3): 
        bg_image[y:y+fg_h, x:x+fg_w, c] = (
            alpha * fg_image[:, :, c] + (1 - alpha) * bg_image[y:y+fg_h, x:x+fg_w, c]
        )
    return bg_image

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            emoji = get_emoji(hand_landmarks.landmark)
            if emoji in emoji_dict:
                emoji_image = emoji_dict[emoji]
                frame = overlay_image(frame, emoji_image, 50, 50)

    cv2.imshow("Hand Gesture to Emoji", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
