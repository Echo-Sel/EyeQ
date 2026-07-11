import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,  # adjust post test
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
cv2.namedWindow("Drawing", 0)
canvas = None
prev_points = {}  # tracks prev_x, prev_y separately per hand index

# color palette: press number keys 1-5 to switch, drawn in BGR
COLORS = {
    ord('1'): (0, 255, 255),   # yellow
    ord('2'): (0, 0, 255),     # red
    ord('3'): (0, 255, 0),     # green
    ord('4'): (255, 0, 0),     # blue
    ord('5'): (255, 0, 255),   # magenta
}
current_color = COLORS[ord('1')]

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = frame.copy() * 0

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for i, hand_landmark in enumerate(results.multi_hand_landmarks):

            mp_draw.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

            # index
            index_tip = hand_landmark.landmark[8]
            index_pip = hand_landmark.landmark[6]

            # middle
            mid_tip = hand_landmark.landmark[12]
            mid_pip = hand_landmark.landmark[10]

            # finger up and down checks
            index_up = index_tip.y < index_pip.y
            index_down = index_tip.y > index_pip.y
            middle_up = mid_tip.y < mid_pip.y
            middle_down = mid_tip.y > mid_pip.y

            ix, iy = int(index_tip.x * w), int(index_tip.y * h)

            if index_up and middle_down:

                prev_x, prev_y = prev_points.get(i, (None, None))
                if prev_x is None:
                    prev_x, prev_y = ix, iy

                cv2.line(canvas, (prev_x, prev_y), (ix, iy), current_color, 5)
                prev_points[i] = (ix, iy)

                cv2.putText(frame, "DRAWING", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, current_color, 2)
            elif index_up and middle_up:
                canvas = frame.copy() * 0
                prev_points.pop(i, None)
            else:
                prev_points.pop(i, None)

    frame = cv2.add(frame, canvas)

    # show current color as a small swatch, top right corner
    cv2.rectangle(frame, (w - 60, 10), (w - 10, 60), current_color, -1)
    cv2.rectangle(frame, (w - 60, 10), (w - 10, 60), (255, 255, 255), 2)

    cv2.imshow("Drawing", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key in COLORS:
        current_color = COLORS[key]

cap.release()
cv2.destroyAllWindows()