import cv2
import mediapipe as mp
import base64
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

def ask_about_crop(crop_img, question="What's in this image? Be concise."):
    """Send a cropped frame to Claude and get back a short description."""
    ok, buf = cv2.imencode(".png", crop_img)
    if not ok:
        return "couldn't encode crop"
    b64 = base64.b64encode(buf).decode("utf-8")

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                {"type": "text", "text": question}
            ]
        }]
    )
    return response.content[0].text

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

# color palette: press number keys 1-6 to switch, drawn in BGR
COLORS = {
    ord('1'): (255, 255, 255), # white
    ord('2'): (0, 255, 255),   # yellow
    ord('3'): (0, 0, 255),     # red
    ord('4'): (0, 255, 0),     # green
    ord('5'): (255, 0, 0),     # blue
    ord('6'): (255, 0, 255),   # magenta
}
current_color = COLORS[ord('1')]

# bounding box of everything drawn so far 
box_min_x, box_min_y = None, None
box_max_x, box_max_y = None, None
detect_result = ""

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

                # grow the tracked bounding box to include this point
                box_min_x = ix if box_min_x is None else min(box_min_x, ix)
                box_min_y = iy if box_min_y is None else min(box_min_y, iy)
                box_max_x = ix if box_max_x is None else max(box_max_x, ix)
                box_max_y = iy if box_max_y is None else max(box_max_y, iy)

                cv2.putText(frame, "DRAWING", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, current_color, 2)
            elif index_up and middle_up:
                canvas = frame.copy() * 0
                prev_points.pop(i, None)
                box_min_x = box_min_y = box_max_x = box_max_y = None
                detect_result = ""
            else:
                prev_points.pop(i, None)

    mask = canvas.astype(bool).any(axis=2)
    frame[mask] = canvas[mask]

    # show current color as a small swatch, top-right corner
    cv2.rectangle(frame, (w - 60, 10), (w - 10, 60), current_color, -1)
    cv2.rectangle(frame, (w - 60, 10), (w - 10, 60), (255, 255, 255), 2)

    # draw the tracked bounding box so you can see what will be cropped
    if box_min_x is not None:
        cv2.rectangle(frame, (box_min_x, box_min_y), (box_max_x, box_max_y), (0, 255, 0), 2)

    if detect_result:
        cv2.putText(frame, detect_result[:60], (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Drawing", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key in COLORS:
        current_color = COLORS[key]
    elif key == ord('d') and box_min_x is not None:
        crop = frame[box_min_y:box_max_y, box_min_x:box_max_x]
        if crop.size > 0:
            detect_result = ask_about_crop(crop)
            print("Claude says:", detect_result)

cap.release()
cv2.destroyAllWindows()