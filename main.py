import cv2
import numpy as np
import mediapipe as mp
import base64
import threading
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


def get_ink_bbox(canvas, pad=15):
    """Bounding box of everything drawn on the canvas so far, with a little padding."""
    ys, xs = np.where(canvas.astype(bool).any(axis=2))
    if len(xs) == 0:
        return None
    h, w = canvas.shape[:2]
    x0 = max(int(xs.min()) - pad, 0)
    y0 = max(int(ys.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad, w - 1)
    y1 = min(int(ys.max()) + pad, h - 1)
    return x0, y0, x1, y1


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
    ord('1'): (255, 255, 255),  # white
    ord('2'): (0, 255, 255),    # yellow
    ord('3'): (0, 0, 255),      # red
    ord('4'): (0, 255, 0),      # green
    ord('5'): (255, 0, 0),      # blue
    ord('6'): (255, 0, 255),    # magenta
}
current_color = COLORS[ord('1')]

# mode: "draw" or "box" 
mode = "draw"
MODE_BTN_TL, MODE_BTN_BR = (10, 10), (160, 55)
HELP_BTN_TL, HELP_BTN_BR = (170, 10), (215, 55)
show_help = False

# box mode 
box_dragging = False
box_start = None
box_min_x = box_min_y = box_max_x = box_max_y = None
#detection
detecting = False
detect_result = ""

HELP_LINES = [
    "CONTROLS",
    "click button / 'm'  -  toggle Draw / Box mode",
    "1-6                 -  change draw color",
    "DRAW: point one finger to draw/circle, two fingers up to clear",
    "BOX: point one finger, drag to size the box, lower finger to lock",
    "'d'                 -  ask Claude about what you drew/boxed",
    "'c'                 -  clear canvas and box",
    "'h' / click '?'      -  toggle this help",
    "ESC                 -  quit",
]


def run_detection(crop):
    global detect_result, detecting
    try:
        detect_result = ask_about_crop(crop)
    except Exception as e:
        detect_result = f"error: {e}"
    detecting = False


def on_mouse(event, x, y, flags, param):
    global mode, show_help
    if event == cv2.EVENT_LBUTTONDOWN:
        if MODE_BTN_TL[0] <= x <= MODE_BTN_BR[0] and MODE_BTN_TL[1] <= y <= MODE_BTN_BR[1]:
            mode = "box" if mode == "draw" else "draw"
        elif HELP_BTN_TL[0] <= x <= HELP_BTN_BR[0] and HELP_BTN_TL[1] <= y <= HELP_BTN_BR[1]:
            show_help = not show_help


cv2.setMouseCallback("Drawing", on_mouse)

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
            middle_up = mid_tip.y < mid_pip.y
            middle_down = mid_tip.y > mid_pip.y

            ix, iy = int(index_tip.x * w), int(index_tip.y * h)

            if mode == "draw":
                if index_up and middle_down:

                    prev_x, prev_y = prev_points.get(i, (None, None))
                    if prev_x is None:
                        prev_x, prev_y = ix, iy

                    cv2.line(canvas, (prev_x, prev_y), (ix, iy), current_color, 5)
                    prev_points[i] = (ix, iy)
                elif index_up and middle_up:
                    canvas = frame.copy() * 0
                    prev_points.pop(i, None)
                    detect_result = ""
                else:
                    prev_points.pop(i, None)

            elif mode == "box" and i == 0:
                # only the first detected hand drives the box, to keep it simple
                if index_up:
                    if not box_dragging:
                        box_dragging = True
                        box_start = (ix, iy)
                        detect_result = ""
                    box_min_x, box_max_x = sorted((box_start[0], ix))
                    box_min_y, box_max_y = sorted((box_start[1], iy))
                else:
                    box_dragging = False  # lifting the finger locks the box in place


    mask = canvas.astype(bool).any(axis=2)
    frame[mask] = canvas[mask]
    clean_frame = frame.copy()

    display = frame

    btn_color = (0, 200, 0) if mode == "draw" else (0, 140, 255)
    cv2.rectangle(display, MODE_BTN_TL, MODE_BTN_BR, btn_color, -1)
    cv2.putText(display, mode.upper(), (MODE_BTN_TL[0] + 15, MODE_BTN_TL[1] + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.rectangle(display, HELP_BTN_TL, HELP_BTN_BR, (90, 90, 90), -1)
    cv2.putText(display, "?", (HELP_BTN_TL[0] + 15, HELP_BTN_TL[1] + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    if mode == "draw":
        cv2.rectangle(display, (w - 60, 10), (w - 10, 60), current_color, -1)
        cv2.rectangle(display, (w - 60, 10), (w - 10, 60), (255, 255, 255), 2)
        cv2.putText(display, "DRAWING", (50, 80), cv2.FONT_HERSHEY_COMPLEX, 1, current_color, 2)

    if mode == "box" and box_min_x is not None:
        cv2.rectangle(display, (box_min_x, box_min_y), (box_max_x, box_max_y), (0, 255, 0), 2)

    if detecting:
        cv2.putText(display, "thinking...", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    elif detect_result:
        cv2.putText(display, detect_result[:60], (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if show_help:
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        display = cv2.addWeighted(overlay, 0.7, display, 0.3, 0)
        y0 = 90
        for line in HELP_LINES:
            cv2.putText(display, line, (30, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y0 += 30

    cv2.imshow("Drawing", display)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('m'):
        mode = "box" if mode == "draw" else "draw"
    elif key == ord('h'):
        show_help = not show_help
    elif key in COLORS:
        current_color = COLORS[key]
    elif key == ord('c'):
        canvas = frame.copy() * 0
        box_min_x = box_min_y = box_max_x = box_max_y = None
        detect_result = ""
    elif key == ord('d') and not detecting:
        crop = None
        if mode == "draw":
            ink_bbox = get_ink_bbox(canvas)
            if ink_bbox:
                x0, y0, x1, y1 = ink_bbox
                crop = clean_frame[y0:y1, x0:x1].copy()
        elif mode == "box" and box_min_x is not None:
            crop = clean_frame[box_min_y:box_max_y, box_min_x:box_max_x].copy()

        if crop is not None and crop.size > 0:
            detecting = True
            detect_result = ""
            threading.Thread(target=run_detection, args=(crop,), daemon=True).start()

cap.release()
cv2.destroyAllWindows()