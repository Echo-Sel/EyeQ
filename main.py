import cv2
import numpy as np
import mediapipe as mp
import base64
import math
import threading
import time
from openai import OpenAI          # Featherless is OpenAI-compatible
import pyttsx3                      # local text-to-speech
import speech_recognition as sr     # voice questions

# ---------------- Featherless (OpenAI-compatible) ----------------
# pip install openai pyttsx3 SpeechRecognition pyaudio
# export FEATHERLESS_API_KEY="your-key-here"
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env in the current working directory into os.environ

FEATHERLESS_KEY = os.environ.get("FEATHERLESS_API_KEY")
if not FEATHERLESS_KEY:
    raise SystemExit(
        "FEATHERLESS_API_KEY is not set.\n"
        "Add this line to a .env file in the same folder as this script:\n"
        "FEATHERLESS_API_KEY=your-key-here\n"
        "(no quotes, no 'export', no spaces around the '=')"
    )

client = OpenAI(
    base_url="https://api.featherless.ai/v1",
    api_key=FEATHERLESS_KEY,
    timeout=25.0,  # fail with a clear error instead of hanging on a cold-start model
)

VISION_MODEL = "MiniMaxAI/MiniMax-M3"  # Featherless multimodal model

DEFAULT_QUESTION = "What's in this image?"


def ask_about_crop(crop_img, question=DEFAULT_QUESTION):
    """Send a cropped frame to the vision model and get back a short description."""
    ok, buf = cv2.imencode(".png", crop_img)
    if not ok:
        return "couldn't encode crop"
    b64 = base64.b64encode(buf).decode("utf-8")

    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            max_tokens=80,  # enough headroom to finish a sentence, low enough it can't ramble
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{question} Answer in one or two short, complete sentences. No preamble, just the answer."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                ]
            }]
        )
        return response.choices[0].message.content
    except Exception as e:
        msg = str(e).lower()
        if "timeout" in msg or "timed out" in msg:
            return "model took too long to respond (likely warming up on a cold start) -- try again in a few seconds"
        if "concurren" in msg or "429" in msg or "rate limit" in msg:
            return "too many requests right now, give it a second and try again"
        raise


def get_ink_bbox(canvas, pad=15):
    ys, xs = np.where(canvas.astype(bool).any(axis=2))
    if len(xs) == 0:
        return None
    h, w = canvas.shape[:2]
    x0 = max(int(xs.min()) - pad, 0)
    y0 = max(int(ys.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad, w - 1)
    y1 = min(int(ys.max()) + pad, h - 1)
    return x0, y0, x1, y1


# ---------------- text-to-speech ----------------
def speak(text):
    """Runs in its own thread so it never blocks the video loop."""
    global active_tts_engine
    try:
        engine = pyttsx3.init()
        active_tts_engine = engine
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("TTS error:", e)
    finally:
        active_tts_engine = None


# ---------------- voice question capture ----------------
def listen_for_question(timeout=4, phrase_time_limit=6):
    """Listens on the default mic and returns transcribed text, or None on failure/timeout."""
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        return recognizer.recognize_google(audio)
    except Exception as e:
        print("Voice capture failed, falling back to default question:", e)
        return None


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
cv2.namedWindow("Drawing", 0)
canvas = None
prev_points = {}

COLORS = {
    ord('1'): (255, 255, 255),  # white
    ord('2'): (0, 255, 255),    # yellow
    ord('3'): (0, 0, 255),      # red
    ord('4'): (0, 255, 0),      # green
    ord('5'): (255, 0, 0),      # blue
    ord('6'): (255, 0, 255),    # magenta
}
current_color = COLORS[ord('1')]

mode = "draw"
MODE_BTN_TL, MODE_BTN_BR = (10, 10), (160, 55)
HELP_BTN_TL, HELP_BTN_BR = (170, 10), (215, 55)
show_help = False

box_dragging = False
box_start = None
box_min_x = box_min_y = box_max_x = box_max_y = None
box_last_pos = None
box_last_move_time = 0
box_locked = False
STILL_PX = 8            # movement under this is considered "holding still"
HOLD_TIME = 0.5          # seconds of stillness before the box freezes

detecting = False
detect_result = ""
status_text = ""  # shows "listening..." etc

request_id = 0            # bumped on every new trigger AND on cancel; stale threads discard their result
active_tts_engine = None  # so a cancel can stop speech mid-sentence

in_flight_count = 0        # requests currently running server-side (even canceled ones, until they finish)
MAX_IN_FLIGHT = 2          # stay well under Featherless's per-plan concurrency limit

# ---- gesture-trigger debounce state, tracked per hand index ----
# instead of firing the instant a gesture is detected for one frame (jittery,
# causes accidental mode switches), each gesture must hold for STABLE_FRAMES
# consecutive frames before it fires.
pinch_streak = {}
open_palm_streak = {}
fist_streak = {}
thumbs_up_streak = {}
pinch_fired = {}       # hand i -> bool, so a held pinch doesn't refire every frame after confirming
open_palm_fired = {}
fist_fired = {}
thumbs_up_fired = {}

STABLE_FRAMES = 5       # ~0.15-0.2s at 30fps -- must hold the shape this long to count
PINCH_PX = 40          # thumb tip to index tip distance threshold, in pixels
GESTURE_COOLDOWN = 1.5  # seconds, prevents immediate re-trigger after firing
last_trigger_time = 0

# ---- new-feature state (none of this touches the AI request/response path) ----
EMA_ALPHA = 0.5            # fingertip smoothing: higher = snappier, lower = smoother
smoothed_points = {}       # hand i -> (sx, sy), exponential moving average of the fingertip

BRUSH_MIN, BRUSH_MAX = 2, 18   # brush thickness scales with how close the hand is (Z-axis depth)

zoom_factor = 1.0          # current digital zoom, driven by two-hand spread
ZOOM_ALPHA = 0.2           # heavy smoothing on the zoom -> kills the twitch/jitter
ZOOM_MAX = 3.0

three_finger_streak = {}   # debounce for the 3-finger timed-snapshot gesture
three_finger_fired = {}
countdown_active = False
countdown_start = 0.0
COUNTDOWN_SECS = 3
frozen_frame = None        # the frozen still shown after the countdown fires
frozen_until = 0.0
snapshot_count = 0

HELP_LINES = [
    "CONTROLS",
    "click button / 'm' / THUMBS UP  -  toggle Draw / Box mode",
    "1-6                 -  change draw color",
    "DRAW: point one finger to draw/circle",
    "BOX: point one finger, drag to size; hold still ~0.5s to lock it in place",
    "PINCH (quick thumb+index touch, no need to hold) -  ask about the drawing/box",
    "  -- everything freezes while it's listening/thinking, so nothing moves",
    "OPEN PALM (4 fingers up)  -  clear canvas and box",
    "FIST (all 5 fingers curled) -  cancel/cut off the AI mid-response",
    "THREE FINGERS (index+middle+ring, pinky down) -  3-2-1 countdown then freeze a snapshot",
    "TWO HANDS - move apart/together to zoom the view in/out",
    "brush gets thicker as your hand moves closer to the camera",
    "'d' / 'c' / 'x' / 't'  -  keyboard fallbacks: detect / clear / cancel / timed snapshot",
    "'h' / click '?'      -  toggle this help",
    "ESC                 -  quit",
]


def run_detection(crop):
    global detect_result, detecting, status_text, request_id, in_flight_count
    my_id = request_id
    in_flight_count += 1
    try:
        status_text = "listening..."
        question = listen_for_question()
        if request_id != my_id:
            return  # canceled while we were listening

        if not question:
            question = DEFAULT_QUESTION
        status_text = "thinking..."
        result = ask_about_crop(crop, question)

        if request_id != my_id:
            return  # canceled while the model was responding

        detect_result = result
        status_text = ""
        threading.Thread(target=speak, args=(detect_result,), daemon=True).start()
    except Exception as e:
        if request_id == my_id:
            detect_result = f"error: {e}"
            status_text = ""
    finally:
        in_flight_count -= 1
        if request_id == my_id:
            detecting = False


def cancel_current():
    """Cuts off an in-progress listen/think/speak cycle."""
    global request_id, detecting, status_text
    request_id += 1  # any in-flight thread will see this mismatch and discard its result
    detecting = False
    status_text = ""
    if active_tts_engine is not None:
        try:
            active_tts_engine.stop()
        except Exception:
            pass


def on_mouse(event, x, y, flags, param):
    global mode, show_help
    if event == cv2.EVENT_LBUTTONDOWN:
        if MODE_BTN_TL[0] <= x <= MODE_BTN_BR[0] and MODE_BTN_TL[1] <= y <= MODE_BTN_BR[1]:
            mode = "box" if mode == "draw" else "draw"
        elif HELP_BTN_TL[0] <= x <= HELP_BTN_BR[0] and HELP_BTN_TL[1] <= y <= HELP_BTN_BR[1]:
            show_help = not show_help


def gesture_confirmed(streak_dict, fired_dict, hand_i, is_active):
    """Returns True exactly once, after a gesture shape has been held for
    STABLE_FRAMES consecutive frames. Resets the moment the shape breaks,
    so it can fire again next time it's formed."""
    if not is_active:
        streak_dict[hand_i] = 0
        fired_dict[hand_i] = False
        return False

    streak_dict[hand_i] = streak_dict.get(hand_i, 0) + 1
    if streak_dict[hand_i] >= STABLE_FRAMES and not fired_dict.get(hand_i, False):
        fired_dict[hand_i] = True
        return True
    return False


cv2.setMouseCallback("Drawing", on_mouse)


def trigger_detect(clean_frame):
    """Shared trigger path for both the pinch gesture and the 'd' key fallback."""
    global detecting, detect_result, request_id, status_text
    if in_flight_count >= MAX_IN_FLIGHT:
        status_text = "still finishing up a previous request, one sec"
        return

    crop = None
    if mode == "draw":
        ink_bbox = get_ink_bbox(canvas)
        if ink_bbox:
            x0, y0, x1, y1 = ink_bbox
            crop = clean_frame[y0:y1, x0:x1].copy()
    elif mode == "box" and box_min_x is not None:
        crop = clean_frame[box_min_y:box_max_y, box_min_x:box_max_x].copy()

    if crop is not None and crop.size > 0:
        request_id += 1
        detecting = True
        detect_result = ""
        threading.Thread(target=run_detection, args=(crop,), daemon=True).start()


def trigger_clear():
    global canvas, box_min_x, box_min_y, box_max_x, box_max_y, detect_result
    canvas = np.zeros_like(canvas)
    box_min_x = box_min_y = box_max_x = box_max_y = None
    detect_result = ""


while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    now = time.time()

    if canvas is None:
        canvas = frame.copy() * 0

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        num_hands = len(results.multi_hand_landmarks)

        for hand_landmark in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

        if num_hands >= 2:
            # ---- two-hand spread -> dynamic digital zoom (EMA-smoothed to kill the twitch) ----
            wrist_a = results.multi_hand_landmarks[0].landmark[0]
            wrist_b = results.multi_hand_landmarks[1].landmark[0]
            spread = math.hypot((wrist_a.x - wrist_b.x) * w, (wrist_a.y - wrist_b.y) * h)
            target_zoom = float(np.interp(spread / w, [0.15, 0.60], [1.0, ZOOM_MAX]))
            zoom_factor = ZOOM_ALPHA * target_zoom + (1 - ZOOM_ALPHA) * zoom_factor
            # while zooming with two hands we deliberately don't draw or fire single-hand gestures

        else:
            for i, hand_landmark in enumerate(results.multi_hand_landmarks):

                thumb_tip = hand_landmark.landmark[4]
                index_tip = hand_landmark.landmark[8]
                index_pip = hand_landmark.landmark[6]
                mid_tip = hand_landmark.landmark[12]
                mid_pip = hand_landmark.landmark[10]
                ring_tip = hand_landmark.landmark[16]
                ring_pip = hand_landmark.landmark[14]
                pinky_tip = hand_landmark.landmark[20]
                pinky_pip = hand_landmark.landmark[18]
                wrist = hand_landmark.landmark[0]
                mid_mcp = hand_landmark.landmark[9]

                index_up = index_tip.y < index_pip.y
                middle_up = mid_tip.y < mid_pip.y
                middle_down = mid_tip.y > mid_pip.y
                ring_up = ring_tip.y < ring_pip.y
                pinky_up = pinky_tip.y < pinky_pip.y

                ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)

                # ---- kinematic smoothing (EMA) on the fingertip -> steadier lines & box edges ----
                psx, psy = smoothed_points.get(i, (ix, iy))
                sx = int(EMA_ALPHA * ix + (1 - EMA_ALPHA) * psx)
                sy = int(EMA_ALPHA * iy + (1 - EMA_ALPHA) * psy)
                smoothed_points[i] = (sx, sy)

                # ---- brush thickness from depth: apparent hand size = how close the hand is ----
                hand_px = math.hypot((wrist.x - mid_mcp.x) * w, (wrist.y - mid_mcp.y) * h)
                brush = int(np.clip(np.interp(hand_px, [50, 160], [BRUSH_MIN, BRUSH_MAX]),
                                    BRUSH_MIN, BRUSH_MAX))

                # ---- pinch gesture (thumb tip touching index tip) -> trigger detect ----
                pinch_dist = math.hypot(ix - tx, iy - ty)
                is_pinching = pinch_dist < PINCH_PX
                if gesture_confirmed(pinch_streak, pinch_fired, i, is_pinching) and not detecting \
                        and (now - last_trigger_time) > GESTURE_COOLDOWN:
                    last_trigger_time = now
                    # clean_frame gets built below each loop; grab a fresh crop source now
                    mask_now = canvas.astype(bool).any(axis=2)
                    snapshot = frame.copy()
                    snapshot[mask_now] = canvas[mask_now]
                    trigger_detect(snapshot)

                # ---- open palm gesture (index+middle+ring+pinky all up) -> clear ----
                is_open_palm = index_up and middle_up and ring_up and pinky_up
                if gesture_confirmed(open_palm_streak, open_palm_fired, i, is_open_palm) and not detecting \
                        and (now - last_trigger_time) > GESTURE_COOLDOWN:
                    last_trigger_time = now
                    trigger_clear()

                thumb_ip = hand_landmark.landmark[3]
                thumb_up = thumb_tip.y < thumb_ip.y

                # ---- closed fist (all five fingers curled, including thumb) -> cancel/cut off the AI ----
                is_fist = not index_up and not middle_up and not ring_up and not pinky_up and not thumb_up
                # fire while thinking OR while speaking -- detecting drops to False the moment
                # the speak thread launches, so guarding on `detecting` alone would make it
                # impossible to cut off the TTS mid-sentence.
                if gesture_confirmed(fist_streak, fist_fired, i, is_fist) \
                        and (detecting or active_tts_engine is not None) \
                        and (now - last_trigger_time) > GESTURE_COOLDOWN:
                    last_trigger_time = now
                    cancel_current()

                # ---- thumbs up (thumb extended upward, other four fingers curled) -> toggle mode ----
                is_thumbs_up = thumb_up and not index_up and not middle_up and not ring_up and not pinky_up
                if gesture_confirmed(thumbs_up_streak, thumbs_up_fired, i, is_thumbs_up) \
                        and (now - last_trigger_time) > GESTURE_COOLDOWN:
                    last_trigger_time = now
                    mode = "box" if mode == "draw" else "draw"

                # ---- three fingers (index+middle+ring up, pinky down) -> start timed snapshot ----
                is_three = index_up and middle_up and ring_up and not pinky_up
                if gesture_confirmed(three_finger_streak, three_finger_fired, i, is_three) \
                        and not countdown_active and not detecting \
                        and (now - last_trigger_time) > GESTURE_COOLDOWN:
                    last_trigger_time = now
                    countdown_active = True
                    countdown_start = now

                if mode == "draw" and not detecting:
                    if index_up and middle_down and not is_pinching:
                        prev_x, prev_y = prev_points.get(i, (None, None))
                        if prev_x is None:
                            prev_x, prev_y = sx, sy
                        cv2.line(canvas, (prev_x, prev_y), (sx, sy), current_color, brush)
                        prev_points[i] = (sx, sy)
                    else:
                        prev_points.pop(i, None)

                elif mode == "box" and i == 0 and not detecting:
                    if index_up and not is_pinching:
                        if not box_dragging:
                            box_dragging = True
                            box_start = (sx, sy)
                            box_last_pos = (sx, sy)
                            box_last_move_time = now
                            box_locked = False
                            detect_result = ""
                        else:
                            moved = math.hypot(sx - box_last_pos[0], sy - box_last_pos[1])
                            if moved > STILL_PX:
                                box_last_pos = (sx, sy)
                                box_last_move_time = now
                                box_locked = False
                            elif not box_locked and (now - box_last_move_time) > HOLD_TIME:
                                box_locked = True  # holding still -- freeze the box in place

                        if not box_locked:
                            box_min_x, box_max_x = sorted((box_start[0], sx))
                            box_min_y, box_max_y = sorted((box_start[1], sy))
                    else:
                        box_dragging = False
                        box_locked = False

    mask = canvas.astype(bool).any(axis=2)
    frame[mask] = canvas[mask]
    clean_frame = frame.copy()  # unzoomed camera+ink -- the AI always analyzes THIS, never the zoomed view

    display = frame

    # ---- dynamic digital zoom: crop a centered region and scale it back up (scene only) ----
    if zoom_factor > 1.01:
        zh, zw = int(h / zoom_factor), int(w / zoom_factor)
        y1c, x1c = (h - zh) // 2, (w - zw) // 2
        display = cv2.resize(display[y1c:y1c + zh, x1c:x1c + zw], (w, h))

    scene = display.copy()  # zoomed scene without UI chrome -- this is what a snapshot captures

    # ---- 3-finger timed capture: run the countdown, then freeze a still ----
    if countdown_active:
        elapsed = now - countdown_start
        if elapsed >= COUNTDOWN_SECS:
            frozen_frame = scene.copy()
            snapshot_count += 1
            cv2.imwrite(f"snapshot_{snapshot_count}.png", frozen_frame)
            frozen_until = now + 3.0
            countdown_active = False
        else:
            n = COUNTDOWN_SECS - int(elapsed)
            cv2.putText(display, str(n), (w // 2 - 40, h // 2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 8)

    if frozen_frame is not None and now < frozen_until:
        display = frozen_frame.copy()
        cv2.putText(display, f"snapshot_{snapshot_count}.png saved", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    elif frozen_frame is not None:
        frozen_frame = None

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
        cv2.putText(display, status_text or "thinking...", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    elif status_text:
        cv2.putText(display, status_text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    elif detect_result:
        cv2.putText(display, detect_result[:60], (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if show_help:
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        display = cv2.addWeighted(overlay, 0.7, display, 0.3, 0)
        y0 = 90
        for line in HELP_LINES:
            cv2.putText(display, line, (30, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            y0 += 28

    if zoom_factor > 1.01:
        cv2.putText(display, f"ZOOM x{zoom_factor:.1f}", (w - 170, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Drawing", display)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('t') and not countdown_active:
        countdown_active = True
        countdown_start = now
    elif key == ord('m'):
        mode = "box" if mode == "draw" else "draw"
    elif key == ord('h'):
        show_help = not show_help
    elif key in COLORS:
        current_color = COLORS[key]
    elif key == ord('c'):
        trigger_clear()
    elif key == ord('x'):
        cancel_current()
    elif key == ord('d') and not detecting:
        trigger_detect(clean_frame)

cap.release()
cv2.destroyAllWindows()