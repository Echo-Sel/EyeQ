import cv2
import numpy as np
import mediapipe as mp
import base64
import math
import threading
import time
from openai import OpenAI
import anthropic
import pyttsx3                      # local text-to-speech
import speech_recognition as sr     # voice questions

import sys
import os
from dotenv import load_dotenv



def get_app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


APP_DIR = get_app_dir()
ENV_PATH = os.path.join(APP_DIR, ".env")

load_dotenv(dotenv_path=ENV_PATH)

FEATHERLESS_KEY = os.environ.get("FEATHERLESS_API_KEY")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")


def prompt_for_keys():
    """First-run setup: ask in the console for whichever key(s) are missing
    and write them to a .env next to the exe, so this only has to happen
    once. Returns the (possibly updated) key values."""
    global FEATHERLESS_KEY, ANTHROPIC_KEY

    print("=" * 60)
    print("EyeQ setup -- no API key(s) found.")
    print("You need at least one of:")
    print("  - a Featherless API key (https://featherless.ai)")
    print("  - an Anthropic API key  (https://console.anthropic.com)")
    print("Press Enter to skip either one.")
    print("=" * 60)

    if not FEATHERLESS_KEY:
        entered = input("Featherless API key (or Enter to skip): ").strip()
        if entered:
            FEATHERLESS_KEY = entered

    if not ANTHROPIC_KEY:
        entered = input("Anthropic API key (or Enter to skip): ").strip()
        if entered:
            ANTHROPIC_KEY = entered

    lines = []
    if FEATHERLESS_KEY:
        lines.append(f"FEATHERLESS_API_KEY={FEATHERLESS_KEY}")
    if ANTHROPIC_KEY:
        lines.append(f"ANTHROPIC_API_KEY={ANTHROPIC_KEY}")

    if lines:
        try:
            with open(ENV_PATH, "w") as f:
                f.write("\n".join(lines) + "\n")
            print(f"Saved to {ENV_PATH} -- you won't be asked again.")
        except Exception as e:
            print(f"Couldn't write .env file ({e}); you'll be asked again next run.")

    return FEATHERLESS_KEY, ANTHROPIC_KEY


if not FEATHERLESS_KEY and not ANTHROPIC_KEY:
    FEATHERLESS_KEY, ANTHROPIC_KEY = prompt_for_keys()

featherless_client = OpenAI(
    base_url="https://api.featherless.ai/v1",
    api_key=FEATHERLESS_KEY,
    timeout=60.0,
) if FEATHERLESS_KEY else None

claude_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY) if ANTHROPIC_KEY else None

if featherless_client is None and claude_client is None:
    print(
        "No API keys provided. Add FEATHERLESS_API_KEY and/or ANTHROPIC_API_KEY to\n"
        f"the .env file at {ENV_PATH}, or re-run and enter one when prompted."
    )
    input("Press Enter to exit...")
    raise SystemExit(1)

VISION_MODEL = "MiniMaxAI/MiniMax-M3"   # Featherless multimodal model
CLAUDE_MODEL = "claude-haiku-4-5-20251001"  # Anthropic vision model (cheapest current tier)
PROVIDER = "featherless" if featherless_client else "claude"  # chosen on the setup screen
DEFAULT_QUESTION = "What's in this image?"


def ask_about_crop(crop_img, question=DEFAULT_QUESTION):
    """Send a cropped frame to whichever vision model the user picked; return a short description."""
    ok, buf = cv2.imencode(".png", crop_img)
    if not ok:
        return "couldn't encode crop"
    b64 = base64.b64encode(buf).decode("utf-8")
    prompt = f"{question} Answer in one or two short, complete sentences. No preamble, just the answer."

    try:
        if PROVIDER == "claude":
            response = claude_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            text = next((b.text for b in response.content if b.type == "text"), "").strip()
            return text or "the model returned a blank answer -- try again or reframe the object"

        # Featherless / OpenAI-compatible path (retry once on an empty message)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }]
        for _ in range(2):
            response = featherless_client.chat.completions.create(
                model=VISION_MODEL,
                max_tokens=2048,
                messages=messages,
            )
            content = (response.choices[0].message.content or "").strip()
            if content:
                return content
        return "the model returned a blank answer -- try again or reframe the object"
    except Exception as e:
        msg = str(e).lower()
        if "timeout" in msg or "timed out" in msg:
            return "model took too long to respond (likely warming up on a cold start) -- try again in a few seconds"
        if "concurren" in msg or "429" in msg or "rate limit" in msg:
            return "too many requests right now, give it a second and try again"
        if "capacity" in msg or "503" in msg:
            return "model is temporarily at capacity -- try again in a few seconds"
        if "authentication" in msg or "401" in msg:
            return "API key error -- check your key in the .env file"
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
STILL_PX = 8
HOLD_TIME = 0.5

detecting = False
detect_result = ""
status_text = ""

request_id = 0
active_tts_engine = None  # so a cancel can stop speech mid-sentence

in_flight_count = 0
MAX_IN_FLIGHT = 2


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
        print("[detect] started, listening on mic...", flush=True)
        status_text = "listening..."
        question = listen_for_question()
        if request_id != my_id:
            return  # canceled while we were listening

        if not question:
            question = DEFAULT_QUESTION
        print(f"[detect] question = {question!r}; asking model...", flush=True)
        status_text = "thinking..."
        result = ask_about_crop(crop, question)
        print(f"[detect] model returned = {result!r}", flush=True)

        if request_id != my_id:
            return  # canceled while the model was responding

        detect_result = result
        status_text = ""
        threading.Thread(target=speak, args=(detect_result,), daemon=True).start()
    except Exception as e:
        print(f"[detect] EXCEPTION: {type(e).__name__}: {e}", flush=True)
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
    request_id += 1
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
        print(f"[trigger] fired in {mode} mode, crop {crop.shape}", flush=True)
        request_id += 1
        detecting = True
        detect_result = ""
        threading.Thread(target=run_detection, args=(crop,), daemon=True).start()
    else:
        print(f"[trigger] NO-OP: nothing to analyze in {mode} mode "
              f"(draw something first, or drag a box)", flush=True)
        status_text = "nothing to analyze -- draw or box something first"


def trigger_clear():
    global canvas, box_min_x, box_min_y, box_max_x, box_max_y, detect_result
    canvas = np.zeros_like(canvas)
    box_min_x = box_min_y = box_max_x = box_max_y = None
    detect_result = ""


# ==================== setup + gesture tutorial ====================

def _hand_flags(hl, w, h):
    """Booleans for the single-hand gestures -- same logic the main loop uses."""
    t = hl.landmark
    index_up = t[8].y < t[6].y
    middle_up = t[12].y < t[10].y
    ring_up = t[16].y < t[14].y
    pinky_up = t[20].y < t[18].y
    thumb_up = t[4].y < t[3].y
    ix, iy = int(t[8].x * w), int(t[8].y * h)
    tx, ty = int(t[4].x * w), int(t[4].y * h)
    pinch = math.hypot(ix - tx, iy - ty) < PINCH_PX
    return {
        "pinch": pinch,
        "open_palm": index_up and middle_up and ring_up and pinky_up,
        "fist": (not index_up and not middle_up and not ring_up and not pinky_up and not thumb_up),
        "thumbs_up": thumb_up and not index_up and not middle_up and not ring_up and not pinky_up,
        "three": index_up and middle_up and ring_up and not pinky_up,
    }


def _banner(img, lines, y0=40, scale=0.7, color=(255, 255, 255)):
    """Draw readable text (black outline + colored fill) at the top-left."""
    for i, line in enumerate(lines):
        pos = (20, y0 + i * 34)
        cv2.putText(img, line, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
        cv2.putText(img, line, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)


def choose_provider():
    """Setup screen -- pick the vision model. Sets and returns the global PROVIDER."""
    global PROVIDER
    while True:
        ret, frame = cap.read()
        if not ret:
            return PROVIDER
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)

        f_ok = featherless_client is not None
        c_ok = claude_client is not None
        _banner(frame, ["EyeQ  --  choose your vision model"], y0=70, scale=1.0, color=(0, 255, 255))
        _banner(frame, [
            f"[1]  Featherless   {VISION_MODEL}" + ("" if f_ok else "   (no key in .env)"),
            f"[2]  Claude        {CLAUDE_MODEL}" + ("" if c_ok else "   (no key in .env)"),
            "",
            "Press 1 or 2 to choose.    ESC = keep default.",
        ], y0=160, scale=0.7)
        cv2.imshow("Drawing", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            return PROVIDER
        if k == ord('1') and f_ok:
            PROVIDER = "featherless"
            return PROVIDER
        if k == ord('2') and c_ok:
            PROVIDER = "claude"
            return PROVIDER


TUTORIAL_STEPS = [
    ("OPEN PALM", "Open hand, all 4 fingers up  ->  CLEAR the canvas/box", "open_palm"),
    ("PINCH", "Touch thumb + index together  ->  ASK the AI about your drawing", "pinch"),
    ("FIST", "Close all fingers into a fist  ->  CANCEL / stop the AI", "fist"),
    ("THUMBS UP", "Thumb up, other fingers curled  ->  SWITCH Draw / Box mode", "thumbs_up"),
    ("THREE FINGERS", "Index+middle+ring up, pinky down  ->  3-2-1 timed SNAPSHOT", "three"),
    ("TWO HANDS", "Show BOTH hands, move apart/together  ->  ZOOM the view", "two_hands"),
]


def run_tutorial():
    """Walk through each gesture; only advance once the user actually performs it."""
    total = len(TUTORIAL_STEPS)
    for idx, (title, desc, key) in enumerate(TUTORIAL_STEPS):
        streak = 0
        done_at = None
        while True:
            ret, frame = cap.read()
            if not ret:
                return
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            active = False
            if res.multi_hand_landmarks:
                for hl in res.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
                if key == "two_hands":
                    active = len(res.multi_hand_landmarks) >= 2
                elif len(res.multi_hand_landmarks) == 1:
                    active = _hand_flags(res.multi_hand_landmarks[0], w, h).get(key, False)

            if done_at is None:
                streak = streak + 1 if active else 0
                if streak >= STABLE_FRAMES:
                    done_at = time.time()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
            cv2.rectangle(overlay, (0, h - 55), (w, h), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
            _banner(frame, [f"STEP {idx + 1}/{total}:  {title}"], y0=45, scale=0.9, color=(0, 255, 255))
            _banner(frame, [desc], y0=95, scale=0.6)
            _banner(frame, ["S = skip step     ESC = skip tutorial"], y0=h - 22, scale=0.55, color=(180, 180, 180))

            if done_at is None:
                frac = min(streak / STABLE_FRAMES, 1.0)
                cv2.rectangle(frame, (20, 122), (20 + int((w - 40) * frac), 136), (0, 200, 0), -1)
                cv2.rectangle(frame, (20, 122), (w - 20, 136), (255, 255, 255), 1)
            else:
                cv2.putText(frame, "NICE!", (w // 2 - 120, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 6)

            cv2.imshow("Drawing", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                return
            if k == ord('s'):
                break
            if done_at is not None and time.time() - done_at > 0.8:
                break


choose_provider()
run_tutorial()


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
    clean_frame = frame.copy()
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
            cv2.imwrite(os.path.join(APP_DIR, f"snapshot_{snapshot_count}.png"), frozen_frame)
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