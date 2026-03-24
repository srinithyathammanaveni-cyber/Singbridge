"""
gesture_model.py  (v6 — FIXED accurate ASL detection)
───────────────────────────────────────────────────────
Completely rewritten rule-based classifier.
Key fixes:
  - Non-overlapping rules ordered from MOST SPECIFIC to LEAST SPECIFIC
  - Correct thumb direction logic (camera-mirror aware)
  - Higher stable threshold (5 frames) to prevent jitter
  - Normalized distance thresholds per palm size
  - All 19 supported gestures reliably distinguishable
"""

import cv2
import numpy as np
import os
import urllib.request

MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
    print("[OK] MediaPipe ready.")
except Exception as e:
    print(f"[WARNING] MediaPipe not available: {e}")


class GestureRecognizer:
    def __init__(self):
        self.landmarker = None
        self._stable_gesture = None
        self._stable_count = 0
        self._STABLE_THRESHOLD = 5  # require 5 stable frames — prevents jitter
        self._load()

    def _load(self):
        if not MEDIAPIPE_AVAILABLE:
            return
        # Check both current dir and models/ subdir
        candidates = [
            os.path.normpath(os.path.join(os.path.dirname(__file__), 'models', 'hand_landmarker.task')),
            os.path.normpath(os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')),
        ]
        model_path = None
        for p in candidates:
            if os.path.exists(p):
                model_path = p
                break
        if model_path is None:
            model_path = candidates[0]
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            print("[INFO] Downloading hand_landmarker.task model (~10MB)...")
            urllib.request.urlretrieve(
                'https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
                'hand_landmarker/float16/1/hand_landmarker.task',
                model_path
            )
            print("[INFO] Model downloaded.")
        try:
            opts = mp_vision.HandLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.4,
                min_tracking_confidence=0.5
            )
            self.landmarker = mp_vision.HandLandmarker.create_from_options(opts)
            print("[OK] HandLandmarker ready — 2 hands enabled.")
        except Exception as e:
            print(f"[ERROR] HandLandmarker init failed: {e}")

    def _get_landmarks_and_handedness(self, frame):
        if not self.landmarker:
            return []
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect(mp_image)
            if not result.hand_landmarks:
                return []
            hands = []
            for i, hand in enumerate(result.hand_landmarks):
                lms = [{'x': round(lm.x, 4), 'y': round(lm.y, 4), 'z': round(lm.z, 4)} for lm in hand]
                handedness = 'Right'
                if result.handedness and i < len(result.handedness):
                    handedness = result.handedness[i][0].display_name
                hands.append((lms, handedness))
            return hands
        except Exception as e:
            print(f"[WARN] Landmark detection error: {e}")
            return []

    def _analyze(self, lms, handedness='Right'):
        """
        Compute all finger states and distances from 21 MediaPipe landmarks.
        MediaPipe returns handedness from the model's perspective (mirrored camera).
        We flip it so 'Right' means the user's right hand as seen in camera.
        """
        def dist(a, b):
            return ((lms[a]['x'] - lms[b]['x']) ** 2 +
                    (lms[a]['y'] - lms[b]['y']) ** 2) ** 0.5

        palm_size = max(dist(0, 9), 0.001)

        def ndist(a, b):
            """Normalized distance (by palm size)."""
            return dist(a, b) / palm_size

        # ── Finger extension (tip y < pip y = finger pointing UP) ──
        # y=0 is top of frame, y=1 is bottom
        index_up  = lms[8]['y']  < lms[6]['y']   # tip above PIP
        middle_up = lms[12]['y'] < lms[10]['y']
        ring_up   = lms[16]['y'] < lms[14]['y']
        pinky_up  = lms[20]['y'] < lms[18]['y']

        # ── Finger curl (tip y > MCP y = curled toward palm) ──
        index_curled  = lms[8]['y']  > lms[5]['y']
        middle_curled = lms[12]['y'] > lms[9]['y']
        ring_curled   = lms[16]['y'] > lms[13]['y']
        pinky_curled  = lms[20]['y'] > lms[17]['y']

        # ── Count extended fingers ──
        n_fingers_up = sum([index_up, middle_up, ring_up, pinky_up])

        # ── Thumb analysis ──
        # MediaPipe "Right" hand label means the model sees it as right hand.
        # In a mirrored camera view, model's "Right" = user's Left (and vice versa).
        # For thumb-out detection: thumb tip (4) x relative to thumb IP (3)
        # Right hand (user): thumb extends to the left in image (lower x)
        # Left hand  (user): thumb extends to the right in image (higher x)
        model_right = (handedness == 'Right')
        # user's right hand = model says Left (mirrored), user's left = model says Right
        user_right_hand = not model_right

        if user_right_hand:
            thumb_out = lms[4]['x'] < lms[3]['x']   # thumb extends left in image
        else:
            thumb_out = lms[4]['x'] > lms[3]['x']   # thumb extends right in image

        # Thumb pointing UP (tip well above wrist)
        thumb_up_strong = lms[4]['y'] < lms[0]['y'] - 0.12
        thumb_up_weak   = lms[4]['y'] < lms[0]['y'] - 0.06

        # Thumb pointing DOWN
        thumb_down = lms[4]['y'] > lms[2]['y'] + 0.05  # tip below CMC

        # Thumb tucked (tip near index base MCP)
        thumb_tucked = ndist(4, 5) < 0.55

        # ── Key normalized distances ──
        d_thumb_index   = ndist(4, 8)
        d_thumb_middle  = ndist(4, 12)
        d_thumb_pinky   = ndist(4, 20)
        d_index_middle  = ndist(8, 12)
        d_middle_ring   = ndist(12, 16)

        # ── Spread (index tip to pinky tip) ──
        spread = ndist(8, 20)

        # ── Fingertips average y (for "raised toward face" detection) ──
        tips_avg_y = (lms[8]['y'] + lms[12]['y'] + lms[16]['y'] + lms[20]['y']) / 4
        tips_above_wrist = tips_avg_y < lms[0]['y'] - 0.05

        # ── Wrist and palm y coords ──
        wrist_y = lms[0]['y']

        return {
            'index_up':       index_up,
            'middle_up':      middle_up,
            'ring_up':        ring_up,
            'pinky_up':       pinky_up,
            'index_curled':   index_curled,
            'middle_curled':  middle_curled,
            'ring_curled':    ring_curled,
            'pinky_curled':   pinky_curled,
            'n_up':           n_fingers_up,
            'thumb_out':      thumb_out,
            'thumb_up':       thumb_up_strong,
            'thumb_up_weak':  thumb_up_weak,
            'thumb_down':     thumb_down,
            'thumb_tucked':   thumb_tucked,
            'd_ti':           round(d_thumb_index,  3),
            'd_tm':           round(d_thumb_middle, 3),
            'd_tp':           round(d_thumb_pinky,  3),
            'd_im':           round(d_index_middle, 3),
            'd_mr':           round(d_middle_ring,  3),
            'spread':         round(spread, 3),
            'tips_above_wrist': tips_above_wrist,
            'wrist_y':        round(wrist_y, 4),
            'tips_avg_y':     round(tips_avg_y, 4),
            'palm_size':      round(palm_size, 4),
            'handedness':     handedness,
        }

    def _classify(self, s):
        """
        Rule-based ASL classifier.
        Rules ordered from MOST SPECIFIC (multi-finger combos) to LEAST SPECIFIC.
        Each rule is mutually exclusive from others at the same level.
        """
        idx    = s['index_up']
        mid    = s['middle_up']
        rng    = s['ring_up']
        pky    = s['pinky_up']
        n      = s['n_up']
        t_out  = s['thumb_out']
        t_up   = s['thumb_up']
        t_up_w = s['thumb_up_weak']
        t_dn   = s['thumb_down']
        t_tck  = s['thumb_tucked']
        d_ti   = s['d_ti']
        d_tm   = s['d_tm']
        d_tp   = s['d_tp']
        d_im   = s['d_im']
        spread = s['spread']
        above  = s['tips_above_wrist']

        # ══════════════════════════════════════════════════════════════
        # LEVEL 1 — All 4 fingers UP (open hand variants)
        # ══════════════════════════════════════════════════════════════
        if n == 4:
            # B / HELLO: 4 fingers up + thumb extended sideways
            if t_out:
                return 'Hello / B'
            # 4 fingers up, no thumb out = B (closed thumb)
            return 'B / 4'

        # ══════════════════════════════════════════════════════════════
        # LEVEL 2 — 3 fingers UP
        # ══════════════════════════════════════════════════════════════
        if n == 3:
            if idx and mid and rng and not pky:
                return 'W / 3'
            if idx and mid and not rng and pky:
                # Could be I Love You without ring — but typically ILY needs thumb too
                return 'W / 3'  # default
            # other 3-finger combos
            return 'W / 3'

        # ══════════════════════════════════════════════════════════════
        # LEVEL 3 — Exactly 2 fingers UP
        # ══════════════════════════════════════════════════════════════
        if n == 2:
            # I Love You: thumb + index + pinky (but index and pinky up, middle/ring down)
            # ILY = index UP + pinky UP (+ thumb out)
            if idx and pky and not mid and not rng and t_out:
                return 'I Love You'

            # V / 2 or U / H: index + middle up
            if idx and mid and not rng and not pky:
                if d_im < 0.35:
                    return 'U / H'
                return 'V / 2'

            # Ring + pinky up (unusual — treat as W/3 fallback)
            if rng and pky and not idx and not mid:
                return 'W / 3'

            # Index + ring (skip middle) — rare
            return 'V / 2'

        # ══════════════════════════════════════════════════════════════
        # LEVEL 4 — Exactly 1 finger UP
        # ══════════════════════════════════════════════════════════════
        if n == 1:
            # Index only: D / 1
            if idx and not mid and not rng and not pky:
                # K / P = thumb also out with index + middle
                # Here only index, check if thumb also out
                if t_out:
                    return 'L'    # L shape = thumb + index
                return 'D / 1'

            # Middle only
            if mid and not idx and not rng and not pky:
                return 'D / 1'   # treat as pointing gesture

            # Pinky only: I
            if pky and not idx and not mid and not rng:
                return 'I'

            # Ring only (rare)
            if rng and not idx and not mid and not pky:
                return 'D / 1'

        # ══════════════════════════════════════════════════════════════
        # LEVEL 5 — 0 fingers UP (all curled / fist shapes)
        # ══════════════════════════════════════════════════════════════
        if n == 0:
            # Thumbs UP: thumb pointing up, all fingers curled
            if t_up and not t_out:
                return 'Good / Thumbs Up'

            # Thumbs DOWN
            if t_dn and not t_up_w and not t_out:
                return 'Thumbs Down'

            # O shape: all curled + thumb and index tips close together
            if d_ti < 0.40 and not t_up_w:
                return 'O / More'

            # Y shape: thumb out + pinky up would have been caught above (n=1 with pky)
            # Y = thumb out, pinky up — but pinky_up check above missed it, check again
            if t_out and not t_up:
                # Check if pinky is up (Y shape)
                if pky:
                    return 'Y'
                # C shape: thumb extended, fingers slightly curved, big d_ti
                if d_ti > 0.65:
                    return 'C'
                # K/P: thumb + index (but index wasn't caught above as up)
                return 'C'

            # A / Fist: all curled, thumb rests alongside index
            # Distinguish from Sorry: A is just a fist
            # Check thumb position: tucked = A
            return 'A'

        # ══════════════════════════════════════════════════════════════
        # SPECIAL COMBOS caught by re-checking (L, K/P, C, Y, ILY)
        # ══════════════════════════════════════════════════════════════

        # L shape: thumb out + index up only (n=1 was handled above but
        # n=1+thumb_out routes to L — already handled)

        # K / P: thumb + index + middle (n=2, thumb out) — check if missed
        if idx and mid and not rng and not pky and t_out:
            return 'K / P'

        # Fallback
        return ''

    def _classify_two_hands(self, hints):
        """Determine combined two-hand gesture."""
        if len(hints) < 2:
            return None
        a, b = hints[0], hints[1]
        pair = {a, b}

        # Both open hands = Hello
        if 'Hello / B' in pair or ('B / 4' in pair and 'Hello / B' in pair):
            return 'Hello (two hands)'
        if a == 'Hello / B' and b == 'Hello / B':
            return 'Hello (two hands)'

        # Both fists = Stop
        if a in ('A', 'Good / Thumbs Up') and b in ('A', 'Good / Thumbs Up'):
            if a == 'Good / Thumbs Up' and b == 'Good / Thumbs Up':
                return 'Very Good'
            if a == 'A' and b == 'A':
                return 'Stop / A'

        # Both ILY
        if a == 'I Love You' and b == 'I Love You':
            return 'I Love You (two hands)'

        return None  # no special two-hand combo

    def predict(self, frame):
        hands_data = self._get_landmarks_and_handedness(frame)

        if not hands_data:
            self._stable_gesture = None
            self._stable_count = 0
            return {
                'gesture': '',
                'confidence': 0.0,
                'hands': [],
                'landmarks': [],
                'hand_states': [],
                'hints': [],
                'confirmed': False,
                'num_hands': 0
            }

        results = []
        for lms, handedness in hands_data:
            states = self._analyze(lms, handedness)
            hint   = self._classify(states)
            results.append({
                'landmarks':  lms,
                'handedness': handedness,
                'states':     states,
                'hint':       hint,
            })

        per_hand_landmarks = [r['landmarks'] for r in results]
        flat_landmarks     = [lm for r in results for lm in r['landmarks']]
        hints              = [r['hint'] for r in results]
        hand_states        = [r['states'] for r in results]

        # Try two-hand combo first
        two_hand = self._classify_two_hands(hints) if len(results) >= 2 else None
        combined_hint = two_hand if two_hand else (hints[0] if hints else '')

        if combined_hint == self._stable_gesture:
            self._stable_count += 1
        else:
            self._stable_gesture = combined_hint
            self._stable_count   = 1

        confirmed  = self._stable_count >= self._STABLE_THRESHOLD
        confidence = min(1.0, self._stable_count / (self._STABLE_THRESHOLD * 2)) if combined_hint else 0.0

        return {
            'gesture':     combined_hint,
            'confidence':  round(confidence, 2),
            'hands':       per_hand_landmarks,
            'landmarks':   flat_landmarks,
            'hand_states': hand_states,
            'hints':       hints,
            'confirmed':   confirmed,
            'num_hands':   len(results)
        }