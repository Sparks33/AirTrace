import cv2
import mediapipe as mp
import math
import numpy as np
import time
from collections import deque

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# ============================================================
# HARDWARE CONFIG - Programmed by Sparks33
# ============================================================
SERIAL_PORT = 'COM5'          # Windows e.g. 'COM5'  |  Mac e.g. '/dev/cu.usbmodem14101' or '/dev/cu.usbserial-XXXX'
SERIAL_BAUD = 115200            # must match Serial.begin(...) in the Arduino sketch

# Flip this to False once your Arduino/robot is actually wired up and plugged
# in. While True, "SEND" never touches the serial port at all - it just
# builds the G-code, previews it, prints/animates it in the console, and
# saves it to GCODE_OUTPUT_FILE. You can also toggle this live by pressing
# the 's' key while the app is running.
SIMULATION = True

GCODE_OUTPUT_FILE = 'drawing.gcode'

# Physical size of the area your plotter can actually reach, in millimeters.
# This is NOT your paper size necessarily - it's how far the pen carriage can
# physically travel on your frame. Measure your build's real X and Y travel
# with a ruler and put those numbers here.
PAPER_WIDTH_MM = 180.0
PAPER_HEIGHT_MM = 130.0

# Keep every drawn point at least this far from the physical edges of the
# plotter's travel, so nothing tries to drive the carriage past its frame.
MACHINE_MARGIN_MM = 5.0

# ---- G-code / motion tuning ----
DRAW_FEED_RATE = 800      # mm/min while pen is down (G1 moves)
TRAVEL_FEED_RATE = 3000   # mm/min while pen is up (G0 moves)
PATH_SIMPLIFY_EPSILON_MM = 0.4   # Douglas-Peucker tolerance, in mm
PATH_MIN_SPACING_MM = 1.5        # drop points closer together than this, in mm
HATCH_SPACING_MM = 2.5           # spacing between fill lines for filled shapes

# ============================================================
# STYLE / CONFIG
# ============================================================
DRAW_COLOR = (0, 0, 0)
PEN_THICKNESS = 8
SHAPE_THICKNESS = 4
PREVIEW_THICKNESS = 2
FILL_ALPHA = 0.35

PANEL_COLOR = (40, 40, 46)
PANEL_ALPHA = 0.55
ACCENT = (255, 140, 20)
DANGER = (70, 70, 230)
SUCCESS = (90, 200, 90)
ICON_COLOR = (225, 225, 230)
ICON_ACTIVE_COLOR = (255, 255, 255)
TITLE_COLOR = (255, 255, 255)
SUBTITLE_COLOR = (175, 175, 185)

BTN_SIZE = 78
BTN_GAP = 26
DIVIDER_GAP = 40
TOPBAR_H = 92
SIDEBAR_LEFT_MARGIN = 32
SIDEBAR_INNER_PAD = 16
SIDEBAR_BOTTOM_MARGIN = 26
HIT_PAD = 14

PINCH_ENTER = 55
PINCH_EXIT = 75

ERASER_MIN = 12
ERASER_MAX = 70
ERASER_RADIUS = 32

GRACE_FRAMES = 6
MAX_INTERP_POINTS = 18
SIMPLIFY_EPSILON = 2.5

DEBUG = False

# ============================================================
# MEDIAPIPE SETUP
# ============================================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.4,
    model_complexity=0
)
drawLandmark = mp.solutions.drawing_utils
landmark_spec = drawLandmark.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3)
connection_spec = drawLandmark.DrawingSpec(color=(180, 180, 180), thickness=2)

# ============================================================
# DRAWING STATE
# ============================================================
selected_tool = None
fill_mode = False
circle_data = []
line_data = []
re_data = []

pen_strokes = []
current_stroke = []
current_stroke_color = DRAW_COLOR

shape_start = None
shape_preview_end = None

was_pinching = False
frames_since_seen = 0
last_erase_pos = None
engaged_region = None
dis_history = deque(maxlen=4)
msg = 'Welcome to AirTracer'
status_overlay = None   # (text, color, expire_time) - shown briefly after actions like Send

# ============================================================
# GEOMETRY HELPERS
# ============================================================
def disx(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    return round(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), 3)

def point_seg_dist(pt, a, b):
    px, py = pt
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return disx(pt, a)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))
    return disx(pt, (ax + t * dx, ay + t * dy))

def interpolate_points(pt1, pt2, spacing=6, max_points=MAX_INTERP_POINTS):
    dist = disx(pt1, pt2)
    if dist <= spacing:
        return [pt2]
    steps = int(dist // spacing)
    if steps > max_points:
        steps = max_points
    pts = []
    for i in range(1, steps + 1):
        t = i / steps
        pts.append((int(pt1[0] + (pt2[0] - pt1[0]) * t),
                     int(pt1[1] + (pt2[1] - pt1[1]) * t)))
    return pts

def simplify_stroke(points, epsilon=SIMPLIFY_EPSILON):
    if len(points) < 3:
        return points
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    approx = cv2.approxPolyDP(pts, epsilon, False)
    return [tuple(int(v) for v in p[0]) for p in approx]

def point_in_circle(pt, center, r):
    return disx(pt, center) <= r

def point_in_rect(pt, pt1, pt2):
    x, y = pt
    x1, y1 = pt1
    x2, y2 = pt2
    return min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)

def try_bucket_fill(x, y):
    for data in reversed(re_data):
        if point_in_rect((x, y), data[0], data[1]):
            data[3] = True
            return True
    for data in reversed(circle_data):
        if point_in_circle((x, y), data[0], data[1]):
            data[3] = True
            return True
    return False

def erase_near(x, y, radius):
    """Erase pen strokes near (x,y), splitting strokes at the erased point."""
    global pen_strokes
    new_strokes = []
    for points, color in pen_strokes:
        segment = []
        for pt in points:
            if disx(pt, (x, y)) > radius:
                segment.append(pt)
            else:
                if len(segment) > 1:
                    new_strokes.append([segment, color])
                segment = []
        if len(segment) > 1:
            new_strokes.append([segment, color])
    pen_strokes = new_strokes

def erase_shapes_near(x, y, radius):
    """FIX: eraser now also removes circles/lines/rectangles it touches,
    not just freehand pen strokes."""
    global circle_data, line_data, re_data

    circle_data = [d for d in circle_data if disx((x, y), d[0]) > d[1] + radius]

    line_data = [d for d in line_data if point_seg_dist((x, y), d[0], d[1]) > radius]

    def rect_edge_dist(pt, pt1, pt2):
        x1, y1 = pt1
        x2, y2 = pt2
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        edges = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
        return min(point_seg_dist(pt, a, b) for a, b in edges)

    kept = []
    for d in re_data:
        near_edge = rect_edge_dist((x, y), d[0], d[1]) <= radius
        inside_filled = d[3] and point_in_rect((x, y), d[0], d[1])
        if not (near_edge or inside_filled):
            kept.append(d)
    re_data = kept

def CreateRect(frame, pt1, pt2, color, fill, thickness=SHAPE_THICKNESS):
    if fill:
        cv2.rectangle(frame, pt1, pt2, color, -1, cv2.LINE_AA)
    else:
        cv2.rectangle(frame, pt1, pt2, color, thickness, cv2.LINE_AA)

# ============================================================
# UI DRAWING HELPERS
# ============================================================
def rounded_rect(img, pt1, pt2, radius, color, thickness=-1):
    x1, y1 = pt1
    x2, y2 = pt2
    lt = cv2.LINE_AA
    if thickness < 0:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1, lt)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1, lt)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, -1, lt)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, -1, lt)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, -1, lt)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, -1, lt)
    else:
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness, lt)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness, lt)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness, lt)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness, lt)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, lt)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, lt)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, lt)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, lt)

def draw_icon(frame, name, cx, cy, s, color, thickness=3):
    r = s // 2
    lt = cv2.LINE_AA
    if name == 'pen':
        p1 = (cx - r + 10, cy + r - 8)
        p2 = (cx + r - 12, cy - r + 10)
        cv2.line(frame, p1, p2, color, thickness, lt)
        cv2.circle(frame, p1, 4, color, -1, lt)
        cv2.line(frame, p2, (p2[0] + 4, p2[1] - 4), color, thickness, lt)
    elif name == 'circle':
        cv2.circle(frame, (cx, cy), r - 10, color, thickness, lt)
    elif name == 'line':
        cv2.line(frame, (cx - r + 10, cy + r - 10), (cx + r - 10, cy - r + 10), color, thickness, lt)
    elif name == 'rect':
        cv2.rectangle(frame, (cx - r + 12, cy - r + 16), (cx + r - 12, cy + r - 16), color, thickness, lt)
    elif name == 'eraser':
        pts = np.array([
            [cx - r + 16, cy + 4], [cx - 4, cy - r + 14],
            [cx + r - 12, cy - 2], [cx + 2, cy + r - 12],
        ], np.int32)
        cv2.polylines(frame, [pts], True, color, thickness, lt)
    elif name == 'bucket':
        cv2.ellipse(frame, (cx, cy + 6), (r - 16, r - 20), 0, 0, 360, color, thickness, lt)
        tri = np.array([[cx, cy - r + 10], [cx - 9, cy - 4], [cx + 9, cy - 4]], np.int32)
        cv2.polylines(frame, [tri], True, color, thickness, lt)
    elif name == 'undo':
        cv2.ellipse(frame, (cx, cy), (r - 10, r - 10), 0, -50, 230, color, thickness, lt)
        ang = math.radians(-50)
        tipx = int(cx + (r - 10) * math.cos(ang))
        tipy = int(cy + (r - 10) * math.sin(ang))
        cv2.line(frame, (tipx, tipy), (tipx - 9, tipy - 4), color, thickness, lt)
        cv2.line(frame, (tipx, tipy), (tipx - 2, tipy - 11), color, thickness, lt)
    elif name == 'clear':
        cv2.line(frame, (cx - r + 12, cy - r + 12), (cx + r - 12, cy + r - 12), color, thickness, lt)
        cv2.line(frame, (cx - r + 12, cy + r - 12), (cx + r - 12, cy - r + 12), color, thickness, lt)
    elif name == 'check':
        cv2.line(frame, (cx - r + 16, cy + 2), (cx - 4, cy + r - 16), color, thickness + 1, lt)
        cv2.line(frame, (cx - 4, cy + r - 16), (cx + r - 12, cy - r + 14), color, thickness + 1, lt)

def draw_title(frame, text, org, scale, color, thickness=2):
    cv2.putText(frame, text, (org[0] + 2, org[1] + 2), cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, (org[0] + 1, org[1]), cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness, cv2.LINE_AA)

def draw_pen_strokes(frame):
    for points, color in pen_strokes:
        if len(points) > 1:
            pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=PEN_THICKNESS, lineType=cv2.LINE_AA)
    if len(current_stroke) > 1:
        pts = np.array(current_stroke, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=False, color=current_stroke_color, thickness=PEN_THICKNESS, lineType=cv2.LINE_AA)

# ============================================================
# CAMERA + LAYOUT
# ============================================================
cap = cv2.VideoCapture(0)
ok, first_frame = cap.read()
if not ok:
    raise SystemExit("Could not read from the camera.")
first_frame = cv2.flip(first_frame, 1)
_fh, _fw, _ = first_frame.shape
h = int(_fh * 1.4)
w = int(_fw * 1.4)

TOOLS = [
    ('pen',    'pen_tool',    'pen'),
    ('circle', 'circle_tool', 'circle'),
    ('line',   'line_tool',   'line'),
    ('rect',   'rect_tool',   'rect'),
    ('eraser', 'eraser_tool', 'eraser'),
    ('bucket', 'fill_toggle', None),
]
REGION_TO_TOOL = {region: tool for _icon, region, tool in TOOLS if tool is not None}

N_TOOLS = len(TOOLS)
available_h = h - TOPBAR_H - SIDEBAR_INNER_PAD - SIDEBAR_BOTTOM_MARGIN
fitted_size = (available_h - (N_TOOLS - 1) * BTN_GAP - DIVIDER_GAP) / (N_TOOLS + 1)
BTN_SIZE = int(max(52, min(BTN_SIZE, fitted_size)))

PANEL_INNER_W = 2 * BTN_SIZE + BTN_GAP
SIDEBAR_PANEL_X1 = SIDEBAR_LEFT_MARGIN
SIDEBAR_PANEL_X2 = SIDEBAR_LEFT_MARGIN + PANEL_INNER_W + 2 * SIDEBAR_INNER_PAD
COL_CX = SIDEBAR_PANEL_X1 + (SIDEBAR_PANEL_X2 - SIDEBAR_PANEL_X1) // 2

button_centers = {}
_y = TOPBAR_H + SIDEBAR_INNER_PAD + BTN_SIZE // 2
for _icon, region, _tool in TOOLS:
    button_centers[region] = (COL_CX, _y)
    _y += BTN_SIZE + BTN_GAP

DIVIDER_Y = (_y - BTN_GAP) + DIVIDER_GAP // 2
row_top = (_y - BTN_GAP) + DIVIDER_GAP
UNDO_CENTER = (COL_CX - (BTN_SIZE + BTN_GAP) // 2, row_top + BTN_SIZE // 2)
CLEAR_CENTER = (COL_CX + (BTN_SIZE + BTN_GAP) // 2, row_top + BTN_SIZE // 2)
_y = row_top + BTN_SIZE

SIDEBAR_BOTTOM = _y + SIDEBAR_INNER_PAD
SIDEBAR_PANEL = (4, TOPBAR_H + 4, SIDEBAR_PANEL_X2, SIDEBAR_BOTTOM)

# ---- SendToRobot button ----
CONFIRM_SIZE = 96
CONFIRM_CENTER = (w - CONFIRM_SIZE // 2 - 30, h - CONFIRM_SIZE // 2 - 30)

# ---- eraser size slider, top bar, right side ----
SLIDER_X1 = w - 280
SLIDER_X2 = w - 60
SLIDER_Y = TOPBAR_H // 2 + 8

# ---- canvas drawing area (used for the pixel -> mm mapping sent to the
# robot). Kept clear of the sidebar and bottom-right confirm button. ----
CANVAS_X1 = SIDEBAR_PANEL_X2 + 30
CANVAS_Y1 = TOPBAR_H + 20
CANVAS_X2 = w - 20
CANVAS_Y2 = h - CONFIRM_SIZE - 60

def slider_value_to_x(value):
    t = (value - ERASER_MIN) / (ERASER_MAX - ERASER_MIN)
    return int(SLIDER_X1 + t * (SLIDER_X2 - SLIDER_X1))

def slider_x_to_value(x):
    t = (x - SLIDER_X1) / (SLIDER_X2 - SLIDER_X1)
    t = max(0.0, min(1.0, t))
    return int(ERASER_MIN + t * (ERASER_MAX - ERASER_MIN))

def slider_hit(x, y):
    return (SLIDER_X1 - 15) <= x <= (SLIDER_X2 + 15) and abs(y - SLIDER_Y) <= 22

def draw_eraser_slider(frame, radius):
    # background pill so it's unmistakably visible against any camera feed
    cv2.rectangle(frame, (SLIDER_X1 - 20, SLIDER_Y - 34), (SLIDER_X2 + 20, SLIDER_Y + 18), (25, 25, 30), -1, cv2.LINE_AA)
    cv2.line(frame, (SLIDER_X1, SLIDER_Y), (SLIDER_X2, SLIDER_Y), (130, 130, 140), 5, cv2.LINE_AA)
    hx = slider_value_to_x(radius)
    cv2.circle(frame, (hx, SLIDER_Y), 13, ACCENT, -1, cv2.LINE_AA)
    cv2.circle(frame, (hx, SLIDER_Y), 13, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Eraser size: {radius}', (SLIDER_X1, SLIDER_Y - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 235), 1, cv2.LINE_AA)

def get_region(x, y):
    half = BTN_SIZE // 2 + HIT_PAD
    for _icon, region, _tool in TOOLS:
        cx, cy = button_centers[region]
        if cx - half <= x <= cx + half and cy - half <= y <= cy + half:
            return region
    ucx, ucy = UNDO_CENTER
    if ucx - half <= x <= ucx + half and ucy - half <= y <= ucy + half:
        return 'undo'
    ccx, ccy = CLEAR_CENTER
    if ccx - half <= x <= ccx + half and ccy - half <= y <= ccy + half:
        return 'clear_all'
    chalf = CONFIRM_SIZE // 2 + HIT_PAD
    fcx, fcy = CONFIRM_CENTER
    if fcx - chalf <= x <= fcx + chalf and fcy - chalf <= y <= fcy + chalf:
        return 'confirm_send'
    return None

# ============================================================
# COORDINATE MAPPING: camera pixels -> physical mm on the paper
# ============================================================
def get_mm_per_px():
    """How many millimeters of paper one canvas pixel represents, preserving
    aspect ratio so a circle drawn on screen stays a circle on paper."""
    cw = CANVAS_X2 - CANVAS_X1
    ch = CANVAS_Y2 - CANVAS_Y1
    return min(PAPER_WIDTH_MM / cw, PAPER_HEIGHT_MM / ch)

def to_paper_mm(px, py):
    cw = CANVAS_X2 - CANVAS_X1
    ch = CANVAS_Y2 - CANVAS_Y1
    scale = get_mm_per_px()
    draw_w = cw * scale
    draw_h = ch * scale
    offset_x = (PAPER_WIDTH_MM - draw_w) / 2
    offset_y = (PAPER_HEIGHT_MM - draw_h) / 2
    x_mm = offset_x + (px - CANVAS_X1) * scale
    y_mm = offset_y + (py - CANVAS_Y1) * scale
    return round(x_mm, 2), round(y_mm, 2)

# ---- bigger circles get more segments so
# they don't look faceted, small ones don't waste points ----
def adaptive_circle_segments(r_px):
    return int(np.clip(r_px * 0.9, 24, 180))

def circle_outline_points(center, r):
    n = adaptive_circle_segments(r)
    return [(int(center[0] + r * math.cos(2 * math.pi * i / n)),
              int(center[1] + r * math.sin(2 * math.pi * i / n))) for i in range(n + 1)]

# ---- hatch fill: zig-zag scanline infill so filled shapes are actually
# filled on paper, not just outlined ----
def hatch_fill_circle(center, r, spacing_px):
    lines = []
    cx, cy = center
    y = cy - r
    flip = False
    while y <= cy + r:
        dy = y - cy
        if abs(dy) <= r:
            dx = math.sqrt(max(r * r - dy * dy, 0))
            x1p, x2p = cx - dx, cx + dx
            a, b = (x2p, y), (x1p, y)
            if not flip:
                a, b = (x1p, y), (x2p, y)
            lines.append([(int(a[0]), int(a[1])), (int(b[0]), int(b[1]))])
            flip = not flip
        y += spacing_px
    return lines

def hatch_fill_rect(pt1, pt2, spacing_px):
    x1, y1 = pt1
    x2, y2 = pt2
    xlo, xhi = min(x1, x2), max(x1, x2)
    ylo, yhi = min(y1, y2), max(y1, y2)
    lines = []
    y = ylo
    flip = False
    while y <= yhi:
        a, b = (xhi, y), (xlo, y)
        if not flip:
            a, b = (xlo, y), (xhi, y)
        lines.append([(int(a[0]), int(a[1])), (int(b[0]), int(b[1]))])
        flip = not flip
        y += spacing_px
    return lines

def collect_all_paths():
    """Turns every stroke and shape currently on the canvas into a list of
    point-paths (each an ordered list of (x,y) pixel points, one continuous
    pen-down path per entry). Circles use adaptive tessellation, and filled
    shapes get real hatch-fill paths added on top of their outline."""
    paths = []
    for points, _color in pen_strokes:
        if len(points) > 1:
            paths.append(points)

    mm_per_px = get_mm_per_px()
    hatch_spacing_px = max(2.0, HATCH_SPACING_MM / mm_per_px)

    for center, r, _color, filled in circle_data:
        paths.append(circle_outline_points(center, r))
        if filled:
            paths.extend(hatch_fill_circle(center, r, hatch_spacing_px))

    for pt1, pt2, _color in line_data:
        paths.append([pt1, pt2])

    for pt1, pt2, _color, filled in re_data:
        x1, y1 = pt1
        x2, y2 = pt2
        paths.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)])
        if filled:
            paths.extend(hatch_fill_rect(pt1, pt2, hatch_spacing_px))
    return paths

# ============================================================
# GEOMETRY PIPELINE (mm-space): simplify -> resample -> clip -> optimize
# ============================================================
def simplify_mm_path(points_mm, epsilon_mm=PATH_SIMPLIFY_EPSILON_MM):
    """Douglas-Peucker simplification to drop redundant points."""
    if len(points_mm) < 3:
        return points_mm
    pts = np.array(points_mm, dtype=np.float32).reshape((-1, 1, 2))
    approx = cv2.approxPolyDP(pts, epsilon_mm, False)
    return [(float(p[0][0]), float(p[0][1])) for p in approx]

def resample_min_spacing(points_mm, spacing_mm=PATH_MIN_SPACING_MM):
    """Drop points that are closer together than spacing_mm, always keeping
    the first and last point of the path."""
    if len(points_mm) < 2:
        return points_mm
    out = [points_mm[0]]
    for p in points_mm[1:-1]:
        if disx(out[-1], p) >= spacing_mm:
            out.append(p)
    if disx(out[-1], points_mm[-1]) > 1e-6:
        out.append(points_mm[-1])
    return out

def clip_segment_to_rect(p1, p2, xmin, ymin, xmax, ymax):
    """Liang-Barsky segment clipping against an axis-aligned rectangle."""
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    p = (-dx, dx, -dy, dy)
    q = (x1 - xmin, xmax - x1, y1 - ymin, ymax - y1)
    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return None
        else:
            t = qi / pi
            if pi < 0:
                if t > u2:
                    return None
                if t > u1:
                    u1 = t
            else:
                if t < u1:
                    return None
                if t < u2:
                    u2 = t
    if u1 > u2:
        return None
    return (x1 + u1 * dx, y1 + u1 * dy), (x1 + u2 * dx, y1 + u2 * dy)

def clip_path_to_paper(points_mm):
    """Clips a path against the plotter's travel (minus margin). Because
    clipping can chop a path into disconnected pieces, this returns a LIST
    of paths, not a single one."""
    xmin = ymin = MACHINE_MARGIN_MM
    xmax = PAPER_WIDTH_MM - MACHINE_MARGIN_MM
    ymax = PAPER_HEIGHT_MM - MACHINE_MARGIN_MM
    result = []
    current = []
    for i in range(len(points_mm) - 1):
        seg = clip_segment_to_rect(points_mm[i], points_mm[i + 1], xmin, ymin, xmax, ymax)
        if seg is None:
            if len(current) > 1:
                result.append(current)
            current = []
            continue
        a, b = seg
        if not current:
            current = [a, b]
        elif disx(current[-1], a) < 1e-6:
            current.append(b)
        else:
            if len(current) > 1:
                result.append(current)
            current = [a, b]
    if len(current) > 1:
        result.append(current)
    return result

def optimize_path_order(paths_mm):
    """Greedy nearest-neighbor reordering (and reversal) of paths to cut
    down on total pen-up travel between disconnected strokes/shapes."""
    if not paths_mm:
        return []
    remaining = paths_mm[:]
    ordered = [remaining.pop(0)]
    while remaining:
        last_pt = ordered[-1][-1]
        best_i, best_d, best_rev = 0, float('inf'), False
        for i, path in enumerate(remaining):
            d_start = disx(last_pt, path[0])
            d_end = disx(last_pt, path[-1])
            if d_start < best_d:
                best_d, best_i, best_rev = d_start, i, False
            if d_end < best_d:
                best_d, best_i, best_rev = d_end, i, True
        nxt = remaining.pop(best_i)
        if best_rev:
            nxt = list(reversed(nxt))
        ordered.append(nxt)
    return ordered

def build_processed_paths_mm():
    """Full pipeline: pixel paths -> mm -> simplify -> resample -> clip ->
    optimize order. Returns a list of mm-space paths, or [] if nothing to
    draw."""
    raw_paths_px = collect_all_paths()
    if not raw_paths_px:
        return []
    paths_mm = []
    for path_px in raw_paths_px:
        path_mm = [to_paper_mm(*pt) for pt in path_px]
        path_mm = simplify_mm_path(path_mm)
        path_mm = resample_min_spacing(path_mm)
        for clipped in clip_path_to_paper(path_mm):
            if len(clipped) > 1:
                paths_mm.append(clipped)
    return optimize_path_order(paths_mm)

# ============================================================
# G-CODE GENERATION / PREVIEW / SAVE
# ============================================================
def generate_gcode(paths_mm):
    lines = ["G21", "G90", "M5"]
    for path in paths_mm:
        if len(path) < 2:
            continue
        x0, y0 = path[0]
        lines.append(f"G0 F{TRAVEL_FEED_RATE} X{x0:.2f} Y{y0:.2f}")
        lines.append("M3")
        lines.append(f"G1 F{DRAW_FEED_RATE}")
        for p in path[1:]:
            lines.append(f"G1 X{p[0]:.2f} Y{p[1]:.2f}")
        lines.append("M5")
    return lines

def save_gcode_file(gcode_lines, filename=GCODE_OUTPUT_FILE):
    with open(filename, 'w') as f:
        f.write('\n'.join(gcode_lines) + '\n')

def show_gcode_preview(paths_mm):
    """Renders exactly what the robot will draw, at true relative scale,
    on a plain white 'paper' canvas. Blocks until a key is pressed."""
    scale = 4  # preview pixels per mm
    margin = 24
    img_w = int(PAPER_WIDTH_MM * scale) + margin * 2
    img_h = int(PAPER_HEIGHT_MM * scale) + margin * 2 + 30
    canvas = np.full((img_h, img_w, 3), 255, np.uint8)
    px1, py1 = margin, margin
    px2, py2 = margin + int(PAPER_WIDTH_MM * scale), margin + int(PAPER_HEIGHT_MM * scale)
    cv2.rectangle(canvas, (px1, py1), (px2, py2), (200, 200, 200), 2, cv2.LINE_AA)
    for path in paths_mm:
        pts = [(int(margin + x * scale), int(margin + y * scale)) for x, y in path]
        for i in range(len(pts) - 1):
            cv2.line(canvas, pts[i], pts[i + 1], (20, 20, 20), 2, cv2.LINE_AA)
    label = f"G-code Preview - {len(paths_mm)} path(s) - press any key to continue"
    cv2.putText(canvas, label, (margin, img_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow("AirTracer - G-code Preview", canvas)
    cv2.waitKey(0)
    cv2.destroyWindow("AirTracer - G-code Preview")

# ============================================================
# SERIAL LINK TO ARDUINO
# ============================================================
_ser = None

def get_serial():
    global _ser
    if not SERIAL_AVAILABLE:
        return None
    if _ser is None or not _ser.is_open:
        _ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=5)
        time.sleep(2)  # Arduino resets on serial connect - give it time to boot
        _ser.reset_input_buffer()
    return _ser

def send_line(ser, text):
    ser.write((text + '\n').encode('utf-8'))
    # wait for the Arduino's "OK" handshake before sending the next line -
    # without this, commands would pile up in its tiny serial buffer while
    # it's still physically moving the motors
    start = time.time()
    while time.time() - start < 15:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line == 'OK':
            return True
        if line:
            print(f"[arduino] {line}")
    print(f"[warn] no OK received for: {text}")
    return False

def simulate_send(gcode_lines):
    """No hardware needed at all - just prints each G-code line as if it
    were being streamed to the Arduino, with a small delay so it reads like
    an actual transmission."""
    print("=== SIMULATION MODE - no Arduino connected, nothing sent over serial ===")
    total = len(gcode_lines)
    for i, line in enumerate(gcode_lines):
        print(f"  [{i + 1}/{total}] -> {line}")
        time.sleep(0.03)
    print(f"=== Simulation complete - {total} line(s), also saved to {GCODE_OUTPUT_FILE} ===")

def send_gcode_serial(ser, gcode_lines):
    total = len(gcode_lines)
    for i, line in enumerate(gcode_lines):
        print(f"  [{i + 1}/{total}] sending: {line}")
        send_line(ser, line)
    print("Done sending.")

def send_drawing_to_robot():
    global status_overlay
    paths_mm = build_processed_paths_mm()
    if not paths_mm:
        status_overlay = ("Nothing to send - draw something first", DANGER, time.time() + 2.5)
        return

    gcode_lines = generate_gcode(paths_mm)
    save_gcode_file(gcode_lines)
    show_gcode_preview(paths_mm)

    if SIMULATION:
        simulate_send(gcode_lines)
        status_overlay = (f"SIMULATED {len(paths_mm)} path(s) - saved {GCODE_OUTPUT_FILE}", SUCCESS, time.time() + 3.5)
        return

    try:
        ser = get_serial()
        if ser is None:
            status_overlay = ("pyserial not installed - run: pip install pyserial", DANGER, time.time() + 4)
            return
    except Exception as e:
        status_overlay = (f"Serial error: {e}", DANGER, time.time() + 4)
        return

    print(f"Sending {len(gcode_lines)} G-code line(s) to the plotter...")
    send_gcode_serial(ser, gcode_lines)
    status_overlay = (f"Sent {len(paths_mm)} path(s) to the plotter", SUCCESS, time.time() + 3)

# ============================================================
# MAIN LOOP
# ============================================================
pending_frame = first_frame
while cap.isOpened():
    if pending_frame is not None:
        frame = pending_frame
        pending_frame = None
    else:
        stat, frame = cap.read()
        if not stat:
            print("Error: Couldn't read frame.")
            break
        frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (w, h))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, TOPBAR_H), PANEL_COLOR, -1)
    rounded_rect(overlay, (SIDEBAR_PANEL[0], SIDEBAR_PANEL[1]), (SIDEBAR_PANEL[2], SIDEBAR_PANEL[3]), 22, PANEL_COLOR, -1)
    frame = cv2.addWeighted(overlay, PANEL_ALPHA, frame, 1 - PANEL_ALPHA, 0)

    draw_title(frame, "AirTracer", (24, 44), 1.05, TITLE_COLOR, 2)
    status = (selected_tool or 'none').upper() + ("  \u2022  FILL ON" if fill_mode else "")
    cv2.putText(frame, status, (26, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.55, SUBTITLE_COLOR, 1, cv2.LINE_AA)

    # ---- SIM / LIVE badge, top-right, and its 's' key hint ----
    mode_text = "SIM MODE" if SIMULATION else "LIVE MODE"
    mode_color = ACCENT if SIMULATION else SUCCESS
    (mtw, mth), _ = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    mbx1, mby1 = w - mtw - 40, 14
    mbx2, mby2 = w - 14, 14 + mth + 16
    rounded_rect(frame, (mbx1, mby1), (mbx2, mby2), 10, (30, 30, 34), -1)
    cv2.putText(frame, mode_text, (mbx1 + 12, mby2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 2, cv2.LINE_AA)
    cv2.putText(frame, "press 's' to toggle", (mbx1, mby2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, SUBTITLE_COLOR, 1, cv2.LINE_AA)

    if selected_tool == 'eraser':
        draw_eraser_slider(frame, ERASER_RADIUS)

    for icon_name, region, tool_value in TOOLS:
        cx, cy = button_centers[region]
        active = fill_mode if region == 'fill_toggle' else (selected_tool == tool_value)
        if active:
            x1, y1 = cx - BTN_SIZE // 2, cy - BTN_SIZE // 2
            x2, y2 = cx + BTN_SIZE // 2, cy + BTN_SIZE // 2
            rounded_rect(frame, (x1, y1), (x2, y2), 18, ACCENT, -1)
        icon_color = ICON_ACTIVE_COLOR if active else ICON_COLOR
        draw_icon(frame, icon_name, cx, cy, BTN_SIZE - 16, icon_color, 3)

    cv2.line(frame, (SIDEBAR_PANEL[0] + 14, DIVIDER_Y), (SIDEBAR_PANEL[2] - 14, DIVIDER_Y), (95, 95, 102), 1, cv2.LINE_AA)

    half = BTN_SIZE // 2
    ux1, uy1 = UNDO_CENTER[0] - half, UNDO_CENTER[1] - half
    ux2, uy2 = UNDO_CENTER[0] + half, UNDO_CENTER[1] + half
    rounded_rect(frame, (ux1, uy1), (ux2, uy2), 16, (60, 60, 68), -1)
    draw_icon(frame, 'undo', UNDO_CENTER[0], UNDO_CENTER[1], BTN_SIZE - 16, ICON_COLOR, 3)

    cx1, cy1 = CLEAR_CENTER[0] - half, CLEAR_CENTER[1] - half
    cx2, cy2 = CLEAR_CENTER[0] + half, CLEAR_CENTER[1] + half
    rounded_rect(frame, (cx1, cy1), (cx2, cy2), 16, DANGER, -1)
    draw_icon(frame, 'clear', CLEAR_CENTER[0], CLEAR_CENTER[1], BTN_SIZE - 18, (255, 255, 255), 3)

    # ---- SEND TO ROBOT button, bottom-right ----
    chalf = CONFIRM_SIZE // 2
    fcx, fcy = CONFIRM_CENTER
    rounded_rect(frame, (fcx - chalf, fcy - chalf), (fcx + chalf, fcy + chalf), 22, SUCCESS, -1)
    draw_icon(frame, 'check', fcx, fcy, CONFIRM_SIZE - 20, (255, 255, 255), 4)
    cv2.putText(frame, "SEND", (fcx - 28, fcy + chalf + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    if any(d[3] for d in circle_data) or any(d[3] for d in re_data):
        fill_overlay = frame.copy()
        for data in circle_data:
            if data[3]:
                cv2.circle(fill_overlay, data[0], data[1], data[2], -1, cv2.LINE_AA)
        for data in re_data:
            if data[3]:
                CreateRect(fill_overlay, data[0], data[1], data[2], True)
        frame = cv2.addWeighted(fill_overlay, FILL_ALPHA, frame, 1 - FILL_ALPHA, 0)

    for data in circle_data:
        cv2.circle(frame, data[0], data[1], data[2], SHAPE_THICKNESS, cv2.LINE_AA)
    for data in line_data:
        cv2.line(frame, data[0], data[1], data[2], SHAPE_THICKNESS, cv2.LINE_AA)
    for data in re_data:
        CreateRect(frame, data[0], data[1], data[2], False)

    if shape_start is not None and shape_preview_end is not None:
        if selected_tool == 'circle':
            r = int(disx(shape_start, shape_preview_end))
            cv2.circle(frame, shape_start, r, DRAW_COLOR, PREVIEW_THICKNESS, cv2.LINE_AA)
        elif selected_tool == 'line':
            cv2.line(frame, shape_start, shape_preview_end, DRAW_COLOR, PREVIEW_THICKNESS, cv2.LINE_AA)
        elif selected_tool == 'rect':
            cv2.rectangle(frame, shape_start, shape_preview_end, DRAW_COLOR, PREVIEW_THICKNESS, cv2.LINE_AA)

    draw_pen_strokes(frame)

    if status_overlay is not None:
        text, color, expire = status_overlay
        if time.time() < expire:
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            bx, by = w // 2 - tw // 2 - 20, 20
            cv2.rectangle(frame, (bx, by), (bx + tw + 40, by + th + 30), (20, 20, 24), -1, cv2.LINE_AA)
            cv2.rectangle(frame, (bx, by), (bx + tw + 40, by + th + 30), color, 2, cv2.LINE_AA)
            cv2.putText(frame, text, (bx + 20, by + th + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            status_overlay = None

    # ---- hand tracking / gestures ----
    if results.multi_hand_landmarks:
        frames_since_seen = 0
        for hand_landmarks in results.multi_hand_landmarks:
            points = []
            drawLandmark.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                         landmark_drawing_spec=landmark_spec,
                                         connection_drawing_spec=connection_spec)
            for idx, landmark in enumerate(hand_landmarks.landmark):
                if idx == 4:
                    cx4, cy4 = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx4, cy4), 6, (255, 0, 0), -1)
                    points.append((cx4, cy4))
                if idx == 8:
                    cx8, cy8 = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx8, cy8), 6, (255, 0, 0), -1)
                    points.append((cx8, cy8))

            if len(points) == 2:
                cv2.line(frame, points[0], points[1], (0, 255, 0), 2)
                midpoint = ((points[0][0] + points[1][0]) // 2, (points[0][1] + points[1][1]) // 2)
                cv2.circle(frame, midpoint, 6, (0, 0, 180), -1)
                dis_raw = disx(points[0], midpoint)
                dis_history.append(dis_raw)
                dis = sum(dis_history) / len(dis_history)
                if DEBUG:
                    print(f"pinch_dist={dis:.1f} (raw={dis_raw:.1f})  midpoint={midpoint}  tool={selected_tool}")

                x, y = midpoint
                pinch_now = (dis < PINCH_EXIT) if was_pinching else (dis < PINCH_ENTER)
                pinch_down = pinch_now and not was_pinching
                pinch_up = (not pinch_now) and was_pinching
                region = get_region(x, y)

                if pinch_up:
                    if selected_tool == 'pen' and len(current_stroke) > 1:
                        pen_strokes.append([simplify_stroke(current_stroke), current_stroke_color])
                    current_stroke = []
                    if selected_tool in ('circle', 'line', 'rect') and shape_start is not None:
                        pt1, pt2 = shape_start, (x, y)
                        if selected_tool == 'circle':
                            r = int(disx(pt1, pt2))
                            circle_data.append([pt1, r, DRAW_COLOR, False])
                        elif selected_tool == 'line':
                            line_data.append([pt1, pt2, DRAW_COLOR])
                        elif selected_tool == 'rect':
                            re_data.append([pt1, pt2, DRAW_COLOR, False])
                        shape_start = None
                        shape_preview_end = None
                    engaged_region = None
                    last_erase_pos = None

                if pinch_now:
                    if region is not None:
                        if region != engaged_region:
                            engaged_region = region
                            if region == 'undo':
                                msg = 'Undo'
                                if selected_tool == 'circle' and circle_data:
                                    circle_data.pop()
                                elif selected_tool == 'rect' and re_data:
                                    re_data.pop()
                                elif selected_tool == 'line' and line_data:
                                    line_data.pop()
                                elif selected_tool == 'pen' and pen_strokes:
                                    pen_strokes.pop()
                            elif region == 'clear_all':
                                circle_data = []
                                line_data = []
                                re_data = []
                                pen_strokes = []
                                msg = 'Canvas cleared'
                            elif region == 'fill_toggle':
                                fill_mode = not fill_mode
                            elif region == 'confirm_send':
                                send_drawing_to_robot()
                            else:
                                tool_value = REGION_TO_TOOL[region]
                                selected_tool = None if selected_tool == tool_value else tool_value
                                fill_mode = False
                            current_stroke = []
                            shape_start = None
                            shape_preview_end = None
                    else:
                        engaged_region = None

                        if fill_mode:
                            if pinch_down:
                                msg = 'Shape filled' if try_bucket_fill(x, y) else 'No shape here to fill'

                        elif selected_tool == 'pen':
                            if pinch_down:
                                current_stroke = [(x, y)]
                                current_stroke_color = DRAW_COLOR
                            else:
                                if current_stroke:
                                    current_stroke.extend(interpolate_points(current_stroke[-1], (x, y)))
                                else:
                                    current_stroke.append((x, y))

                        elif selected_tool == 'eraser':
                            if slider_hit(x, y):
                                ERASER_RADIUS = slider_x_to_value(x)
                                last_erase_pos = None
                            else:
                                if last_erase_pos is not None:
                                    for pt in interpolate_points(last_erase_pos, (x, y), spacing=ERASER_RADIUS / 2):
                                        erase_near(pt[0], pt[1], ERASER_RADIUS)
                                        erase_shapes_near(pt[0], pt[1], ERASER_RADIUS)
                                erase_near(x, y, ERASER_RADIUS)
                                erase_shapes_near(x, y, ERASER_RADIUS)
                                last_erase_pos = (x, y)

                        elif selected_tool in ('circle', 'line', 'rect'):
                            if pinch_down:
                                shape_start = (x, y)
                                shape_preview_end = (x, y)
                            elif shape_start is not None:
                                shape_preview_end = (x, y)
                else:
                    engaged_region = None

                was_pinching = pinch_now
    else:
        frames_since_seen += 1
        last_erase_pos = None
        if frames_since_seen > GRACE_FRAMES:
            if selected_tool == 'pen' and len(current_stroke) > 1:
                pen_strokes.append([simplify_stroke(current_stroke), current_stroke_color])
            was_pinching = False
            engaged_region = None
            current_stroke = []
            shape_start = None
            shape_preview_end = None

    cv2.imshow('AirTracer', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        SIMULATION = not SIMULATION
        status_overlay = ("SIMULATION MODE ON" if SIMULATION else "LIVE MODE - will use serial port", ACCENT if SIMULATION else SUCCESS, time.time() + 2)

cap.release()
cv2.destroyAllWindows()
if _ser is not None and _ser.is_open:
    _ser.close()
