import os

# =========================================================
# ĐƯỜNG DẪN DỮ LIỆU TRÊN Ổ D
# =========================================================

YOLO_DIR = r"D:\YOLO"
YOLO_MODEL_PATH = os.path.join(YOLO_DIR, "yolo11n.pt")
YOLO_CONFIG_DIR = os.path.join(YOLO_DIR, "config")
TEMP_SLEEP_IMAGE = os.path.join(YOLO_DIR, "temp_sleep.jpg")

os.makedirs(YOLO_DIR, exist_ok=True)
os.makedirs(YOLO_CONFIG_DIR, exist_ok=True)

# Đưa cấu hình Ultralytics sang D
os.environ["YOLO_CONFIG_DIR"] = YOLO_CONFIG_DIR

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# =========================================================
# IMPORT
# =========================================================

import cv2
import mediapipe as mp
import requests
import math
import time
import pyttsx3

from concurrent.futures import ThreadPoolExecutor

from inference_sdk import InferenceHTTPClient

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ultralytics import YOLO


# =========================================================
# BIẾN CHUNG
# =========================================================

sleep_start_time = None
sleep_count = 0
eye_closed_start = None
yawn_detected = False
frame_count = 0


# =========================================================
# MEDIAPIPE FACE LANDMARKER
# =========================================================

base_options = python.BaseOptions(
    model_asset_path=r"D:\test.py\face_landmarker.task"
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    running_mode=vision.RunningMode.VIDEO
)

face_landmarker = vision.FaceLandmarker.create_from_options(
    options
)


# =========================================================
# TELEGRAM
# =========================================================

TELEGRAM_TOKEN = "TOKEN_MOI_CUA_BAN"
CHAT_ID = "8624123792"


def send_telegram_alert(message):

    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_TOKEN}/sendMessage"
    )

    data = {
        "chat_id": CHAT_ID,
        "text": message
    }

    try:

        r = requests.post(
            url,
            data=data,
            timeout=5
        )

        print("Telegram:", r.status_code)

    except Exception as e:

        print("Telegram error:", e)


# =========================================================
# GIỌNG NÓI
# =========================================================

def get_vietnamese_voice(engine):

    voices = engine.getProperty("voices")

    for v in voices:

        if (
            "Vietnam" in v.name
            or "Vietnamese" in v.name
            or "vi" in v.id.lower()
        ):

            return v.id

    return None


def voice_alert(message):

    try:

        engine = pyttsx3.init()

        engine.setProperty(
            "rate",
            170
        )

        engine.setProperty(
            "volume",
            1.0
        )

        vi_voice = get_vietnamese_voice(engine)

        if vi_voice:

            engine.setProperty(
                "voice",
                vi_voice
            )

        engine.say(message)

        engine.runAndWait()

        engine.stop()

    except Exception as e:

        print("Voice error:", e)


# =========================================================
# ROBOFLOW
# CHỈ DÙNG CHO SLEEP
# =========================================================

client = InferenceHTTPClient(

    api_url="https://detect.roboflow.com",

    api_key="6VQqcqCzr7wPXHVPk4us"
)


def safe_run_workflow(
    workflow_id,
    image_path
):

    try:

        return client.run_workflow(

            workspace_name="nguyen-ho",

            workflow_id=workflow_id,

            images={
                "image": image_path
            },

            use_cache=False
        )

    except Exception as e:

        print(
            f"❌ {workflow_id} lỗi:",
            e
        )

        return {}


# =========================================================
# YOLO LOCAL - NHẬN DIỆN ĐIỆN THOẠI
# =========================================================

print("⏳ Đang kiểm tra YOLO...")


# Kiểm tra thư mục
if not os.path.isdir(YOLO_DIR):

    os.makedirs(
        YOLO_DIR,
        exist_ok=True
    )


# Kiểm tra model
if not os.path.isfile(YOLO_MODEL_PATH):

    print()
    print("❌ KHÔNG TÌM THẤY MODEL YOLO")
    print()
    print("Đường dẫn đang tìm:")
    print(YOLO_MODEL_PATH)
    print()
    print("Hãy đặt file:")
    print("yolo11n.pt")
    print("vào thư mục:")
    print(YOLO_DIR)
    print()

    raise FileNotFoundError(
        YOLO_MODEL_PATH
    )


print(
    "📁 YOLO model:",
    YOLO_MODEL_PATH
)


# =========================================================
# COCO
# 67 = cell phone
# =========================================================

PHONE_CLASS_ID = 67

PHONE_YOLO_CONF = 0.10
PHONE_YOLO_IMGSZ = 640


try:

    print("⏳ Đang tải YOLO từ ổ D...")

    phone_yolo = YOLO(
        YOLO_MODEL_PATH
    )

    print(
        "✅ YOLO điện thoại đã sẵn sàng"
    )

except Exception as e:

    print(
        "❌ Không tải được YOLO:",
        e
    )

    raise


# =========================================================
# THREAD
# =========================================================

phone_executor = ThreadPoolExecutor(
    max_workers=1
)

sleep_executor = ThreadPoolExecutor(
    max_workers=1
)

alert_executor = ThreadPoolExecutor(
    max_workers=2
)


phone_future = None
sleep_future = None


# =========================================================
# TỐC ĐỘ AI
# =========================================================

PHONE_AI_INTERVAL = 0.03

SLEEP_AI_INTERVAL = 2.0


last_phone_ai_time = 0
last_sleep_ai_time = 0


# =========================================================
# CHẠY YOLO PHONE
# =========================================================

def run_phone_detection(frame):
    try:
        h, w = frame.shape[:2]

        all_boxes = []

        # =====================================================
        # 1. YOLO QUÉT TOÀN BỘ ẢNH
        # =====================================================

        results_full = phone_yolo.predict(
            source=frame,
            imgsz=PHONE_YOLO_IMGSZ,
            conf=PHONE_YOLO_CONF,
            classes=[PHONE_CLASS_ID],
            device="cpu",
            verbose=False,
            max_det=10
        )

        if results_full:
            result = results_full[0]

            if result.boxes is not None:
                for box in result.boxes:
                    try:
                        coords = (
                            box.xyxy[0]
                            .cpu()
                            .numpy()
                        )

                        conf = float(
                            box.conf[0]
                            .cpu()
                            .item()
                        )

                        x1 = int(coords[0])
                        y1 = int(coords[1])
                        x2 = int(coords[2])
                        y2 = int(coords[3])

                        if x2 > x1 and y2 > y1:
                            all_boxes.append({
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2,
                                "conf": conf
                            })

                    except Exception:
                        pass

        # =====================================================
        # 2. CHIA ẢNH THÀNH 4 VÙNG
        # =====================================================
        #
        # Mục đích:
        # Nếu điện thoại nhỏ hoặc chỉ xuất hiện một phần,
        # YOLO sẽ nhìn gần hơn vào từng vùng.
        #

        overlap = 80

        crop_w = w // 2 + overlap
        crop_h = h // 2 + overlap

        crops = [
            (0, 0, crop_w, crop_h),

            (w // 2 - overlap, 0, w, crop_h),

            (0, h // 2 - overlap, crop_w, h),

            (w // 2 - overlap,
             h // 2 - overlap,
             w, h)
        ]

        # =====================================================
        # 3. YOLO QUÉT TỪNG VÙNG
        # =====================================================

        for cx1, cy1, cx2, cy2 in crops:

            cx1 = max(0, cx1)
            cy1 = max(0, cy1)
            cx2 = min(w, cx2)
            cy2 = min(h, cy2)

            crop = frame[
                cy1:cy2,
                cx1:cx2
            ]

            if crop.size == 0:
                continue

            crop_results = phone_yolo.predict(
                source=crop,
                imgsz=640,
                conf=PHONE_YOLO_CONF,
                classes=[PHONE_CLASS_ID],
                device="cpu",
                verbose=False,
                max_det=10
            )

            if not crop_results:
                continue

            result = crop_results[0]

            if result.boxes is None:
                continue

            for box in result.boxes:

                try:
                    coords = (
                        box.xyxy[0]
                        .cpu()
                        .numpy()
                    )

                    conf = float(
                        box.conf[0]
                        .cpu()
                        .item()
                    )

                    # Tọa độ trong crop
                    x1 = int(coords[0]) + cx1
                    y1 = int(coords[1]) + cy1
                    x2 = int(coords[2]) + cx1
                    y2 = int(coords[3]) + cy1

                    # Giới hạn trong webcam
                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(0, min(x2, w - 1))
                    y2 = max(0, min(y2, h - 1))

                    if x2 > x1 and y2 > y1:

                        all_boxes.append({
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "conf": conf
                        })

                except Exception:
                    pass

        # =====================================================
        # 4. TRẢ VỀ TẤT CẢ BOX
        # =====================================================

        return all_boxes

    except Exception as e:

        print(
            "❌ YOLO phone error:",
            e
        )

        return []


# =========================================================
# CHẠY ROBOFLOW SLEEP
# =========================================================

def run_sleep_detection(frame):

    try:

        small_frame = cv2.resize(
            frame,
            (320, 240)
        )


        cv2.imwrite(

            TEMP_SLEEP_IMAGE,

            small_frame,

            [
                cv2.IMWRITE_JPEG_QUALITY,
                75
            ]
        )


        result_sleep = safe_run_workflow(

            "find-slepping",

            TEMP_SLEEP_IMAGE
        )


        return result_sleep

    except Exception as e:

        print(
            "❌ Sleep background error:",
            e
        )

        return {}


# =========================================================
# LẤY PREDICTIONS ROBOFLOW
# =========================================================

def get_predictions(result):

    if not result:

        return []


    if isinstance(
        result,
        dict
    ):

        predictions = result.get(
            "predictions",
            []
        )


        if isinstance(
            predictions,
            list
        ):

            return predictions


        if isinstance(
            predictions,
            dict
        ):

            return predictions.get(
                "predictions",
                []
            )


    if isinstance(
        result,
        list
    ):

        all_predictions = []


        for item in result:

            if not isinstance(
                item,
                dict
            ):

                continue


            predictions = item.get(
                "predictions",
                []
            )


            if isinstance(
                predictions,
                list
            ):

                all_predictions.extend(
                    predictions
                )


            elif isinstance(
                predictions,
                dict
            ):

                nested = predictions.get(
                    "predictions",
                    []
                )


                if isinstance(
                    nested,
                    list
                ):

                    all_predictions.extend(
                        nested
                    )


        return all_predictions


    return []


# =========================================================
# NGƯỠNG
# =========================================================

EYE_AR_THRESH = 0.20

MOUTH_OPEN_THRESH = 0.04

SLEEP_CONF_THRESH = 0.60


# =========================================================
# ALERT
# =========================================================

last_alert_time = {

    "phone": 0,

    "slepping": 0,

    "eye": 0,

    "head": 0,

    "yawn": 0

}


ALERT_DELAY = 10


def handle_alert(
    alert_key,
    text,
    voice
):

    now = time.time()


    if (

        now
        -
        last_alert_time[alert_key]

        >

        ALERT_DELAY

    ):

        last_alert_time[alert_key] = now


        alert_executor.submit(

            voice_alert,

            voice

        )


        alert_executor.submit(

            send_telegram_alert,

            text

        )


# =========================================================
# PHONE TRACKING
# =========================================================

tracked_boxes = []

last_phone_box = None

last_phone_box_time = 0


PHONE_BOX_HOLD_TIME = 1.5

BOX_SMOOTHING = 0.25


# =========================================================
# OPTICAL FLOW
# =========================================================

phone_track_points = None

phone_prev_gray = None

phone_tracking_fail = 0

MAX_TRACKING_FAIL = 5


# =========================================================
# KHỞI TẠO TRACKING
# =========================================================

def init_phone_tracker(
    frame,
    box
):

    global phone_track_points
    global phone_prev_gray
    global phone_tracking_fail


    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    x1 = max(
        0,
        int(box["x1"])
    )


    y1 = max(
        0,
        int(box["y1"])
    )


    x2 = min(
        frame.shape[1] - 1,
        int(box["x2"])
    )


    y2 = min(
        frame.shape[0] - 1,
        int(box["y2"])
    )


    if x2 <= x1 or y2 <= y1:

        phone_track_points = None

        phone_prev_gray = gray

        return


    roi = gray[
        y1:y2,
        x1:x2
    ]


    if roi.size == 0:

        phone_track_points = None

        phone_prev_gray = gray

        return


    points = cv2.goodFeaturesToTrack(

        roi,

        maxCorners=50,

        qualityLevel=0.01,

        minDistance=5,

        blockSize=5

    )


    if (

        points is None

        or

        len(points) < 5

    ):

        points = cv2.goodFeaturesToTrack(

            roi,

            maxCorners=80,

            qualityLevel=0.005,

            minDistance=3,

            blockSize=5

        )


    if points is not None:

        points[:, 0, 0] += x1

        points[:, 0, 1] += y1

        phone_track_points = points

    else:

        phone_track_points = None


    phone_prev_gray = gray

    phone_tracking_fail = 0


# =========================================================
# TRACK PHONE TỪNG FRAME
# =========================================================

def track_phone(frame):

    global phone_track_points
    global phone_prev_gray
    global phone_tracking_fail
    global last_phone_box


    if (

        last_phone_box is None

        or

        phone_prev_gray is None

        or

        phone_track_points is None

    ):

        return False


    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    next_points, status, error = cv2.calcOpticalFlowPyrLK(

        phone_prev_gray,

        gray,

        phone_track_points,

        None,

        winSize=(21, 21),

        maxLevel=3,

        criteria=(

            cv2.TERM_CRITERIA_EPS

            |

            cv2.TERM_CRITERIA_COUNT,

            15,

            0.03

        )

    )


    if (

        next_points is None

        or

        status is None

    ):

        phone_tracking_fail += 1

        phone_prev_gray = gray

        return False


    good_old = phone_track_points[
        status == 1
    ]


    good_new = next_points[
        status == 1
    ]


    if len(good_new) < 4:

        phone_tracking_fail += 1

        phone_prev_gray = gray

        return False


    movement = (
        good_new
        -
        good_old
    )


    dx = float(

        math.fsum(
            movement[:, 0]
        )

        /

        len(movement)

    )


    dy = float(

        math.fsum(
            movement[:, 1]
        )

        /

        len(movement)

    )


    distance = math.sqrt(
        dx * dx + dy * dy
    )


    if distance > 80:

        phone_tracking_fail += 1

        phone_prev_gray = gray

        return False


    x1 = int(
        last_phone_box["x1"]
        +
        dx
    )


    y1 = int(
        last_phone_box["y1"]
        +
        dy
    )


    x2 = int(
        last_phone_box["x2"]
        +
        dx
    )


    y2 = int(
        last_phone_box["y2"]
        +
        dy
    )


    h, w = frame.shape[:2]


    x1 = max(
        0,
        min(
            x1,
            w - 1
        )
    )


    y1 = max(
        0,
        min(
            y1,
            h - 1
        )
    )


    x2 = max(
        x1 + 5,
        min(
            x2,
            w - 1
        )
    )


    y2 = max(
        y1 + 5,
        min(
            y2,
            h - 1
        )
    )


    last_phone_box["x1"] = x1
    last_phone_box["y1"] = y1
    last_phone_box["x2"] = x2
    last_phone_box["y2"] = y2


    phone_track_points = (

        good_new

        .reshape(
            -1,
            1,
            2
        )

        .astype(
            "float32"
        )

    )


    phone_prev_gray = gray

    phone_tracking_fail = 0

    return True


# =========================================================
# HIỂN THỊ BOX
# =========================================================

def update_phone_display():

    global tracked_boxes


    if last_phone_box is None:

        tracked_boxes = []

        return


    if (

        time.time()
        -
        last_phone_box_time

        >

        PHONE_BOX_HOLD_TIME

        and

        phone_track_points is None

    ):

        tracked_boxes = []

        return


    tracked_boxes = [

        {

            "x1":
                last_phone_box["x1"],

            "y1":
                last_phone_box["y1"],

            "x2":
                last_phone_box["x2"],

            "y2":
                last_phone_box["y2"],

            "label":
                "PHONE",

            "conf":
                last_phone_box.get(
                    "conf",
                    0
                ),

            "color":
                (
                    0,
                    0,
                    255
                )

        }

    ]


# =========================================================
# KHOẢNG CÁCH
# =========================================================

def euclidean_distance(
    p1,
    p2
):

    return math.sqrt(

        (p1.x - p2.x) ** 2

        +

        (p1.y - p2.y) ** 2

    )


# =========================================================
# WEBCAM
# =========================================================

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)

cap.set(
    cv2.CAP_PROP_FPS,
    30
)

cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


if not cap.isOpened():

    print(
        "❌ Không mở được webcam"
    )

    exit()


print(
    "✅ Webcam đã kết nối"
)


# =========================================================
# WINDOW
# =========================================================

cv2.namedWindow(
    "Webcam Detection",
    cv2.WINDOW_NORMAL
)

cv2.setWindowProperty(
    "Webcam Detection",
    cv2.WND_PROP_TOPMOST,
    1
)


# =========================================================
# FPS
# =========================================================

fps_start_time = time.time()

fps_counter = 0

fps = 0


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    # =====================================================
    # ĐỌC WEBCAM
    # =====================================================

    ret, frame = cap.read()


    if not ret:

        print(
            "❌ Không nhận được frame"
        )

        break


    frame_count += 1


    h, w, _ = frame.shape


    # =====================================================
    # FPS
    # =====================================================

    fps_counter += 1

    current_time = time.time()


    if (

        current_time
        -
        fps_start_time

        >=

        1

    ):

        fps = fps_counter

        fps_counter = 0

        fps_start_time = current_time


    # =====================================================
    # MEDIAPIPE
    # =====================================================

    rgb_image = cv2.cvtColor(

        frame,

        cv2.COLOR_BGR2RGB

    )


    mp_image = mp.Image(

        image_format=mp.ImageFormat.SRGB,

        data=rgb_image

    )


    frame_timestamp_ms = (
        frame_count * 33
    )


    results = face_landmarker.detect_for_video(

        mp_image,

        frame_timestamp_ms

    )


    # =====================================================
    # TRACK PHONE
    # =====================================================

    if last_phone_box is not None:

        track_phone(frame)


        if (

            phone_tracking_fail
            >=
            MAX_TRACKING_FAIL

        ):

            phone_track_points = None

            phone_tracking_fail = 0


    # =====================================================
    # NHẬN KẾT QUẢ YOLO
    # =====================================================

    if (

        phone_future is not None

        and

        phone_future.done()

    ):

        try:

            yolo_results = phone_future.result()

            valid_boxes = []

            if yolo_results:

                for box_data in yolo_results:

                    try:

                        x1 = int(box_data["x1"])
                        y1 = int(box_data["y1"])
                        x2 = int(box_data["x2"])
                        y2 = int(box_data["y2"])

                        conf = float(
                            box_data["conf"]
                        )

                        if (
                            x2 <= x1
                            or
                            y2 <= y1
                        ):
                            continue

                        valid_boxes.append({
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "conf": conf
                        })

                    except Exception as e:

                        print(
                            "⚠️ YOLO box lỗi:",
                            e
                        )

                        try:

                            coords = (

                                box.xyxy[0]

                                .cpu()

                                .numpy()

                            )


                            conf = float(

                                box.conf[0]

                                .cpu()

                                .item()

                            )


                            x1 = int(
                                coords[0]
                            )

                            y1 = int(
                                coords[1]
                            )

                            x2 = int(
                                coords[2]
                            )

                            y2 = int(
                                coords[3]
                            )


                            if (

                                x2 <= x1

                                or

                                y2 <= y1

                            ):

                                continue


                            valid_boxes.append(

                                {

                                    "x1": x1,

                                    "y1": y1,

                                    "x2": x2,

                                    "y2": y2,

                                    "conf": conf

                                }

                            )


                        except Exception as e:

                            print(
                                "⚠️ YOLO box lỗi:",
                                e
                            )


            # =================================================
            # CÓ ĐIỆN THOẠI
            # =================================================

            if valid_boxes:

                best_box = max(

                    valid_boxes,

                    key=lambda b:
                        b["conf"]

                )


                new_x1 = best_box["x1"]

                new_y1 = best_box["y1"]

                new_x2 = best_box["x2"]

                new_y2 = best_box["y2"]


                # =============================================
                # LÀM MƯỢT BOX
                # =============================================

                if last_phone_box is not None:

                    old_x1 = last_phone_box["x1"]

                    old_y1 = last_phone_box["y1"]

                    old_x2 = last_phone_box["x2"]

                    old_y2 = last_phone_box["y2"]


                    new_x1 = int(

                        old_x1 * BOX_SMOOTHING

                        +

                        new_x1 *
                        (1 - BOX_SMOOTHING)

                    )


                    new_y1 = int(

                        old_y1 * BOX_SMOOTHING

                        +

                        new_y1 *
                        (1 - BOX_SMOOTHING)

                    )


                    new_x2 = int(

                        old_x2 * BOX_SMOOTHING

                        +

                        new_x2 *
                        (1 - BOX_SMOOTHING)

                    )


                    new_y2 = int(

                        old_y2 * BOX_SMOOTHING

                        +

                        new_y2 *
                        (1 - BOX_SMOOTHING)

                    )


                # =============================================
                # LƯU BOX
                # =============================================

                last_phone_box = {

                    "x1": new_x1,

                    "y1": new_y1,

                    "x2": new_x2,

                    "y2": new_y2,

                    "conf":
                        best_box["conf"]

                }


                last_phone_box_time = time.time()


                # =============================================
                # TRACKING
                # =============================================

                init_phone_tracker(

                    frame,

                    last_phone_box

                )


                update_phone_display()


                print(

                    f"📱 PHONE | "

                    f"conf="
                    f"{best_box['conf']:.2f} | "

                    f"box="
                    f"({new_x1},{new_y1})"
                    f"-"
                    f"({new_x2},{new_y2})"

                )


                # =============================================
                # CẢNH BÁO
                # =============================================

                handle_alert(

                    "phone",

                    "📱 Phát hiện sử dụng điện thoại!",

                    "Bạn đang dùng điện thoại"

                )


            else:

                if (

                    last_phone_box is not None

                    and

                    time.time()
                    -
                    last_phone_box_time

                    >

                    PHONE_BOX_HOLD_TIME

                ):

                    last_phone_box = None

                    tracked_boxes = []

                    phone_track_points = None

                    phone_prev_gray = None


            phone_future = None


        except Exception as e:

            print(
                "❌ YOLO result error:",
                e
            )

            phone_future = None


    # =====================================================
    # GỬI FRAME MỚI CHO YOLO
    # =====================================================

    if (

        phone_future is None

        and

        (

            time.time()
            -
            last_phone_ai_time

            >=

            PHONE_AI_INTERVAL

        )

    ):

        phone_future = (

            phone_executor.submit(

                run_phone_detection,

                frame.copy()

            )

        )


        last_phone_ai_time = time.time()


    # =====================================================
    # CẬP NHẬT PHONE
    # =====================================================

    update_phone_display()


    # =====================================================
    # ROBOFLOW SLEEP
    # =====================================================

    if (

        sleep_future is not None

        and

        sleep_future.done()

    ):

        try:

            result_sleep = (
                sleep_future.result()
            )


            sleep_preds = get_predictions(

                result_sleep

            )


            valid_sleep_preds = []


            for pred in sleep_preds:

                try:

                    conf = float(

                        pred.get(
                            "confidence",
                            0
                        )

                    )


                    if (

                        conf
                        >=
                        SLEEP_CONF_THRESH

                    ):

                        valid_sleep_preds.append(
                            pred
                        )


                except:

                    pass


            if valid_sleep_preds:

                sleep_count += 1

            else:

                sleep_count = 0

                sleep_start_time = None


            if sleep_count >= 2:

                print(
                    "😴 Roboflow phát hiện buồn ngủ"
                )


                handle_alert(

                    "slepping",

                    "⚠️ Phát hiện buồn ngủ!",

                    "Bạn đang buồn ngủ"

                )


                sleep_count = 0


            sleep_future = None


        except Exception as e:

            print(
                "❌ Sleep result error:",
                e
            )

            sleep_future = None


    # =====================================================
    # GỬI FRAME SLEEP
    # =====================================================

    if (

        sleep_future is None

        and

        (

            time.time()
            -
            last_sleep_ai_time

            >=

            SLEEP_AI_INTERVAL

        )

    ):

        sleep_future = (

            sleep_executor.submit(

                run_sleep_detection,

                frame.copy()

            )

        )


        last_sleep_ai_time = time.time()


    # =====================================================
    # MEDIAPIPE MẮT + NGÁP
    # =====================================================

    if results.face_landmarks:

        for face_landmarks in results.face_landmarks:

            # =============================================
            # MIỆNG
            # =============================================

            mouth_top = face_landmarks[13]

            mouth_bottom = face_landmarks[14]


            mouth_dist = abs(

                mouth_top.y
                -
                mouth_bottom.y

            )


            if (

                mouth_dist
                >
                MOUTH_OPEN_THRESH

            ):

                yawn_detected = True

            else:

                yawn_detected = False


            # =============================================
            # MẮT
            # =============================================

            left_eye_ids = [

                33,
                160,
                158,
                133,
                153,
                144

            ]


            right_eye_ids = [

                362,
                385,
                387,
                263,
                373,
                380

            ]


            left_eye = [

                face_landmarks[i]

                for i in left_eye_ids

            ]


            right_eye = [

                face_landmarks[i]

                for i in right_eye_ids

            ]


            # =============================================
            # VẼ ĐIỂM MẮT
            # =============================================

            for idx in (

                left_eye_ids
                +
                right_eye_ids

            ):

                p = face_landmarks[idx]


                px = int(
                    p.x * w
                )


                py = int(
                    p.y * h
                )


                cv2.circle(

                    frame,

                    (px, py),

                    2,

                    (0, 255, 0),

                    -1

                )


            # =============================================
            # EAR
            # =============================================

            left_horizontal = (

                euclidean_distance(

                    left_eye[0],

                    left_eye[3]

                )

            )


            right_horizontal = (

                euclidean_distance(

                    right_eye[0],

                    right_eye[3]

                )

            )


            if (

                left_horizontal > 0

                and

                right_horizontal > 0

            ):

                leftEAR = (

                    euclidean_distance(

                        left_eye[1],

                        left_eye[5]

                    )

                    +

                    euclidean_distance(

                        left_eye[2],

                        left_eye[4]

                    )

                ) / (

                    2.0
                    *
                    left_horizontal

                )


                rightEAR = (

                    euclidean_distance(

                        right_eye[1],

                        right_eye[5]

                    )

                    +

                    euclidean_distance(

                        right_eye[2],

                        right_eye[4]

                    )

                ) / (

                    2.0
                    *
                    right_horizontal

                )


                ear = (

                    leftEAR
                    +
                    rightEAR

                ) / 2.0


                # =========================================
                # HIỂN THỊ EAR
                # =========================================

                cv2.putText(

                    frame,

                    f"EAR: {ear:.2f}",

                    (20, 40),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.8,

                    (0, 255, 0),

                    2

                )


                # =========================================
                # NHẮM MẮT
                # =========================================

                if ear < EYE_AR_THRESH:

                    if eye_closed_start is None:

                        eye_closed_start = time.time()


                    closed_duration = (

                        time.time()
                        -
                        eye_closed_start

                    )


                    cv2.putText(

                        frame,

                        f"Closed: "
                        f"{closed_duration:.2f}s",

                        (20, 75),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.7,

                        (0, 0, 255),

                        2

                    )


                    if closed_duration >= 0.2:

                        print(
                            "😴 NHẮM MẮT"
                        )


                        handle_alert(

                            "eye",

                            "⚠️ Nhắm mắt quá lâu!",

                            "Bạn đang buồn ngủ"

                        )


                        eye_closed_start = None


                else:

                    eye_closed_start = None


    # =====================================================
    # VẼ KHUNG ĐIỆN THOẠI
    # =====================================================

    for box in tracked_boxes:

        x1 = int(
            box["x1"]
        )

        y1 = int(
            box["y1"]
        )

        x2 = int(
            box["x2"]
        )

        y2 = int(
            box["y2"]
        )


        color = box["color"]

        label = box["label"]

        conf = box["conf"]


        cv2.rectangle(

            frame,

            (x1, y1),

            (x2, y2),

            color,

            3

        )


        text = (

            f"{label} "
            f"{conf:.2f}"

        )


        cv2.putText(

            frame,

            text,

            (

                x1,

                max(
                    30,
                    y1 - 10
                )

            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            color,

            2

        )


    # =====================================================
    # FPS
    # =====================================================

    cv2.putText(

        frame,

        f"FPS: {fps}",

        (20, 115),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (255, 255, 0),

        2

    )


    # =====================================================
    # TRẠNG THÁI PHONE
    # =====================================================

    if phone_future is not None:

        ai_status = "PHONE AI: YOLO LOCAL"

    elif last_phone_box is not None:

        ai_status = "PHONE: TRACKING REALTIME"

    else:

        ai_status = "PHONE: SEARCHING..."


    cv2.putText(

        frame,

        ai_status,

        (20, 150),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2

    )


    # =====================================================
    # SHOW
    # =====================================================

    cv2.imshow(

        "Webcam Detection",

        frame

    )


    # =====================================================
    # PHÍM THOÁT
    # =====================================================

    key = cv2.waitKey(1) & 0xFF


    if (

        key == ord("q")

        or

        key == ord("1")

    ):

        break


# =========================================================
# GIẢI PHÓNG
# =========================================================

print(
    "Đang đóng chương trình..."
)


cap.release()

cv2.destroyAllWindows()


phone_executor.shutdown(
    wait=False,
    cancel_futures=True
)

sleep_executor.shutdown(
    wait=False,
    cancel_futures=True
)

alert_executor.shutdown(
    wait=False,
    cancel_futures=True
)

face_landmarker.close()


print(
    "✅ Đã đóng Webcam Detection"
)