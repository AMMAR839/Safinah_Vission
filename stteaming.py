import os, time, datetime, threading
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import openvino as ov
from ultralytics import YOLO
from flask import Flask, Response, jsonify
from supabase import create_client, Client

# ================== KONFIGURASI ==================
SUPABASE_URL = "https://jyjunbzusfrmaywmndpa.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imp5anVuYnp1c2ZybWF5d21uZHBhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTM4NDMxMTgsImV4cCI6MjA2OTQxOTExOH0.IQ6yyyR2OpvQj1lIL1yFsWfVNhJIm2_EFt5Pnv4Bd38"

# Sumber kamera (boleh int index USB cam, atau string RTSP)
CAM1_SRC: Union[int, str] = 0  # kamera_atas
CAM2_SRC: Union[int, str] = 1  # kamera_bawah

# Model OpenVINO (seperti milikmu)
DET_MODEL_PATH = Path("hijau2_openvino_model/hijau2.xml")

# Parameter deteksi
CONF_THRESHOLD = 0.9
MIN_AREA = 500
TOLERANCE_METER = 3

# Param video/stream
WIDTH, HEIGHT = 640, 480
FPS = 20
JPEG_QUALITY = 70
PORT = 5000



# ================== INIT GLOBAL ==================
app = Flask(__name__)
client: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
core = ov.Core()
timestamp_base = int(time.time())

# Buffer MJPEG (bytes) + frame numpy terbaru per kamera
latest_jpeg: Dict[str, bytes] = {"atas": b"", "bawah": b""}
latest_frame: Dict[str, Optional[np.ndarray]] = {"atas": None, "bawah": None}
buf_lock = threading.Lock()

# Status upload (agar upload 1x/slot)
uploaded_done: Dict[str, bool] = {"atas": False, "bawah": False}
status_flags = {"image_atas": "belum", "image_bawah": "belum"}

# ================== UTIL ==================
def get_cardinal_direction(value, coord_type):
    if coord_type == "lat": return "N" if value >= 0 else "S"
    if coord_type == "lon": return "E" if value >= 0 else "W"
    return ""

def formatA(lat, lon):
    lat_dir = get_cardinal_direction(lat, "lat")
    lon_dir = get_cardinal_direction(lon, "lon")
    return f"{lat_dir} {abs(lat):.6f} {lon_dir} {abs(lon):.6f}"

def tolerance_distance(lat1, lon1, lat2, lon2, tolerance_meter=TOLERANCE_METER):
    R = 6371000
    phi1 = np.radians(lat1); phi2 = np.radians(lat2)
    d_phi = np.radians(lat2 - lat1); d_lam = np.radians(lon2 - lon1)
    a = np.sin(d_phi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(d_lam/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return (R*c) <= tolerance_meter

def ms_to_kmh(sog_ms):
    if sog_ms is None or not isinstance(sog_ms, (int, float)): return 0.0
    return float(sog_ms) * 3.6

def update_mission_status(mission_id: str, status: str) -> None:
    try:
        res = client.table("data_mission").update({mission_id: status}).eq("id", 1).execute()
        status_flags[mission_id] = status
        if getattr(res, "error", None): print(f"Gagal update status {mission_id}: {res.error}")
        else: print(f"Status {mission_id} -> {status}")
    except Exception as e:
        print("Gagal update status:", e)

def file_existing(bucket_name: str, filename: str) -> bool:
    try:
        files = client.storage.from_(bucket_name).list()
        return any(f["name"] == filename for f in files)
    except Exception as e:
        print("Cek file existing error:", e)
        return False

def get_target_location_by_id(target_id: int):
    res = client.table("target_gambar").select("lat, lon").eq("id", target_id).limit(1).execute()
    if not res.data: return None, None
    return res.data[0]["lat"], res.data[0]["lon"]

def get_latest_nav_and_cog(target_lat, target_lon):
    nav_res = client.table("nav_data").select("*").order("timestamp", desc=True).limit(1).execute()
    if not nav_res.data: return None, False
    nav = nav_res.data[0]
    lat, lon = nav["latitude"], nav["longitude"]
    sog_kmsh = ms_to_kmh(nav["sog_ms"])
    koordinat_str = formatA(lat, lon)
    cog_res = client.table("cog_data").select("*").order("timestamp", desc=True).limit(1).execute()
    cog = cog_res.data[0]["cog"] if cog_res.data else 0.0
    latest_nav_data = {"Koordinat": koordinat_str, "sog_kmsh": sog_kmsh, "cog": float(cog)}
    tol_ok = tolerance_distance(lat, lon, target_lat, target_lon)
    return latest_nav_data, tol_ok

def tulis_metadata_ke_frame(frame, latest_nav_data):
    lines = [
        f"Day: {datetime.datetime.now().strftime('%a')}",
        f"Date: {datetime.datetime.now().strftime('%d/%m/%Y')}",
        f"Time: {datetime.datetime.now().strftime('%H:%M:%S')}",
        f"Coordinate: {latest_nav_data['Koordinat']}",
        f"SOG: {latest_nav_data['sog_kmsh']:.2f} km/h",
        f"COG: {latest_nav_data['cog']:.2f} deg",
    ]
    y = 30
    for t in lines:
        cv2.putText(frame, t, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(frame, t, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
        y += 30

# ================== YOLO + OpenVINO ==================
core = ov.Core()
def compile_model(det_model_path: Path, device: str):
    det_ov_model = core.read_model(det_model_path)
    ov_config = {}
    if device != "CPU": det_ov_model.reshape({0: [1, 3, 640, 640]})
    if ("GPU" in device) or ("AUTO" in device and "GPU" in core.available_devices):
        ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
    return core.compile_model(det_ov_model, device, ov_config)

def load_model(det_model_path: Path, device: str):
    compiled_model = compile_model(det_model_path, device)
    det = YOLO(det_model_path.parent, task="detect")
    if det.predictor is None:
        custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict"}
        args = {**det.overrides, **custom}
        det.predictor = det._smart_load("predictor")(overrides=args, _callbacks=det.callbacks)
        det.predictor.setup_model(model=det.model)
    det.predictor.model.ov_compiled_model = compiled_model
    return det

# ================== KAMERA ==================
def open_cap(src: Union[int, str]) -> cv2.VideoCapture:
    if isinstance(src, int):
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)  # DSHOW membantu di Windows
    else:
        cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    return cap

def capture_worker(name: str, src: Union[int, str]):
    """BACA kamera terus-menerus → update latest_frame (numpy) dan latest_jpeg (bytes)"""
    cap = open_cap(src)
    print(f"[{name}] capture start src={src} {WIDTH}x{HEIGHT}@{FPS}fps")
    while True:
        if not cap.isOpened():
            cap.release(); time.sleep(0.3); cap = open_cap(src); continue

        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02); continue

        if WIDTH and HEIGHT:
            frame = cv2.resize(frame, (WIDTH, HEIGHT))

        # Simpan frame numpy (untuk deteksi)
        with buf_lock:
            latest_frame[name] = frame.copy()

        # Simpan JPEG (untuk stream RAW)
        ok_jpg, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok_jpg:
            with buf_lock:
                latest_jpeg[name] = buf.tobytes()

    

def detector_worker(
    name: str,
    slot_name: str,        # "kamera_atas" / "kamera_bawah"
    mission_key: str,      # "image_atas" / "image_bawah"
    target_id: int,        # 1 / 2
    det_model
):
    """Ambil frame terbaru tiap interval → deteksi YOLO+OpenVINO → upload 1x jika valid."""
    tlat, tlon = get_target_location_by_id(target_id)
    if tlat is None or tlon is None:
        print(f"[{name}] gagal ambil target id={target_id}"); return
    print(f"[{name}] target:", formatA(tlat, tlon))

    filename = f"{slot_name}_{timestamp_base}.jpg"
    if file_existing("missionimages", filename):
        uploaded_done[name] = True

    while True:
        if uploaded_done.get(name, False):
            time.sleep(1.0); continue  # sudah upload, tidak perlu deteksi lagi (hemat CPU)

        # Ambil copy frame terbaru
        with buf_lock:
            frame = None if latest_frame[name] is None else latest_frame[name].copy()

        # Jalankan deteksi pada frame saat ini
        try:
            dets = det_model(frame, conf=CONF_THRESHOLD, verbose=False)
        except Exception as e:
            print(f"[{name}] error deteksi:", e)
            continue

        triggered = False
        if dets and dets[0].boxes:
            for box in dets[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                luas = (x2-x1)*(y2-y1)
                if luas >= MIN_AREA:
                    latest_nav, ok_tol = get_latest_nav_and_cog(tlat, tlon)
                    if latest_nav: tulis_metadata_ke_frame(frame, latest_nav)
                    # (opsional) gambar bbox
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    if ok_tol:
                        triggered = True
                        break

        # Upload 1x bila triggered
        if triggered and (not uploaded_done.get(name, False)):
            try:
                ok_jpg, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if ok_jpg:
                    client.storage.from_("missionimages").upload(
                        filename, buf.tobytes(), {"content-type":"image/jpeg"}
                    )
                    public_url = client.storage.from_("missionimages").get_public_url(filename)
                    client.table("image_mission").insert({
                        "image_url": public_url,
                        "image_slot_name": slot_name,
                    }).execute()
                    update_mission_status(mission_key, "selesai")
                    uploaded_done[name] = True
                    print(f"[{name}] Upload sukses → {filename}")
            except Exception as e:
                print(f"[{name}] upload gagal:", e)

        

# ================== MJPEG STREAM ==================
def mjpeg_generator(slot: str):
    boundary = b"--frame"
    while True:
        with buf_lock:
            jpg = latest_jpeg.get(slot, b"")
        if jpg:
            yield (boundary + b"\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                   str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
        else:
            time.sleep(0.05)

@app.get("/")
def root():
    return jsonify({
        "ok": True,
        "streams": ["/video_feed/atas", "/video_feed/bawah"],
        "status": status_flags,
        "uploaded_done": uploaded_done
    })

@app.get("/video_feed/atas")
def feed_atas():
    return Response(mjpeg_generator("atas"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.get("/video_feed/bawah")
def feed_bawah():
    return Response(mjpeg_generator("bawah"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ================== MAIN ==================
if __name__ == "__main__":
    print("Load model YOLO+OpenVINO ...")
    det_model = load_model(DET_MODEL_PATH, "CPU")
    print("Model siap.")

    # Mulai thread capture (RAW stream) untuk 2 kamera
    threading.Thread(target=capture_worker, args=("atas", CAM1_SRC), daemon=True).start()
    # threading.Thread(target=capture_worker, args=("bawah", CAM2_SRC), daemon=True).start()

    # Mulai thread detektor (pakai frame terbaru dari capture)
    threading.Thread(target=detector_worker, args=("atas", "kamera_atas", "image_atas", 1, det_model), daemon=True).start()
    # threading.Thread(target=detector_worker, args=("bawah", "kamera_bawah", "image_bawah", 2, det_model), daemon=True).start()

    app.run(host="0.0.0.0", port=PORT, threaded=True)
