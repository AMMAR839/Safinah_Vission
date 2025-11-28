import datetime
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
from ultralytics import YOLO
from supabase import create_client, Client
import time

from pymavlink import mavutil


RUN_TIMESTAMP = int(time.time())

SUPABASE_URL = "https://jyjunbzusfrmaywmndpa.supabase.co"
SUPABASE_ANON_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imp5anVuYnp1c2ZybWF5d21uZHBhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTM4NDMxMTgsImV4cCI6MjA2OTQxOTExOH0."
    "IQ6yyyR2OpvQj1lIL1yFsWfVNhJIm2_EFt5Pnv4Bd38"
)

# Index kamera
CAMERA_1_INDEX = 0    # kamera atas (surface)
CAMERA_2_INDEX = 3    # kamera bawah (underwater) 
max_kandidat = 10


# Path model OpenVINO
MODEL_HIJAU_PATH = Path("hijau2_openvino_model/hijau2.xml")  
MODEL_BIRU_PATH  = Path("best2_openvino_model/best2.xml")    


# Parameter deteksi
CONF_THRESHOLD  = 0.9
MIN_AREA        = 700      # minimal luas bbox
TOLERANCE_METER = 3        # toleransi jarak ke target dalam meter

#   mavproxy.py --master=/dev/ttyACM0,57600 --out=udp:0.0.0.0:14551
MAVLINK_CONNECTION_STRING = "udp:127.0.0.1:14551"
mavlink_master = None

# Supabase & OpenVINO
client: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
core = ov.Core()

def skor_ketajaman(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var = lap.var()
    return float(var)

def update_mission_status(mission_id: str, status: str) -> None:
    try:
        update_data = {mission_id: status}
        res = client.table("data_mission").update(update_data).eq("id", 1).execute()
        if getattr(res, "error", None):
            print(f"Gagal memperbarui status misi: {res.error}")
        else:
            print(f"Status misi '{mission_id}' berhasil diperbarui menjadi {status}.")
    except Exception as e:
        print("Gagal memperbarui status misi:", e)


def get_cardinal_direction(value, coord_type):
    if coord_type == "lat":
        return "N" if value >= 0 else "S"
    elif coord_type == "lon":
        return "E" if value >= 0 else "W"
    return ""


def formatA(lat, lon):
    lat_dir = get_cardinal_direction(lat, "lat")
    lon_dir = get_cardinal_direction(lon, "lon")
    return f"{lat_dir} {abs(lat):.6f} {lon_dir} {abs(lon):.6f}"


def tolerance_distance(lat1, lon1, lat2, lon2, tolerance_meter=TOLERANCE_METER):
    R = 6371000  # meter
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    d_phi = np.radians(lat2 - lat1)
    d_lam = np.radians(lon2 - lon1)

    a = np.sin(d_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lam / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance <= tolerance_meter


def ms_to_kmh(sog_ms):
    if sog_ms is None or not isinstance(sog_ms, (int, float)):
        return 0.0
    return float(sog_ms) * 3.6


def file_existing(client: Client, bucket_name: str, filename: str) -> bool:
    files = client.storage.from_(bucket_name).list()
    return any(f["name"] == filename for f in files)


def get_current_view_type():
    try:
        res = (
            client.table("map_state")
            .select("id, view_type")
            .order("id", desc=True)
            .limit(1)
            .execute()
        )
        if not res.data:
            print("Tabel map_state kosong, default ke 'lintasan1'.")
            return "lintasan1"
        view_type = res.data[0]["view_type"]
        print(f"View type aktif dari map_state: {view_type}")
        return view_type
    except Exception as e:
        print("Gagal membaca map_state:", e)
        return "lintasan1"


def get_target_location_by_id(target_id: int):
    """
      id=1 → target misi 1 (surface hijau)
      id=2 → target misi 2 (surface biru)
    """
    res = (
        client.table("target_gambar")
        .select("lat, lon")
        .eq("id", target_id)
        .limit(1)
        .execute()
    )
    if not res.data:
        return None, None
    return res.data[0]["lat"], res.data[0]["lon"]


def image_slot_already_filled(slot_name: str) -> bool:
    res = (
        client.table("image_mission")
        .select("id")
        .eq("image_slot_name", slot_name)
        .limit(1)
        .execute()
    )
    return bool(res.data)

def compile_model(det_model_path: Path, device: str):
    det_ov_model = core.read_model(det_model_path)
    ov_config = {}
    if device != "CPU":
        det_ov_model.reshape({0: [1, 3, 640, 640]})
    if "GPU" in device or ("AUTO" in device and "GPU" in core.available_devices):
        ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
    det_compiled_model = core.compile_model(det_ov_model, device, ov_config)
    return det_compiled_model


def load_model(det_model_path: Path, device: str):
    compiled_model = compile_model(det_model_path, device)
    det_model = YOLO(det_model_path.parent, task="detect")

    if det_model.predictor is None:
        custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict"}
        args = {**det_model.overrides, **custom}
        det_model.predictor = det_model._smart_load("predictor")(
            overrides=args, _callbacks=det_model.callbacks
        )
        det_model.predictor.setup_model(model=det_model.model)

    det_model.predictor.model.ov_compiled_model = compiled_model
    return det_model

def tulis_metadata_ke_frame(frame, latest_nav_data):
    metadata_text = [
        f"Day: {datetime.datetime.now().strftime('%a')}",
        f"Date: {datetime.datetime.now().strftime('%d/%m/%Y')}",
        f"Time: {datetime.datetime.now().strftime('%H:%M:%S')}",
        f"Coordinate: {latest_nav_data['Koordinat']}",
        f"SOG: {latest_nav_data['sog_kmsh']:.2f} km/h",
        f"COG: {latest_nav_data['cog']:.2f} deg",
    ]
    y_offset = 30
    for text_line in metadata_text:
        cv2.putText(
            frame,
            text_line,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text_line,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y_offset += 30



def connect_mavlink():
    global mavlink_master
    try:
        print(f"Menghubungkan ke autopilot di {MAVLINK_CONNECTION_STRING} ...")
        mavlink_master = mavutil.mavlink_connection(MAVLINK_CONNECTION_STRING)
        print("Menunggu heartbeat...")
        mavlink_master.wait_heartbeat()
        print(
            f"Terhubung ke sysid={mavlink_master.target_system}, "
            f"compid={mavlink_master.target_component}"
        )
    except Exception as e:
        print("Gagal konek MAVLink:", e)
        mavlink_master = None


def goto_next_waypoint():
    global mavlink_master
    if mavlink_master is None:
        print("MAVLink belum terkoneksi, tidak bisa lanjut waypoint.")
        return

    try:
        mavlink_master.mav.mission_request_current_send(
            mavlink_master.target_system,
            mavlink_master.target_component,
        )

        msg = mavlink_master.recv_match(
            type="MISSION_CURRENT",
            blocking=True,
            timeout=3,
        )
        if msg is None:
            print("Tidak mendapat MISSION_CURRENT (timeout).")
            return

        current_seq = msg.seq
        next_seq = current_seq + 1
        print(f"Waypoint saat ini: {current_seq}, lanjut ke: {next_seq}")

        mavlink_master.mav.mission_set_current_send(
            mavlink_master.target_system,
            mavlink_master.target_component,
            next_seq,
        )
        print("Perintah lanjut ke waypoint berikutnya dikirim.")
    except Exception as e:
        print("Gagal mengirim perintah lanjut waypoint:", e)


def set_mode(mode_name: str):
    global mavlink_master
    if mavlink_master is None:
        print("MAVLink belum terkoneksi, tidak bisa set mode.")
        return

    try:
        mode_mapping = mavlink_master.mode_mapping()
    except Exception as e:
        print("Gagal ambil mode mapping:", e)
        return

    if mode_name not in mode_mapping:
        print(f"Mode {mode_name} tidak ada di mode_mapping autopilot.")
        print(f"Mode yang tersedia: {list(mode_mapping.keys())}")
        return

    mode_id = mode_mapping[mode_name]
    print(f"Mengirim mode {mode_name} (id={mode_id})...")
    mavlink_master.mav.set_mode_send(
        mavlink_master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id
    )

PAUSE_MODE_NAME = "STABILIZE"    # ganti ke mode pause yang sesuai


def pause_mission():
    print(f"PAUSE mission: set mode {PAUSE_MODE_NAME}")
    set_mode(PAUSE_MODE_NAME)


def resume_mission_after_photo():
    print("Resume mission: set waypoint berikutnya dan mode AUTO")
    goto_next_waypoint()
    set_mode("AUTO")


def get_latest_nav_and_cog(target_lat, target_lon):
    global mavlink_master
    if mavlink_master is None:
        print("MAVLink belum terkoneksi, tidak bisa ambil nav data.")
        return None, False

    try:
        msg = mavlink_master.recv_match(
            type="GLOBAL_POSITION_INT",
            blocking=True,
            timeout=1,
        )
        if msg is None:
            print("Tidak menerima GLOBAL_POSITION_INT dari autopilot.")
            return None, False

        lat = msg.lat / 1e7
        lon = msg.lon / 1e7

        if getattr(msg, "hdg", None) not in (None, 65535):
            cog_deg = msg.hdg / 100.0
        else:
            cog_deg = 0.0

        sog_ms = 0.0
        if getattr(msg, "vx", None) is not None and getattr(msg, "vy", None) is not None:
            vx = msg.vx / 100.0
            vy = msg.vy / 100.0
            sog_ms = (vx**2 + vy**2) ** 0.5
        else:
            vfr = mavlink_master.recv_match(type="VFR_HUD", blocking=False)
            if vfr is not None and getattr(vfr, "groundspeed", None) is not None:
                sog_ms = float(vfr.groundspeed)

        sog_kmsh = ms_to_kmh(sog_ms)
        koordinat_str = formatA(lat, lon)

        latest_nav_data = {
            "Koordinat": koordinat_str,
            "sog_kmsh": sog_kmsh,
            "cog": float(cog_deg),
        }

        tolerance_ok = tolerance_distance(lat, lon, target_lat, target_lon)
        return latest_nav_data, tolerance_ok

    except Exception as e:
        print("Gagal membaca nav dari MAVLink:", e)
        return None, False


def ada_kotak_target(detections, target_cls: int, min_area=MIN_AREA) -> bool:
    if not detections:
        return False

    det = detections[0]
    if not det.boxes:
        return False

    for box in det.boxes:
        cls_id = int(box.cls[0])
        if cls_id != target_cls:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        luas = (x2 - x1) * (y2 - y1)
        if luas >= min_area:
            return True
    return False


def foto_kamera_bawah_dan_upload(
    image_filename: str,
    image_slot_name: str,
    mission_camera: str,
    latest_nav_data: dict,
):
    
    cap_bawah = cv2.VideoCapture(CAMERA_2_INDEX)
    if not cap_bawah.isOpened():
        print("Tidak bisa membuka kamera bawah.")
        return

    ret, frame_bawah = cap_bawah.read()
    cap_bawah.release()

    if not ret:
        print("Gagal baca frame dari kamera bawah.")
        return

    frame_bawah = cv2.resize(frame_bawah, (640, 480))
    tulis_metadata_ke_frame(frame_bawah, latest_nav_data)

    success, encoded_img = cv2.imencode(".jpg", frame_bawah)
    if not success:
        print("Gagal encode frame kamera bawah ke JPEG.")
        return

    image_bytes = encoded_img.tobytes()

    if file_existing(client, "missionimages", image_filename):
        print(f" File {image_filename} sudah ada di storage, skip upload.")
        return

    client.storage.from_("missionimages").upload(
        image_filename,
        image_bytes,
        {"content-type": "image/jpeg"},
    )

    public_url = client.storage.from_("missionimages").get_public_url(
        image_filename
    )

    client.table("image_mission").insert(
        {
            "image_url": public_url,
            "image_slot_name": image_slot_name,
        }
    ).execute()

    print(f"Foto dari kamera bawah ({image_filename}) berhasil diunggah.")
    update_mission_status(mission_camera, "selesai")


def foto_kamera_atas_dan_upload(
    frame_atas,
    image_filename: str,
    image_slot_name: str,
    mission_camera: str,
    latest_nav_data: dict,
):
   
    frame = cv2.resize(frame_atas, (640, 480))
    tulis_metadata_ke_frame(frame, latest_nav_data)

    success, encoded_img = cv2.imencode(".jpg", frame)
    if not success:
        print("Gagal encode frame kamera atas ke JPEG.")
        return

    image_bytes = encoded_img.tobytes()

    if file_existing(client, "missionimages", image_filename):
        print(f" File {image_filename} sudah ada di storage, skip upload.")
        return

    client.storage.from_("missionimages").upload(
        image_filename,
        image_bytes,
        {"content-type": "image/jpeg"},
    )

    public_url = client.storage.from_("missionimages").get_public_url(
        image_filename
    )

    client.table("image_mission").insert(
        {
            "image_url": public_url,
            "image_slot_name": image_slot_name,
        }
    ).execute()

    print(f"Foto dari kamera atas ({image_filename}) berhasil diunggah.")
    update_mission_status(mission_camera, "selesai")


def run_mission_surface_photo_top(
    det_model,
    target_id: int,
    target_cls: int,
    image_slot_name: str,
    mission_camera: str,
    mission_label: str,
):

    target_lat, target_lon = get_target_location_by_id(target_id)
    if target_lat is None or target_lon is None:
        print(f"[{mission_label}] Gagal ambil target lat/lon untuk target_id={target_id}.")
        return

    print(f"[{mission_label}] Target: {formatA(target_lat, target_lon)}")

    cap_atas = cv2.VideoCapture(CAMERA_1_INDEX)
    if not cap_atas.isOpened():
        print(f"[{mission_label}] Tidak bisa membuka kamera atas.")
        return

    window_name = f"{mission_label}-KameraAtas"
    paused_here = False

    while cap_atas.isOpened() :
        ret, frame_atas = cap_atas.read()
        if not ret:
            print(f"[{mission_label}] Tidak ada frame dari kamera atas.")
            break

        frame_atas = cv2.resize(frame_atas, (640, 480))

        detections = det_model(frame_atas, conf=CONF_THRESHOLD, verbose=False)
        ada_target = ada_kotak_target(detections, target_cls=target_cls)

        latest_nav_data, tolerance_ok = get_latest_nav_and_cog(
            target_lat, target_lon
        )

        if ada_target and tolerance_ok:
            cv2.putText(
                frame_atas,
                latest_nav_data["Koordinat"],
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame_atas,
                "TARGET DETECTED",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            print(f"[{mission_label}] Kotak target (cls={target_cls}) terdeteksi & posisi OK.")

            if not paused_here:
                pause_mission()
                paused_here = True

            if latest_nav_data is None:
                print(f"[{mission_label}] Nav data tidak valid, tunda foto.")
            else:
                image_filename = f"{mission_label}_kamera_atas_{RUN_TIMESTAMP}.jpg"
                foto_kamera_atas_dan_upload(
                    frame_atas=frame_atas,
                    image_filename=image_filename,
                    image_slot_name=image_slot_name,
                    mission_camera=mission_camera,
                    latest_nav_data=latest_nav_data,
                )
                break
                
                resume_mission_after_photo()


        if cv2.waitKey(1) & 0xFF == ord("q"):
            print(f"[{mission_label}] Dihentikan oleh user (q).")
            break

    cap_atas.release()
    cv2.destroyAllWindows()
    print(f"[{mission_label}] Loop kamera atas selesai.")


def run_mission_surface_detect_underwater_photo(
    det_model,
    target_id: int,
    target_cls: int,
    image_slot_name: str,
    mission_camera: str,
    mission_label: str,
):

    target_lat, target_lon = get_target_location_by_id(target_id)
    if target_lat is None or target_lon is None:
        print(f"[{mission_label}] Gagal ambil target lat/lon untuk target_id={target_id}.")
        return

    print(f"[{mission_label}] Target: {formatA(target_lat, target_lon)}")


    cap_atas = cv2.VideoCapture(CAMERA_1_INDEX)
    if not cap_atas.isOpened():
        print(f"[{mission_label}] Tidak bisa membuka kamera atas.")
        return

    window_name = f"{mission_label}-KameraAtas"
    paused_here = False

    while cap_atas.isOpened() :
        ret, frame_atas = cap_atas.read()
        if not ret:
            print(f"[{mission_label}] Tidak ada frame dari kamera atas.")
            break

        frame_atas = cv2.resize(frame_atas, (640, 480))

        detections = det_model(frame_atas, conf=CONF_THRESHOLD, verbose=False)
        ada_target = ada_kotak_target(detections, target_cls=target_cls)

        latest_nav_data, tolerance_ok = get_latest_nav_and_cog(
            target_lat, target_lon
        )

        if ada_target and tolerance_ok:
            cv2.putText(
                frame_atas,
                latest_nav_data["Koordinat"],
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame_atas,
                "TARGET DETECTED",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            print(f"[{mission_label}] Kotak target (cls={target_cls}) terdeteksi & posisi OK.")

            if not paused_here:
                pause_mission()
                paused_here = True

            if latest_nav_data is None:
                print(f"[{mission_label}] Nav data tidak valid, tunda foto.")
            else:
                image_filename = f"{mission_label}_kamera_bawah_{RUN_TIMESTAMP}.jpg"
                foto_kamera_bawah_dan_upload(
                    image_filename=image_filename,
                    image_slot_name=image_slot_name,
                    mission_camera=mission_camera,
                    latest_nav_data=latest_nav_data,
                )
                break
                # lanjut misi (waypoint selanjutnya + AUTO)
                resume_mission_after_photo()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print(f"[{mission_label}] Dihentikan oleh user (q).")
            break

    cv2.destroyAllWindows()
    print(f"[{mission_label}] Loop kamera atas selesai.")

def main():
    # Bersihkan storage & tabel misi
    try:
        files = client.storage.from_("missionimages").list()
        paths = [f["name"] for f in files]
        if paths:
            client.storage.from_("missionimages").remove(paths)
            print("Semua file di bucket missionimages berhasil dihapus.")
        else:
            print("Tidak ada file di bucket missionimages.")
        update_mission_status("image_atas", "belum")
        update_mission_status("image_bawah", "belum")
    except Exception as e:
        print("Gagal menghapus file dari storage:", e)

    try:
        client.table("image_mission").delete().neq("id", 0).execute()
        print("Semua data di tabel image_mission berhasil dihapus.")
    except Exception as e:
        print("Gagal menghapus data dari tabel image_mission:", e)

    # Konek ke autopilot via MAVLink
    connect_mavlink()

    view_type = get_current_view_type()

    
    if view_type == "lintasan1":
        print("=== MENJALANKAN MISI 1 (surface hijau → foto kamera atas) ===")
        det_model_hijau = load_model(MODEL_HIJAU_PATH, "CPU")
        run_mission_surface_photo_top(
            det_model=det_model_hijau,
            target_id=1,              # target_gambar.id untuk lokasi misi 1
            target_cls=0,             # class id kotak hijau di model hijau
            image_slot_name="kamera_atas",
            mission_camera="image_atas",   # update kolom image_atas di data_mission
            mission_label="kamera_atas",
        )
        det_model_biru = load_model(MODEL_BIRU_PATH, "CPU")
        run_mission_surface_detect_underwater_photo(
            det_model=det_model_biru,
            target_id=2,              # target_gambar.id untuk lokasi misi 2
            target_cls=0,             # class id kotak biru di model biru
            image_slot_name="kamera_bawah",
            mission_camera="image_bawah",  # update kolom image_bawah di data_mission
            mission_label="kamera_bawah",
        )


    elif view_type == "lintasan2":
        print("=== MENJALANKAN MISI 2 (surface biru → foto kamera bawah) ===")
        det_model_hijau = load_model(MODEL_HIJAU_PATH, "CPU")
        run_mission_surface_photo_top(
            det_model=det_model_hijau,
            target_id=3,              # target_gambar.id untuk lokasi misi 1
            target_cls=0,             # class id kotak hijau di model hijau
            image_slot_name="kamera_atas",
            mission_camera="image_atas",   # update kolom image_atas di data_mission
            mission_label="kamera_atas",
        )

        det_model_biru = load_model(MODEL_BIRU_PATH, "CPU")
        run_mission_surface_detect_underwater_photo(
            det_model=det_model_biru,
            target_id=4,              # target_gambar.id untuk lokasi misi 2
            target_cls=0,             # class id kotak biru di model biru
            image_slot_name="misi2_kamera_bawah",
            mission_camera="image_bawah",  # update kolom image_bawah di data_mission
            mission_label="misi2_biru",
        )

    else:
        print(f"view_type tidak dikenal: {view_type}. Harusnya 'lintasan1' atau 'lintasan2'.")


if __name__ == "__main__":
    main()
