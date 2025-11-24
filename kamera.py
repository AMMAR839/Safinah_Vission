import datetime
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
from ultralytics import YOLO
from supabase import create_client, Client
import time

timestamp= int(time.time())

SUPABASE_URL = "https://jyjunbzusfrmaywmndpa.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imp5anVuYnp1c2ZybWF5d21uZHBhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTM4NDMxMTgsImV4cCI6MjA2OTQxOTExOH0.IQ6yyyR2OpvQj1lIL1yFsWfVNhJIm2_EFt5Pnv4Bd38"

# Index kamera 
CAMERA_1_INDEX = 0   
CAMERA_2_INDEX = 1   # kamera_bawah



# Model OpenVINO
DET_MODEL_PATH = Path("hijau2_openvino_model/hijau2.xml")

# Parameter deteksi
CONF_THRESHOLD = 0.9
MIN_AREA = 700           # minimal luas bbox
TOLERANCE_METER = 3       # toleransi jarak ke target dalam meter

# Supabase client
client: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# OpenVINO core

core = ov.Core()



def skor_ketajaman(frame):
    # Semakin besar varians Laplacian, semakin tajam gambarnya.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var = lap.var()
    return float(var)


def update_mission_status(mission_id: str, status: str) -> None:
    """ image_atas, image_bawah, misssion_finish"""
    try:
        update_data = {mission_id: status}
        res = client.table("data_mission").update(update_data).eq("id", 1).execute()
        # supabase-py mengembalikan objek dengan atribut `error`
        if getattr(res, "error", None):
            print(f"Gagal memperbarui status misi: {res.error}")
        else:
            print(f"Status misi '{mission_id}' berhasil diperbarui menjadi {status}.")
    except Exception as e:
        print("Gagal memperbarui status misi:", e)
          
# format A

def get_cardinal_direction(value, coord_type):
    if coord_type == "lat":
        return "N" if value >= 0 else "S"
    elif coord_type == "lon":
        return "E" if value >= 0 else "W"
    return ""


def formatA(lat, lon):
    # Format koordinat: N 1.234567 E 2.345678
    lat_dir = get_cardinal_direction(lat, "lat")
    lon_dir = get_cardinal_direction(lon, "lon")
    return f"{lat_dir} {abs(lat):.6f} {lon_dir} {abs(lon):.6f}"


def tolerance_distance(lat1, lon1, lat2, lon2, tolerance_meter=TOLERANCE_METER):
    R = 6371000  # radius bumi dalam meter
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

#  SUPABASE DATA
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
        # fallback supaya script tetap jalan
        return "lintasan1"


def get_target_location_by_id(target_id: int):
    """
      id=1 → target untuk kamera_atas
      id=2 → target untuk kamera_bawah
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


def get_latest_nav_and_cog(target_lat, target_lon):
    nav_res = (
        client.table("nav_data")
        .select("*")
        .order("timestamp", desc=True)
        .limit(1)
        .execute()
    )
    if not nav_res.data:
        return None, False

    nav = nav_res.data[0]
    lat = nav["latitude"]
    lon = nav["longitude"]
    sog_kmsh = ms_to_kmh(nav["sog_ms"])
    koordinat_str = formatA(lat, lon)

    cog_res = (
        client.table("cog_data")
        .select("*")
        .order("timestamp", desc=True)
        .limit(1)
        .execute()
    )
    cog = cog_res.data[0]["cog"] if cog_res.data else 0.0

    latest_nav_data = {
        "Koordinat": koordinat_str,
        "sog_kmsh": sog_kmsh,
        "cog": float(cog),
    }

    tolerance_ok = tolerance_distance(lat, lon, target_lat, target_lon)
    return latest_nav_data, tolerance_ok


def image_slot_already_filled(slot_name: str) -> bool:
    res = (
        client.table("image_mission")
        .select("id")
        .eq("image_slot_name", slot_name)
        .limit(1)
        .execute()
    )
    return bool(res.data)

#  OPENVINO + YOLO

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


#  Nulis metadata ke frame

def tulis_metadata_ke_frame(frame, latest_nav_data):
    """Tulis teks metadata ke dalam frame."""
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
    # outline hitam (lebih tebal)
        cv2.putText(
            frame,
            text_line,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),        # hitam
            3,                # tebal outline
            cv2.LINE_AA,
        )
        # isi putih (lebih tipis)
        cv2.putText(
            frame,
            text_line,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),  # putih
            1,                # tebal isi
            cv2.LINE_AA,
        )
        y_offset += 30


#  PROSES KAMERA
def capture_from_camera(
    det_model,
    camera_index: int,
    image_slot_name: str,
    image_filename: str,
    target_lat: float,
    target_lon: float,
    mission_camera: str = "",
    max_kandidat: int = 20,
):

    # Jika slot sudah terisi, tidak usah buka kamera
    if image_slot_already_filled(image_slot_name):
        print(f" Slot {image_slot_name} sudah punya foto. Skip kamera {camera_index}.")
        return

    print(f"Menyalakan kamera {camera_index} untuk slot {image_slot_name} ...")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Tidak dapat membuka kamera {camera_index}.")
        return

    best_frame = None
    best_score = -1.0
    kandidat_terkumpul = 0
   
    window_name = f"Kamera {camera_index} - {image_slot_name}"
    while cap.isOpened() :
        ret, frame = cap.read()
        if not ret:
            print("Tidak ada frame, keluar dari loop.")
            break

        frame = cv2.resize(frame, (640, 480))

        # Jalankan deteksi YOLO + OpenVINO
        detections = det_model(frame, conf=CONF_THRESHOLD, verbose=False)

        if detections and detections[0].boxes:
            for box in detections[0].boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Gambar bbox untuk visual
                color = (0, 255, 0) if cls_id == 0 else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                luas = (x2 - x1) * (y2 - y1)
                if luas >= MIN_AREA:
                    if file_existing(client, "missionimages", image_filename):
                        print(f" File {image_filename} sudah ada di storage, skip upload.")
                        break  

                    latest_nav_data, tolerance_ok = get_latest_nav_and_cog(
                        target_lat, target_lon
                    )

                    if tolerance_ok :


                        score = skor_ketajaman(frame)
                        kandidat_terkumpul += 1

                        if score > best_score:
                            best_score = score
                            best_frame = frame.copy()
                        
                        if kandidat_terkumpul >= max_kandidat:
                            # Tulis metadata ke frame terbaik
                            tulis_metadata_ke_frame(best_frame, latest_nav_data)
                            success, encoded_img = cv2.imencode(".jpg", best_frame)
                            if not success:
                                print(" Gagal meng-encode frame ke JPEG.")
                            image_bytes = encoded_img.tobytes()

                            # Upload langsung bytes ke Supabase Storage
                            client.storage.from_("missionimages").upload(
                                image_filename,
                                image_bytes,
                                {"content-type": "image/jpeg"},
                            )

                            # Dapatkan public URL dan simpan ke tabel image_mission (ROW BARU)
                            public_url = client.storage.from_("missionimages").get_public_url(
                                image_filename
                            )
                            client.table("image_mission").insert(
                                {
                                    "image_url": public_url,
                                    "image_slot_name": image_slot_name,
                                }
                            ).execute()

                            print(f" Foto {image_filename} ({image_slot_name}) berhasil diunggah.")
                            update_mission_status(mission_camera, "selesai")
                            cap.release()
                            break
                        # Encode frame ke JPEG  
                    else:
                        print(" Objek terdeteksi, tapi di luar toleransi jarak ke target.")

            

        # Tampilkan frame
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Dihentikan oleh user (q).")
            break
    
    cv2.destroyAllWindows()    
    cap.release()
    print(f" Kamera {camera_index} dimatikan.\n")



def main():
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
        response = client.table("image_mission").delete().neq("id", 0).execute()
        print(f"Semua data di tabel image_mission berhasil dihapus.")
    except Exception as e:
        print("Gagal menghapus data dari tabel image_mission:", e)

        # Tentukan lintasan dari tabel map_state
    view_type = get_current_view_type()

    if view_type == "lintasan1":
        # lintasan1 → pakai ID 1 dan 2
        target_atas_lat, target_atas_lon = get_target_location_by_id(1)
        target_bawah_lat, target_bawah_lon = get_target_location_by_id(2)
    elif view_type == "lintasan2":
        # lintasan2 → pakai ID 3 dan 4
        target_atas_lat, target_atas_lon = get_target_location_by_id(3)
        target_bawah_lat, target_bawah_lon = get_target_location_by_id(4)
    else:
        print(f"view_type tidak dikenal: {view_type}. Harusnya 'lintasan1' atau 'lintasan2'.")
        return


    if target_atas_lat is None or target_atas_lon is None:
        print(" Gagal mengambil target_lokasi kamera_atas (id=1) dari database.")
        return

    if target_bawah_lat is None or target_bawah_lon is None:
        print(" Gagal mengambil target_lokasi kamera_bawah (id=2) dari database.")
        return

    print(f"Target kamera_atas : {formatA(target_atas_lat, target_atas_lon)}")
    print(f"Target kamera_bawah: {formatA(target_bawah_lat, target_bawah_lon)}")

    #  Load model deteksi 
    print("Meload model YOLO + OpenVINO ...")
    det_model = load_model(DET_MODEL_PATH, "CPU")
    print(" Model siap digunakan.")

    
    # Kamera 1 (kamera_atas,target titik A)
    capture_from_camera(
        det_model=det_model,
        camera_index=CAMERA_1_INDEX,
        image_slot_name="kamera_atas",
        image_filename=f"kamera_atas_{timestamp}.jpg",
        target_lat=target_atas_lat,
        target_lon=target_atas_lon,
        mission_camera="image_atas",
        
    )
    
    # Kamera 2 (kamera_bawah,target titik B)
    capture_from_camera(
        det_model=det_model,
        camera_index=CAMERA_2_INDEX,
        image_slot_name="kamera_bawah",
        image_filename=f"kamera_bawah_{timestamp}.jpg",
        target_lat=target_bawah_lat,
        target_lon=target_bawah_lon,
        mission_camera="image_bawah",
    )
    


if __name__ == "__main__":
    main()
