import datetime
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
from ultralytics import YOLO
from supabase import create_client, Client
import time

timestamp = int(time.time())

SUPABASE_URL = "https://jyjunbzusfrmaywmndpa.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imp5anVuYnp1c2ZybWF5d21uZHBhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTM4NDMxMTgsImV4cCI6MjA2OTQxOTExOH0.IQ6yyyR2OpvQj1lIL1yFsWfVNhJIm2_EFt5Pnv4Bd38"

# Index kamera
CAMERA_1_INDEX = 0      # kamera atas (di atas permukaan air)
CAMERA_2_INDEX = 2      # kamera bawah air

# Model OpenVINO
# Sesuaikan path ini dengan model yang kamu punya
DET_MODEL_HIJAU_PATH = Path("hijau2_openvino_model/hijau2.xml")    # model kotak hijau (misi 1)
DET_MODEL_BIRU_PATH = Path("best2_openvino_model/best2.xml")         # model kotak biru (misi 2)

# Parameter deteksi
CONF_HIJAU = 0.9        # confidence threshold untuk hijau
CONF_BIRU = 0.6         # confidence threshold untuk biru
MIN_AREA = 700          # minimal luas bbox
TOLERANCE_METER = 4     # toleransi jarak ke target dalam meter

# Supabase client
client: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# OpenVINO core
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


def mission1_capture_green_top(
    det_model_hijau,
    target_lat: float,
    target_lon: float,
    camera_index: int = CAMERA_1_INDEX,
    image_slot_name: str = "kamera_atas",
    image_filename: str = None,
    max_kandidat: int = 20,
):
 

    if image_filename is None:
        image_filename = f"{image_slot_name}_{timestamp}.jpg"


    print(f"[MISI 1] Menyalakan kamera atas (index {camera_index}) untuk deteksi kotak hijau ...")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Tidak dapat membuka kamera {camera_index} (kamera atas).")
        return

    best_frame = None
    best_score = -1.0
    best_nav_data = None
    kandidat_terkumpul = 0
    selesai = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Tidak ada frame dari kamera atas, keluar dari loop.")
            break

        frame = cv2.resize(frame, (640, 480))

        # Jalankan deteksi YOLO (model hijau)
        detections = det_model_hijau(frame, conf=CONF_HIJAU, verbose=False)

        if detections and detections[0].boxes:
            for box in detections[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Gambar bbox untuk debugging/visual (opsional)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                luas = (x2 - x1) * (y2 - y1)
                if luas >= MIN_AREA:
                    # Cek jarak ke target
                    latest_nav_data, tolerance_ok = get_latest_nav_and_cog(
                        target_lat, target_lon
                    )

                    if not tolerance_ok or latest_nav_data is None:
                        print(" Kotak hijau terdeteksi, tapi kapal di luar toleransi jarak target misi 1.")
                        continue

                    # Hitung skor ketajaman dan pilih frame terbaik
                    score = skor_ketajaman(frame)
                    kandidat_terkumpul += 1

                    if score > best_score:
                        best_score = score
                        best_frame = frame.copy()
                        best_nav_data = latest_nav_data

                    print(f"[MISI 1] Kandidat ke-{kandidat_terkumpul}, skor ketajaman = {score:.2f}")

                    if kandidat_terkumpul >= max_kandidat:
                        if best_frame is None or best_nav_data is None:
                            print(" Tidak ada frame kandidat yang valid untuk misi 1.")
                            selesai = True
                            break

                        # Tulis metadata ke frame terbaik
                        tulis_metadata_ke_frame(best_frame, best_nav_data)
                        success, encoded_img = cv2.imencode(".jpg", best_frame)
                        if not success:
                            print(" Gagal meng-encode frame ke JPEG (misi 1).")
                            selesai = True
                            break

                        image_bytes = encoded_img.tobytes()

                        # Upload ke Supabase Storage
                        client.storage.from_("missionimages").upload(
                            image_filename,
                            image_bytes,
                            {"content-type": "image/jpeg"},
                        )

                        # Dapatkan public URL dan simpan ke tabel image_mission
                        public_url = client.storage.from_("missionimages").get_public_url(
                            image_filename
                        )
                        client.table("image_mission").insert(
                            {
                                "image_url": public_url,
                                "image_slot_name": image_slot_name,
                            }
                        ).execute()

                        print(f"[MISI 1] Foto {image_filename} ({image_slot_name}) berhasil diunggah.")
                        update_mission_status("image_atas", "selesai")
                        selesai = True
                        break

            if selesai:
                break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Misi 1 dihentikan oleh user (q).")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(" Kamera atas (misi 1) dimatikan.\n")


def capture_underwater_only(
    camera_index: int,
    image_slot_name: str,
    image_filename: str,
    target_lat: float,
    target_lon: float,
    mission_camera: str = "image_bawah",
    max_kandidat: int = 20,
):
   
    print(f"[MISI 2] Menyalakan kamera bawah (index {camera_index}) ...")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Tidak dapat membuka kamera bawah index {camera_index}.")
        return

    best_frame = None
    best_score = -1.0
    best_nav_data = None
    kandidat_terkumpul = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Tidak ada frame dari kamera bawah, keluar dari loop.")
            break

        frame = cv2.resize(frame, (640, 480))

        latest_nav_data, tolerance_ok = get_latest_nav_and_cog(target_lat, target_lon)
        if not tolerance_ok or latest_nav_data is None:
            print(" Kapal di luar toleransi jarak target misi 2 (kamera bawah).")
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Misi 2 (kamera bawah) dihentikan oleh user (q).")
                break
            continue

        # Skor ketajaman
        score = skor_ketajaman(frame)
        kandidat_terkumpul += 1

        if score > best_score:
            best_score = score
            best_frame = frame.copy()
            best_nav_data = latest_nav_data

        print(f"[MISI 2] Kamera bawah kandidat ke-{kandidat_terkumpul}, skor ketajaman = {score:.2f}")

        if kandidat_terkumpul >= max_kandidat:
            if best_frame is None or best_nav_data is None:
                print(" Tidak ada frame kandidat yang valid dari kamera bawah.")
                break

            if file_existing(client, "missionimages", image_filename):
                print(f" File {image_filename} sudah ada di storage, skip upload.")
                break

            tulis_metadata_ke_frame(best_frame, best_nav_data)
            success, encoded_img = cv2.imencode(".jpg", best_frame)
            if not success:
                print(" Gagal meng-encode frame ke JPEG (kamera bawah).")
                break

            image_bytes = encoded_img.tobytes()

            # Upload ke Supabase Storage
            client.storage.from_("missionimages").upload(
                image_filename,
                image_bytes,
                {"content-type": "image/jpeg"},
            )

            # Dapatkan public URL dan simpan ke tabel image_mission
            public_url = client.storage.from_("missionimages").get_public_url(
                image_filename
            )
            client.table("image_mission").insert(
                {
                    "image_url": public_url,
                    "image_slot_name": image_slot_name,
                }
            ).execute()

            print(f"[MISI 2] Foto {image_filename} ({image_slot_name}) dari kamera bawah berhasil diunggah.")
            update_mission_status(mission_camera, "selesai")
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Misi 2 (kamera bawah) dihentikan oleh user (q).")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(" Kamera bawah (misi 2) dimatikan.\n")


def mission2_detect_blue_and_trigger_underwater(
    det_model_biru,
    camera_atas_index: int,
    camera_bawah_index: int,
    target_biru_lat: float,
    target_biru_lon: float,
    target_bawah_lat: float,
    target_bawah_lon: float,
):

    print(f"[MISI 2] Menyalakan kamera atas (index {camera_atas_index}) untuk deteksi kotak biru ...")
    cap_atas = cv2.VideoCapture(camera_atas_index)

    if not cap_atas.isOpened():
        print(f"Tidak dapat membuka kamera atas index {camera_atas_index} (misi 2).")
        return

    triggered = False
    MIN_AREA = 850 

    while cap_atas.isOpened():
        ret, frame_atas = cap_atas.read()
        if not ret:
            print("Tidak ada frame dari kamera atas (misi 2), keluar dari loop.")
            break

        frame_atas = cv2.resize(frame_atas, (640, 480))

        # Jalankan deteksi YOLO (model biru)
        detections = det_model_biru(frame_atas, conf=CONF_BIRU, verbose=False)

        if detections and detections[0].boxes:
            for box in detections[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                luas = (x2 - x1) * (y2 - y1)
                if luas >= MIN_AREA:
                    latest_nav_data, tolerance_ok = get_latest_nav_and_cog(
                        target_biru_lat, target_biru_lon
                    )

                    if tolerance_ok and latest_nav_data is not None:
                        print(" Kotak BIRU terdeteksi dan kapal dalam toleransi jarak target misi 2.")
                        triggered = True
                        break
                    else:
                        print(" Kotak BIRU terdeteksi, tapi kapal di luar toleransi jarak target misi 2.")

            if triggered:
                break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Misi 2 (kamera atas) dihentikan oleh user (q).")
            break

    cap_atas.release()
    cv2.destroyAllWindows()
    print(" Kamera atas (misi 2) dimatikan.\n")

    # Jika ter-trigger, jalankan kamera bawah
    if triggered:
        print("[MISI 2] Memicu kamera bawah air untuk mengambil foto dengan Laplacian ...")
        capture_underwater_only(
            camera_index=camera_bawah_index,
            image_slot_name="kamera_bawah",
            image_filename=f"kamera_bawah_{timestamp}.jpg",
            target_lat=target_bawah_lat,
            target_lon=target_bawah_lon,
            mission_camera="image_bawah",
            max_kandidat=20,
        )
    else:
        print("Kamera bawah TIDAK dipicu karena tidak ada deteksi biru yang valid.")


def main():
    # Bersihkan file dan data lama di Supabase (sekali di awal sebelum dua misi)
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

    # Tentukan lintasan dari tabel map_state
    view_type = get_current_view_type()

    if view_type == "lintasan1":
        target_hijau_lat, target_hijau_lon = get_target_location_by_id(1)  # misi 1
        target_biru_lat, target_biru_lon = get_target_location_by_id(2)   # misi 2
    elif view_type == "lintasan2":
        target_hijau_lat, target_hijau_lon = get_target_location_by_id(3)  # misi 1
        target_biru_lat, target_biru_lon = get_target_location_by_id(4)    # misi 2
    else:
        print(f"view_type tidak dikenal: {view_type}. Harusnya 'lintasan1' atau 'lintasan2'.")
        return

    if target_hijau_lat is None or target_hijau_lon is None:
        print(" Gagal mengambil target_lokasi misi 1 (hijau) dari database.")
        return

    if target_biru_lat is None or target_biru_lon is None:
        print(" Gagal mengambil target_lokasi misi 2 (biru/bawah) dari database.")
        return

    print(f"Target misi 1 (hijau, kamera atas) : {formatA(target_hijau_lat, target_hijau_lon)}")
    print(f"Target misi 2 (biru & kamera bawah): {formatA(target_biru_lat, target_biru_lon)}")

    # Load model deteksi untuk misi 1 (hijau)
    print("Meload model YOLO + OpenVINO untuk misi 1 (kotak hijau) ...")
    det_model_hijau = load_model(DET_MODEL_HIJAU_PATH, "CPU")
    print(" Model hijau siap digunakan.\n")

    # Load model deteksi untuk misi 2 (biru)
    print("Meload model YOLO + OpenVINO untuk misi 2 (kotak biru) ...")
    det_model_biru = load_model(DET_MODEL_BIRU_PATH, "CPU")
    print(" Model biru siap digunakan.\n")

    print("MULAI MISI 1: Deteksi kotak hijau dengan kamera atas ")
    mission1_capture_green_top(
        det_model_hijau=det_model_hijau,
        target_lat=target_hijau_lat,
        target_lon=target_hijau_lon,
        camera_index=CAMERA_1_INDEX,
        image_slot_name="kamera_atas",
        image_filename=f"kamera_atas_{timestamp}.jpg",
        max_kandidat=10,
    )
    print("MISI 1 SELESAI (atau dihentikan) \n")


    print("MULAI MISI 2: Deteksi kotak biru (kamera atas) → trigger kamera bawah ")
    mission2_detect_blue_and_trigger_underwater(
        det_model_biru=det_model_biru,
        camera_atas_index=CAMERA_1_INDEX,
        camera_bawah_index=CAMERA_2_INDEX,
        target_biru_lat=target_biru_lat,
        target_biru_lon=target_biru_lon,
        target_bawah_lat=target_biru_lat,   # pakai posisi yang sama untuk kamera bawah
        target_bawah_lon=target_biru_lon,
    )
    print("MISI 2 SELESAI (atau dihentikan) \n")


if __name__ == "__main__":
    main()
