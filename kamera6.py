import datetime
import cv2
import numpy as np
from supabase import create_client, Client
import time

timestamp = int(time.time())

SUPABASE_URL = "https://jyjunbzusfrmaywmndpa.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imp5anVuYnp1c2ZybWF5d21uZHBhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTM4NDMxMTgsImV4cCI6MjA2OTQxOTExOH0.IQ6yyyR2OpvQj1lIL1yFsWfVNhJIm2_EFt5Pnv4Bd38"

# Di Windows: kamera pakai index angka
CAMERA_1_INDEX = 0   
CAMERA_2_INDEX = 1   

TOLERANCE_METER = 1     

# Supabase client
client: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


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


def tulis_metadata_ke_frame(frame, latest_nav_data):
    """Tulis teks metadata ke dalam frame."""
    now = datetime.datetime.now()
    metadata_text = [
        f"Day: {now.strftime('%a')}",
        f"Date: {now.strftime('%d/%m/%Y')}",
        f"Time: {now.strftime('%H:%M:%S')}",
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


def mission1_capture_distance_only(
    target_lat: float,
    target_lon: float,
    camera_index: int = CAMERA_1_INDEX,
    image_slot_name: str = "kamera_atas",
    image_filename: str = None,
):
    """
    Ambil foto dari kamera atas TANPA AI.
    Begitu kapal pertama kali masuk toleransi jarak ke target, langsung foto.
    """
    if image_filename is None:
        image_filename = f"{image_slot_name}_{int(time.time())}.jpg"

    print(f"[MISI 1] Menyalakan kamera {camera_index} (atas, berdasarkan jarak saja) ...")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Tidak dapat membuka kamera {camera_index}.")
        return

    sudah_foto = False

    while cap.isOpened():
        time.sleep(0.3)
        ret, frame = cap.read()
        if not ret:
            print("Tidak ada frame dari kamera, keluar dari loop.")
            break

        frame = cv2.resize(frame, (640, 480))

        # ambil nav dan cek toleransi
        latest_nav_data, tolerance_ok = get_latest_nav_and_cog(target_lat, target_lon)
        if latest_nav_data is None:
            print("Belum ada data nav_data, skip dulu.")
            
        else:
            # tulisan info di frame (optional)
            info_text = "DALAM TOLERANSI" if tolerance_ok else "DI LUAR TOLERANSI"
            cv2.putText(
                frame, info_text, (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0) if tolerance_ok else (0, 0, 255),
                2, cv2.LINE_AA
            )
           

            # Kalau SUDAH MASUK TOLERANSI → FOTO SEKALI
            if tolerance_ok and not sudah_foto:
                sudah_foto = True
                print("[MISI 1] Kapal sudah dalam toleransi, ambil foto dan upload.")

                # tulis metadata ke frame
                tulis_metadata_ke_frame(frame, latest_nav_data)

                success, encoded_img = cv2.imencode(".jpg", frame)
                if not success:
                    print("Gagal meng-encode frame ke JPEG.")
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
                update_mission_status("image_bawah", "proses")

                break  # selesai misi 1

        # tombol manual stop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Misi 1 dihentikan oleh user (q).")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Kamera atas dimatikan (misi 1).\n")


def mission2_capture_underwater_distance_only(
    target_lat: float,
    target_lon: float,
    camera_index: int = CAMERA_2_INDEX,
    image_slot_name: str = "kamera_bawah",
    image_filename: str = None,
):
    """
    Ambil foto dari kamera bawah TANPA AI.
    Begitu kapal pertama kali masuk toleransi jarak ke target biru, langsung foto underwater.
    """
    if image_filename is None:
        image_filename = f"{image_slot_name}_{int(time.time())}.jpg"

    print(f"[MISI 2] Menyalakan kamera {camera_index} (bawah, berdasarkan jarak saja) ...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Tidak dapat membuka kamera bawah {camera_index}.")
        return

    sudah_foto = False

    while cap.isOpened():
        time.sleep(0.3)
        ret, frame = cap.read()
        if not ret:
            print("Tidak ada frame dari kamera bawah, keluar dari loop.")
            break

        frame = cv2.resize(frame, (640, 480))

        latest_nav_data, tolerance_ok = get_latest_nav_and_cog(target_lat, target_lon)
        if latest_nav_data is None:
            print("Belum ada data nav_data, skip dulu.")
        else:
            info_text = "DALAM TOLERANSI" if tolerance_ok else "DI LUAR TOLERANSI"
            cv2.putText(
                frame, info_text, (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0) if tolerance_ok else (0, 0, 255),
                2, cv2.LINE_AA
            )
            

            if tolerance_ok and not sudah_foto:
                sudah_foto = True
                print("[MISI 2] Kapal sudah dalam toleransi target biru, ambil foto underwater & upload.")

                tulis_metadata_ke_frame(frame, latest_nav_data)

                success, encoded_img = cv2.imencode(".jpg", frame)
                if not success:
                    print("Gagal meng-encode frame ke JPEG (bawah).")
                    break

                image_bytes = encoded_img.tobytes()

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

                print(f"[MISI 2] Foto {image_filename} ({image_slot_name}) berhasil diunggah.")
                update_mission_status("image_bawah", "selesai")

                break  # selesai misi 2

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Misi 2 dihentikan oleh user (q).")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Kamera bawah dimatikan (misi 2).\n")


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

    # MISI 1 – BERDASARKAN JARAK SAJA
    print("MULAI MISI 1: Foto otomatis ketika sudah dalam toleransi target hijau (tanpa AI)")
    mission1_capture_distance_only(
        target_lat=target_hijau_lat,
        target_lon=target_hijau_lon,
        camera_index=CAMERA_1_INDEX,
        image_slot_name="kamera_atas",
        image_filename=f"kamera_atas_{timestamp}.jpg",
    )
    print("MISI 1 SELESAI (atau dihentikan) \n")

    # MISI 2 – BERDASARKAN JARAK SAJA
    print("MULAI MISI 2: Foto underwater otomatis ketika dalam toleransi target biru (tanpa AI)")
    mission2_capture_underwater_distance_only(
        target_lat=target_biru_lat,
        target_lon=target_biru_lon,
        camera_index=CAMERA_2_INDEX,
        image_slot_name="kamera_bawah",
        image_filename=f"kamera_bawah_{timestamp}.jpg",
    )
    print("MISI 2 SELESAI (atau dihentikan) \n")


if __name__ == "__main__":
    main()
