import math

R = 6378137  # radius Bumi (WGS84), meter

def destination_point(lat_deg, lon_deg, distance_m, bearing_deg):
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    brg = math.radians(bearing_deg)
    dR = distance_m / R

    lat2 = math.asin(
        math.sin(lat)*math.cos(dR) + math.cos(lat)*math.sin(dR)*math.cos(brg)
    )
    lon2 = lon + math.atan2(
        math.sin(brg)*math.sin(dR)*math.cos(lat),
        math.cos(dR) - math.sin(lat)*math.sin(lat2)
    )

    # normalisasi longitude ke [-180, 180)
    return (math.degrees(lat2), (math.degrees(lon2) + 540) % 360 - 180)

def main():
    print("=== Buat 2 titik berjarak 1 m dari titik tengah ===")
    print("Pilih mode: 1=Atas/Bawah (Utara/Selatan), 2=Kanan/Kiri (Timur/Barat)")
    pilihan = input("Masukkan pilihan (1/2): ").strip()

    try:
        lat = float(input("Masukkan latitude (derajat, contoh -6.2): ").strip())
        lon = float(input("Masukkan longitude (derajat, contoh 106.816666): ").strip())
    except ValueError:
        print("Latitude/longitude tidak valid.")
        return

    jarak = 1.0  # meter (sesuai kebutuhan: 1 m dari titik tengah)

    if pilihan == "1":
        # Atas/Bawah: 0° dan 180°
        p1 = destination_point(lat, lon, jarak, 0)    # Atas/Utara
        p2 = destination_point(lat, lon, jarak, 180)  # Bawah/Selatan
        print("\nMode: Atas/Bawah (±1 m)")
        print(f"Titik tengah -> lat={lat:.9f}, lon={lon:.9f}")
        print(f"Atas (Utara) -> lat={p1[0]:.9f}, lon={p1[1]:.9f}")
        print(f"Bawah(Selatan)-> lat={p2[0]:.9f}, lon={p2[1]:.9f}")

    elif pilihan == "2":
        # Kanan/Kiri: 90° dan 270°
        p1 = destination_point(lat, lon, jarak, 90)   # Kanan/Timur
        p2 = destination_point(lat, lon, jarak, 270)  # Kiri/Barat
        print("\nMode: Kanan/Kiri (±1 m)")
        print(f"Titik tengah -> lat={lat:.9f}, lon={lon:.9f}")
        print(f"Kanan (Timur) -> lat={p1[0]:.9f}, lon={p1[1]:.9f}")
        print(f"Kiri  (Barat) -> lat={p2[0]:.9f}, lon={p2[1]:.9f}")

    else:
        print("Pilihan tidak dikenali. Gunakan 1 atau 2.")
        return

if __name__ == "__main__":
    main()
