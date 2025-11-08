import time
from pymavlink import mavutil
from supabase import create_client, Client
import numpy as np
from datetime import datetime, timezone
from datetime import datetime
from zoneinfo import ZoneInfo

# --- Konfigurasi Supabase ---
SUPABASE_URL = 'https://jyjunbzusfrmaywmndpa.supabase.co'
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imp5anVuYnp1c2ZybWF5d21uZHBhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTM4NDMxMTgsImV4cCI6MjA2OTQxOTExOH0.IQ6yyyR2OpvQj1lIL1yFsWfVNhJIm2_EFt5Pnv4Bd38"
TABLE_NAV_DATA = "nav_data"  # Nama tabel navigasi
TABLE_COG_DATA = "cog_data"       # Nama tabel COG baru

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

last_time = time.time() 

# --- Konfigurasi Koneksi GCS ---
gcs_connection_string = 'udpin:0.0.0.0:14550'
the_connection = mavutil.mavlink_connection(gcs_connection_string, planner_format=False)

def get_mavlink_data_and_send_to_supabase(last_time=last_time):
    """
    Mendengarkan data MAVLink, dan mengirimkannya ke tabel yang berbeda.
    """
    try:
        print(f"Menunggu data MAVLink pada {gcs_connection_string}...")
        
        while True:
            current_time = time.time ()
            # Mendengarkan GLOBAL_POSITION_INT dan GPS_RAW_INT
            msg = the_connection.recv_match(type=['GLOBAL_POSITION_INT', 'GPS_RAW_INT'], blocking=True, timeout=5)
            
            if not msg:
                print("Timeout, tidak ada pesan baru.")
                continue

            current_timestamp =  datetime.now(ZoneInfo("Asia/Jakarta")).isoformat()
           
            
            # --- Logika Pengiriman Data ---

            if msg.get_type() == 'GLOBAL_POSITION_INT':
                nav_data = {
                    "timestamp": current_timestamp,
                    "latitude": msg.lat / 1e7,
                    "longitude": msg.lon / 1e7,
                    "sog_ms": np.sqrt(pow(msg.vx, 2) + pow(msg.vy, 2)) / 100,
                }
                # Kirim data navigasi ke tabel nav_data
                send_to_supabase(nav_data, TABLE_NAV_DATA)
                
            elif msg.get_type() == 'GPS_RAW_INT':
                cog_data = {
                    "cog": msg.cog / 100,  # Konversi dari centi-degrees ke degrees
                    "timestamp": current_timestamp,
                }
                # Kirim data COG ke tabel cog
                send_to_supabase(cog_data, TABLE_COG_DATA)
            
           
            if current_time - last_time > 1000:
                last_time = time.time()
                periodic_delete_all_data()

    except Exception as e:
        print(f"Terjadi error: {e}")
        the_connection.close()
        get_mavlink_data_and_send_to_supabase()

def send_to_supabase(data, table_name):
    """
    Mengirimkan satu baris data ke tabel yang ditentukan di Supabase.
    """
    try:
        response = supabase.table(table_name).insert(data).execute()
        
        if response.data:
            print(f"Data berhasil dikirim ke tabel '{table_name}'.")
        else:
            print(f"Gagal mengirim data ke tabel '{table_name}'.")
            print("Respons error:", response.error)
            
    except Exception as e:
        print(f"Error saat mengirim data ke Supabase: {e}")

def periodic_delete_all_data():
    """
    Menghapus semua data pada kedua tabel setiap 'interval' detik.
    """
    while True:
        try:
            for table in [TABLE_NAV_DATA, TABLE_COG_DATA]:
                response = supabase.table(table).delete().gt('id', 0).execute()
                print(f"Semua data di tabel '{table}' telah dihapus.")
        except Exception as e:
            print(f"Error saat menghapus semua data: {e}")
        

# Jalankan skrip GCS
if __name__ == "__main__":
    get_mavlink_data_and_send_to_supabase(last_time)