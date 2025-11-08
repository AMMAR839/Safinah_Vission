from supabase import create_client, Client

# Supabase config
SUPABASE_URL = "https://jyjunbzusfrmaywmndpa.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imp5anVuYnp1c2ZybWF5d21uZHBhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTM4NDMxMTgsImV4cCI6MjA2OTQxOTExOH0.IQ6yyyR2OpvQj1lIL1yFsWfVNhJIm2_EFt5Pnv4Bd38"


client: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

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

# Hapus file dari bucket Storage
try:
    client.storage.from_("missionimages").remove(["kamera_atas.jpg", "kamera_bawah.jpg"])
    print("File kamera_atas.jpg dan kamera_bawah.jpg berhasil dihapus dari storage.")
    update_mission_status("image_atas", "belum")
    update_mission_status("image_bawah", "belum")
except Exception as e:
    print("Gagal menghapus file dari storage:", e)

# Hapus semua record di tabel image_mission
try:
    response = client.table("image_mission").delete().neq("id", 0).execute()
    print(f"Semua data di tabel image_mission berhasil dihapus.")
except Exception as e:
    print("Gagal menghapus data dari tabel image_mission:", e)
