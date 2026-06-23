import json
import torch
import time
import os
import shutil

def load_database(path="face_embeddings.json"):
    if not os.path.exists(path):
        print(f"❌ '{path}' existiert nicht!")
        return []
    with open(path, "r") as f:
        return json.load(f)

def find_duplicates_hash(database, threshold=0, batch_size=5000):
    """
    Findet Duplikate ausschließlich über den pHash (mittels Hamming-Distanz).
    Ignores Embeddings entirely!
    """
    print("📋 Extrahiere und kovertiere Hashes...")
    valid_indices = []
    bit_arrays = []
    
    for i, entry in enumerate(database):
        h = entry.get("phash")
        if h and isinstance(h, str) and len(h) > 0:
            try:
                # Konvertiere Hex-String zu Binär (z.B. 16 Hex-Zeichen = 64 Bits)
                # len(h) * 4 stellt sicher, dass Nullen am Anfang nicht weggelassen werden
                bit_len = len(h) * 4
                bits = [float(b) for b in format(int(h, 16), f'0{bit_len}b')]
                bit_arrays.append(bits)
                valid_indices.append(i)
            except ValueError:
                continue
                
    n = len(valid_indices)
    if n == 0:
        print("❌ Keine gültigen Hashes gefunden!")
        return set()
    
    # Vereinheitliche Bit-Länge für alle Hashes (verhindert Tensor-Crashes bei ungleichen Längen)
    max_len = max(len(b) for b in bit_arrays)
    for i, b in enumerate(bit_arrays):
        if len(b) < max_len:
            bit_arrays[i] = [0.0] * (max_len - len(b)) + b
            
    print(f"✅ {n} Hashes (jeweils {max_len} Bit) für Vergleich vorbereitet.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Nutze Berechnungs-Einheit: {device}")
    
    # Tensor laden
    X = torch.tensor(bit_arrays, dtype=torch.float32).to(device)
    duplicates_to_remove = set()
    
    print(f"🔍 Berechne Hamming-Distanz in Blöcken von {batch_size} (Max = {threshold} Bits Unterschied)...")
    start_time = time.time()
    
    for i in range(0, n, batch_size):
        end_i = min(i + batch_size, n)
        batch_X = X[i:end_i]
        
        # Manhattan-Distanz (p=1.0) auf 0/1 Vektoren ist exakt die Hamming-Distanz!
        dists = torch.cdist(batch_X, X, p=1.0)
        
        # Finde Paare, deren Distanz <= Schwellenwert ist
        close_pairs = torch.where(dists <= threshold)
        
        b_indices = close_pairs[0].cpu().numpy()
        g_indices = close_pairs[1].cpu().numpy()
        
        for b_idx, g_idx in zip(b_indices, g_indices):
            actual_i = i + b_idx
            actual_j = g_idx
            
            if actual_i < actual_j:
                valid_i = valid_indices[actual_i]
                valid_j = valid_indices[actual_j]
                
                if valid_i not in duplicates_to_remove:
                    duplicates_to_remove.add(valid_j)
        
        elapsed_so_far = time.time() - start_time
        print(f"⏳ Verarbeitet: {end_i}/{n} | Duplikate bisher: {len(duplicates_to_remove)} | Zeit: {elapsed_so_far:.1f}s")
        
        del dists, close_pairs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    print(f"✅ Hash-Vergleich in {elapsed:.2f} Sekunden abgeschlossen!")
    
    return duplicates_to_remove

def main():
    print("🧹 DATENBANK-BEREINIGER (NUR HASHES)")
    print("=" * 50)
    
    db_path = "face_embeddings.json"
    database = load_database(db_path)
    if not database:
        return
        
    print(f"📊 Aktuelle Datenbankgröße: {len(database)}")
    
    # Interaktiver Modus
    print("\nWie strikt soll der pHash-Vergleich sein?")
    print(" 0 = Exakt identisch (100% gleicher Hash)")
    print(" 2 = Minimalste Unterschiede (z.B. minimal andere Komprimierung)")
    print(" 4 = Kleine Unterschiede (z.B. skaliert oder beschnitten)")
    try:
        user_input = input("👉 Wähle max. erlaubte Bit-Unterschiede (Standard=0): ").strip()
        threshold = int(user_input) if user_input else 0
    except ValueError:
        threshold = 0
        
    duplicates_to_remove = find_duplicates_hash(database, threshold=threshold, batch_size=5000)
    
    if not duplicates_to_remove:
        print("🎉 Keine Duplikate gefunden!")
        return
        
    print(f"\n⚠️ Es wurden {len(duplicates_to_remove)} Duplikate gefunden! (Max {threshold} Bits Unterschied)")
    
    response = input("❓ Möchten Sie diese Duplikate jetzt entfernen? (y/N): ")
    if response.lower() != "y":
        print("❌ Abbruch.")
        return
        
    print("🧹 Entferne Duplikate...")
    cleaned_database = [
        entry for i, entry in enumerate(database) 
        if i not in duplicates_to_remove
    ]
    
    backup_path = "face_embeddings_backup.json"
    print(f"💾 Erstelle Backup in {backup_path}")
    shutil.copy2(db_path, backup_path)
    
    out_path = "face_embeddings_cleaned_hash.json"
    print(f"💾 Speichere bereinigte Datenbank unter: {out_path}")
    with open(out_path, "w") as f:
        json.dump(cleaned_database, f, indent=2)
        
    print("🔄 Ersetze Originaldatei...")
    shutil.move(out_path, db_path)
    
    print(f"✅ Fertig! Neue Datenbankgröße: {len(cleaned_database)}")

if __name__ == "__main__":
    main()