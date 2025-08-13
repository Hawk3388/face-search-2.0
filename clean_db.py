import json
import imagehash
from collections import defaultdict
import shutil
import os

def load_database(path="face_embeddings.json"):
    """Loads the face database"""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File {path} not found!")
        return []
    except json.JSONDecodeError:
        print(f"Error loading JSON file {path}")
        return []

def str_to_phash(phash_str):
    """Converts string to phash object"""
    try:
        return imagehash.hex_to_hash(phash_str)
    except (ValueError, TypeError) as e:
        print(f"Error converting phash '{phash_str}': {e}")
        return None

def create_backup(original_file):
    """Creates a backup of the original file"""
    backup_file = "face_embeddings_backup.json"
    try:
        shutil.copy2(original_file, backup_file)
        print(f"✅ Backup created: {backup_file}")
        return backup_file
    except Exception as e:
        print(f"❌ Error creating backup: {e}")
        return None

def find_hash_duplicates(database, hash_threshold=5):
    """
    Finds duplicates based on phash comparison (ultra-optimized version with LSH)
    
    Args:
        database: List of database entries
        hash_threshold: Maximum Hamming distance for phash comparison (default: 5)
    
    Returns:
        List of duplicate groups, where each group is a list of similar entries
    """
    print(f"🔍 Searching for duplicates with hash threshold: {hash_threshold}")
    
    # Collect all valid hashes with their indices
    print("📋 Loading and validating hashes...")
    valid_entries = []
    for i, entry in enumerate(database):
        phash = str_to_phash(entry.get("phash", ""))
        if phash is not None:
            # Convert hash to integer for faster comparisons
            hash_int = int(str(phash), 16)
            valid_entries.append({
                "index": i,
                "entry": entry,
                "phash": phash,
                "hash_int": hash_int
            })
    
    print(f"✅ {len(valid_entries)} of {len(database)} entries have valid hashes")
    
    if len(valid_entries) < 2:
        print("❌ Not enough valid hashes for comparison")
        return []
    
    print("🚀 Using bucket-based optimization...")
    
    # Bucket-based pre-sorting for massive speedup
    # Group hashes by their first 16 bits (65536 buckets)
    buckets = defaultdict(list)
    for entry in valid_entries:
        bucket_key = entry["hash_int"] >> 48  # First 16 bits
        buckets[bucket_key].append(entry)
    
    duplicate_groups = []
    processed_global = set()
    total_comparisons = 0
    
    print("🔄 Processing buckets...")
    bucket_count = 0
    for bucket_key, bucket_entries in buckets.items():
        bucket_count += 1
        if bucket_count % 1000 == 0:
            print(f"Bucket progress: {bucket_count}/{len(buckets)}")
        
        if len(bucket_entries) < 2:
            continue
        
        # Only compare within bucket + adjacent buckets for edge cases
        candidates = bucket_entries.copy()
        
        # Add similar buckets (for hashes near bucket boundary)
        for offset in [-1, 1]:
            neighbor_key = bucket_key + offset
            if neighbor_key in buckets:
                candidates.extend(buckets[neighbor_key])
        
        # Dedupliziere candidates
        seen_indices = set()
        unique_candidates = []
        for entry in candidates:
            if entry["index"] not in seen_indices:
                unique_candidates.append(entry)
                seen_indices.add(entry["index"])
        
        # Schnelle Vergleiche innerhalb der Kandidaten
        processed_local = set()
        for i, entry1 in enumerate(unique_candidates):
            if entry1["index"] in processed_global or i in processed_local:
                continue
                
            current_group = [entry1]
            processed_local.add(i)
            processed_global.add(entry1["index"])
            
            for j in range(i + 1, len(unique_candidates)):
                if j in processed_local:
                    continue
                    
                entry2 = unique_candidates[j]
                if entry2["index"] in processed_global:
                    continue
                
                # Ultra-schneller Bit-XOR Vergleich
                xor_result = entry1["hash_int"] ^ entry2["hash_int"]
                distance = bin(xor_result).count('1')  # Hamming-Distanz
                total_comparisons += 1
                
                if distance <= hash_threshold:
                    current_group.append(entry2)
                    processed_local.add(j)
                    processed_global.add(entry2["index"])
            
            if len(current_group) > 1:
                duplicate_groups.append(current_group)
    
    print(f"✅ Fertig! {len(duplicate_groups)} Duplikat-Gruppen gefunden")
    print(f"🚀 Nur {total_comparisons:,} Vergleiche statt {len(valid_entries)*(len(valid_entries)-1)//2:,} (Speedup: {(len(valid_entries)*(len(valid_entries)-1)//2)/max(total_comparisons,1):.1f}x)")
    return duplicate_groups

def analyze_duplicates(duplicate_groups):
    """Analysiert und zeigt Statistiken über gefundene Duplikate"""
    total_duplicates = sum(len(group) for group in duplicate_groups)
    total_to_remove = sum(len(group) - 1 for group in duplicate_groups)
    
    print(f"\n📊 DUPLIKAT-ANALYSE:")
    print(f"Anzahl Duplikat-Gruppen: {len(duplicate_groups)}")
    print(f"Gesamte duplizierte Einträge: {total_duplicates}")
    print(f"Einträge die entfernt werden: {total_to_remove}")
    
    # Zeige Details der ersten paar Gruppen
    print(f"\n🔍 DETAILS DER ERSTEN {min(5, len(duplicate_groups))} GRUPPEN:")
    for i, group in enumerate(duplicate_groups[:5]):
        print(f"\nGruppe {i+1} ({len(group)} Einträge):")
        for item in group:
            entry = item["entry"]
            print(f"  - Index {item['index']}: {entry.get('image_url', 'N/A')}")
            print(f"    Seite: {entry.get('page_url', 'N/A')}")
            print(f"    Hash: {entry.get('phash', 'N/A')}")

def remove_duplicates(database, duplicate_groups, keep_strategy="first"):
    """
    Entfernt Duplikate aus der Datenbank
    
    Args:
        database: Ursprüngliche Datenbank
        duplicate_groups: Gefundene Duplikat-Gruppen
        keep_strategy: "first" = ersten behalten, "last" = letzten behalten
    
    Returns:
        Bereinigte Datenbank
    """
    print(f"🧹 Entferne Duplikate (Strategie: {keep_strategy})")
    
    # Indizes der zu entfernenden Einträge sammeln
    indices_to_remove = set()
    
    for group in duplicate_groups:
        # Sortiere nach Index
        group_sorted = sorted(group, key=lambda x: x["index"])
        
        if keep_strategy == "first":
            # Ersten behalten, Rest entfernen
            for item in group_sorted[1:]:
                indices_to_remove.add(item["index"])
        elif keep_strategy == "last":
            # Letzten behalten, Rest entfernen
            for item in group_sorted[:-1]:
                indices_to_remove.add(item["index"])
    
    # Neue Datenbank ohne Duplikate erstellen
    cleaned_database = []
    for i, entry in enumerate(database):
        if i not in indices_to_remove:
            cleaned_database.append(entry)
    
    removed_count = len(database) - len(cleaned_database)
    print(f"✅ {removed_count} Duplikate entfernt")
    print(f"📈 Datenbankgröße: {len(database)} → {len(cleaned_database)}")
    
    return cleaned_database

def save_cleaned_database(cleaned_database, output_file="face_embeddings_cleaned.json"):
    """Speichert die bereinigte Datenbank"""
    try:
        with open(output_file, "w") as f:
            json.dump(cleaned_database, f, indent=2)
        print(f"✅ Bereinigte Datenbank gespeichert: {output_file}")
        
        # Dateigröße vergleichen
        original_size = os.path.getsize("face_embeddings.json") if os.path.exists("face_embeddings.json") else 0
        new_size = os.path.getsize(output_file)
        
        if original_size > 0:
            size_reduction = ((original_size - new_size) / original_size) * 100
            print(f"📉 Dateigröße reduziert um {size_reduction:.1f}% ({original_size/1024/1024:.1f}MB → {new_size/1024/1024:.1f}MB)")
        
        return True
    except Exception as e:
        print(f"❌ Fehler beim Speichern: {e}")
        return False

def interactive_duplicate_removal():
    """Interaktiver Modus zur Duplikat-Entfernung"""
    print("🤖 INTERAKTIVE DUPLIKAT-BEREINIGUNG")
    print("=" * 50)
    
    # Schwellenwert abfragen
    while True:
        try:
            threshold = int(input("Hash-Schwellenwert (0-64, empfohlen: 5): "))
            if 0 <= threshold <= 64:
                break
            else:
                print("Bitte einen Wert zwischen 0 und 64 eingeben.")
        except ValueError:
            print("Bitte eine gültige Zahl eingeben.")
    
    # Strategie abfragen
    while True:
        strategy = input("Welchen Eintrag pro Gruppe behalten? (first/last): ").lower()
        if strategy in ["first", "last"]:
            break
        print("Bitte 'first' oder 'last' eingeben.")
    
    return threshold, strategy

def main():
    """Hauptfunktion"""
    print("🧹 DATENBANK-BEREINIGER")
    print("=" * 50)
    
    # Datenbank laden
    print("📂 Lade Datenbank...")
    database = load_database()
    
    if not database:
        print("❌ Keine Datenbank gefunden oder leer!")
        return
    
    print(f"✅ {len(database)} Einträge geladen")
    
    # Backup erstellen
    backup_file = create_backup("face_embeddings.json")
    if not backup_file:
        response = input("⚠️ Backup fehlgeschlagen. Trotzdem fortfahren? (y/N): ")
        if response.lower() != 'y':
            return
    
    # Interaktive Einstellungen oder Standard verwenden
    use_interactive = input("Interaktiven Modus verwenden? (y/N): ").lower() == 'y'
    
    if use_interactive:
        threshold, strategy = interactive_duplicate_removal()
    else:
        threshold = 5
        strategy = "first"
        print(f"🔧 Standard-Einstellungen: Schwellenwert={threshold}, Strategie={strategy}")
    
    # Duplikate finden
    duplicate_groups = find_hash_duplicates(database, threshold)
    
    if not duplicate_groups:
        print("🎉 Keine Duplikate gefunden! Datenbank ist bereits sauber.")
        return
    
    # Analyse anzeigen
    analyze_duplicates(duplicate_groups)
    
    # Bestätigung abfragen
    response = input(f"\n❓ Möchten Sie die Duplikate entfernen? (y/N): ")
    if response.lower() != 'y':
        print("❌ Abgebrochen.")
        return
    
    # Duplikate entfernen
    cleaned_database = remove_duplicates(database, duplicate_groups, strategy)
    
    # Bereinigte Datenbank speichern
    success = save_cleaned_database(cleaned_database)
    
    if success:
        print("\n🎉 Bereinigung erfolgreich abgeschlossen!")
        
        # Fragen ob die ursprüngliche Datei ersetzt werden soll
        replace = input("📝 Soll die ursprüngliche Datei ersetzt werden? (y/N): ")
        if replace.lower() == 'y':
            try:
                shutil.move("face_embeddings_cleaned.json", "face_embeddings.json")
                print("✅ Ursprüngliche Datei erfolgreich ersetzt")
            except Exception as e:
                print(f"❌ Fehler beim Ersetzen: {e}")
    
    print(f"\n💾 Backup verfügbar unter: {backup_file}")

if __name__ == "__main__":
    main()
