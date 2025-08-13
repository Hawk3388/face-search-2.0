import json
import imagehash
from collections import defaultdict
import shutil
import os

def load_database(path="face_embeddings.json"):
    """Lädt die Gesichts-Datenbank"""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Datei {path} nicht gefunden!")
        return []
    except json.JSONDecodeError:
        print(f"Fehler beim Laden der JSON-Datei {path}")
        return []

def str_to_phash(phash_str):
    """Konvertiert String zu phash-Objekt"""
    try:
        return imagehash.hex_to_hash(phash_str)
    except (ValueError, TypeError) as e:
        print(f"Fehler beim Konvertieren des phash '{phash_str}': {e}")
        return None

def create_backup(original_file):
    """Erstellt ein Backup der ursprünglichen Datei"""
    backup_file = "face_embeddings_backup.json"
    try:
        shutil.copy2(original_file, backup_file)
        print(f"✅ Backup erstellt: {backup_file}")
        return backup_file
    except Exception as e:
        print(f"❌ Fehler beim Erstellen des Backups: {e}")
        return None

def find_hash_duplicates(database, hash_threshold=5):
    """
    Findet Duplikate basierend auf phash-Vergleich (ultra-optimierte Version mit LSH)
    
    Args:
        database: Liste der Datenbankeinträge
        hash_threshold: Maximale Hamming-Distanz für phash-Vergleich (Standard: 5)
    
    Returns:
        Liste von Duplikat-Gruppen, wobei jede Gruppe eine Liste ähnlicher Einträge ist
    """
    print(f"🔍 Suche nach Duplikaten mit Hash-Schwellenwert: {hash_threshold}")
    
    # Alle gültigen Hashes mit ihren Indizes sammeln
    print("📋 Lade und validiere Hashes...")
    valid_entries = []
    for i, entry in enumerate(database):
        phash = str_to_phash(entry.get("phash", ""))
        if phash is not None:
            # Konvertiere Hash zu Integer für schnellere Vergleiche
            hash_int = int(str(phash), 16)
            valid_entries.append({
                "index": i,
                "entry": entry,
                "phash": phash,
                "hash_int": hash_int
            })
    
    print(f"✅ {len(valid_entries)} von {len(database)} Einträgen haben gültige Hashes")
    
    if len(valid_entries) < 2:
        print("❌ Nicht genug gültige Hashes für Vergleich")
        return []
    
    print("🚀 Verwende Bucket-basierte Optimierung...")
    
    # Bucket-basierte Vorsortierung für massive Speedup
    # Gruppiere Hashes nach ihren ersten 16 Bits (65536 Buckets)
    buckets = defaultdict(list)
    for entry in valid_entries:
        bucket_key = entry["hash_int"] >> 48  # Erste 16 Bits
        buckets[bucket_key].append(entry)
    
    duplicate_groups = []
    processed_global = set()
    total_comparisons = 0
    
    print("🔄 Verarbeite Buckets...")
    bucket_count = 0
    for bucket_key, bucket_entries in buckets.items():
        bucket_count += 1
        if bucket_count % 1000 == 0:
            print(f"Bucket-Fortschritt: {bucket_count}/{len(buckets)}")
        
        if len(bucket_entries) < 2:
            continue
        
        # Nur innerhalb des Buckets vergleichen + angrenzende Buckets für Grenzfälle
        candidates = bucket_entries.copy()
        
        # Füge ähnliche Buckets hinzu (für Hashes nahe der Bucket-Grenze)
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
