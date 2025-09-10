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
    except json.JSONDecodeError as e:
        print(f"Error loading JSON file {path}: {e}")
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

def find_exact_duplicates(database):
    """
    Finds exact duplicates: first by hash (exact match), then by encoding if hash matches
    Uses the same logic as the crawler for consistency
    
    Args:
        database: List of database entries
    
    Returns:
        List of duplicate groups, where each group is a list of exact duplicates
    """
    import numpy as np
    
    print(f"🔍 Searching for exact duplicates (hash = 0, then encoding <= 0.05)")
    
    # Collect all valid entries with hashes and encodings
    print("📋 Loading and validating data...")
    valid_entries = []
    for i, entry in enumerate(database):
        phash = str_to_phash(entry.get("phash", ""))
        embedding = entry.get("embedding")
        if phash is not None and embedding is not None:
            valid_entries.append({
                "index": i,
                "entry": entry,
                "phash": phash,
                "embedding": np.array(embedding)
            })
    
    print(f"✅ {len(valid_entries)} of {len(database)} entries have valid hashes and embeddings")
    
    if len(valid_entries) < 2:
        print("❌ Not enough valid entries for comparison")
        return []
    
    print("🚀 Comparing entries for exact duplicates...")
    
    duplicate_groups = []
    processed_indices = set()
    total_comparisons = 0
    hash_matches = 0
    encoding_matches = 0
    
    for i, entry1 in enumerate(valid_entries):
        if entry1["index"] in processed_indices:
            continue
            
        current_group = [entry1]
        processed_indices.add(entry1["index"])
        
        for j in range(i + 1, len(valid_entries)):
            entry2 = valid_entries[j]
            if entry2["index"] in processed_indices:
                continue
                
            total_comparisons += 1
            
            # First check: exact hash match
            hash_distance = entry1["phash"] - entry2["phash"]
            if hash_distance == 0:  # Exact hash match
                hash_matches += 1
                
                # Second check: encoding similarity for exact duplicates
                embedding_distance = np.linalg.norm(entry1["embedding"] - entry2["embedding"])
                if embedding_distance <= 0.05:  # Very strict threshold for exact duplicates
                    encoding_matches += 1
                    current_group.append(entry2)
                    processed_indices.add(entry2["index"])
        
        if len(current_group) > 1:
            duplicate_groups.append(current_group)
        
        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"Progress: {i+1}/{len(valid_entries)} entries processed")
    
    print(f"✅ Finished! {len(duplicate_groups)} exact duplicate groups found")
    print(f"📊 Statistics:")
    print(f"   - Total comparisons: {total_comparisons:,}")
    print(f"   - Hash matches (exact): {hash_matches}")
    print(f"   - Encoding matches (≤0.05): {encoding_matches}")
    print(f"   - Final duplicates: {sum(len(group) for group in duplicate_groups)}")
    
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
    print("ℹ️ Hinweis: Das Tool sucht jetzt nach exakten Duplikaten")
    print("   - Hash: Exakte Übereinstimmung (Hamming-Distanz = 0)")
    print("   - Encoding: Sehr ähnlich (Distanz ≤ 0.05)")
    
    # Strategie abfragen
    while True:
        strategy = input("Welchen Eintrag pro Gruppe behalten? (first/last): ").lower()
        if strategy in ["first", "last"]:
            break
        print("Bitte 'first' oder 'last' eingeben.")
    
    return None, strategy  # threshold wird nicht mehr verwendet

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
        print("⚠️ Hinweis: Bei exakter Duplikatsuche wird der Schwellenwert ignoriert")
    else:
        strategy = "first"
        print(f"🔧 Standard-Einstellungen: Exakte Duplikate, Strategie={strategy}")
    
    # Duplikate finden (jetzt mit exakter Suche)
    duplicate_groups = find_exact_duplicates(database)
    
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
