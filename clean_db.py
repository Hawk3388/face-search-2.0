import json
import imagehash
from PIL import Image
from io import BytesIO
import requests
from collections import defaultdict
import shutil
import os
from datetime import datetime

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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"{original_file}.backup_{timestamp}"
    try:
        shutil.copy2(original_file, backup_file)
        print(f"✅ Backup erstellt: {backup_file}")
        return backup_file
    except Exception as e:
        print(f"❌ Fehler beim Erstellen des Backups: {e}")
        return None

def find_hash_duplicates(database, hash_threshold=5):
    """
    Findet Duplikate basierend auf phash-Vergleich
    
    Args:
        database: Liste der Datenbankeinträge
        hash_threshold: Maximale Hamming-Distanz für phash-Vergleich (Standard: 5)
    
    Returns:
        Liste von Duplikat-Gruppen, wobei jede Gruppe eine Liste ähnlicher Einträge ist
    """
    print(f"🔍 Suche nach Duplikaten mit Hash-Schwellenwert: {hash_threshold}")
    
    duplicate_groups = []
    processed_indices = set()
    
    total_entries = len(database)
    
    for i, entry1 in enumerate(database):
        if i in processed_indices:
            continue
            
        # Fortschritt anzeigen
        if i % 100 == 0:
            print(f"Fortschritt: {i}/{total_entries} ({i/total_entries*100:.1f}%)")
        
        # phash des aktuellen Eintrags laden
        phash1 = str_to_phash(entry1.get("phash", ""))
        if phash1 is None:
            continue
            
        current_group = [{"index": i, "entry": entry1}]
        processed_indices.add(i)
        
        # Mit allen anderen vergleichen
        for j, entry2 in enumerate(database[i+1:], start=i+1):
            if j in processed_indices:
                continue
                
            phash2 = str_to_phash(entry2.get("phash", ""))
            if phash2 is None:
                continue
            
            # Hamming-Distanz berechnen
            try:
                distance = phash1 - phash2
                if distance <= hash_threshold:
                    current_group.append({"index": j, "entry": entry2})
                    processed_indices.add(j)
            except Exception as e:
                print(f"Fehler beim Vergleichen der Hashes: {e}")
                continue
        
        # Nur Gruppen mit mehr als einem Eintrag sind Duplikate
        if len(current_group) > 1:
            duplicate_groups.append(current_group)
    
    print(f"✅ Fertig! {len(duplicate_groups)} Duplikat-Gruppen gefunden")
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
