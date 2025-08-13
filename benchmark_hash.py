import time
import json
import imagehash
from PIL import Image
import random
import string
from clean_db import find_hash_duplicates, str_to_phash

def generate_test_data(count=40000):
    """Generiert Testdaten mit echten phash-Werten"""
    print(f"📝 Generiere {count} Test-Einträge...")
    
    test_data = []
    
    # Generiere verschiedene Basis-Hashes
    base_hashes = []
    for i in range(min(1000, count // 40)):  # Etwa 40 ähnliche Bilder pro Basis-Hash
        # Erstelle einen zufälligen 64-bit Hash
        random_hash = ''.join(random.choices('0123456789abcdef', k=16))
        base_hashes.append(random_hash)
    
    for i in range(count):
        if i % 5000 == 0:
            print(f"Fortschritt: {i}/{count}")
        
        # Wähle einen Basis-Hash
        base_hash = random.choice(base_hashes)
        
        # Erstelle leichte Variationen (für Duplikate) oder komplett neue Hashes
        if random.random() < 0.1:  # 10% Chance auf Duplikat
            # Leichte Variation des Basis-Hash (1-3 Bit Unterschied)
            hash_int = int(base_hash, 16)
            # Flippe 1-3 zufällige Bits
            for _ in range(random.randint(1, 3)):
                bit_pos = random.randint(0, 63)
                hash_int ^= (1 << bit_pos)
            varied_hash = f"{hash_int:016x}"
        else:
            # Komplett neuer Hash
            varied_hash = ''.join(random.choices('0123456789abcdef', k=16))
        
        entry = {
            "image_url": f"https://example.com/image_{i}.jpg",
            "page_url": f"https://example.com/page_{i}",
            "phash": varied_hash,
            "embedding": [random.random() for _ in range(128)]  # Dummy embedding
        }
        test_data.append(entry)
    
    print(f"✅ {count} Test-Einträge generiert")
    return test_data

def benchmark_performance():
    """Führt Performance-Tests durch"""
    sizes = [1000, 5000, 10000, 20000, 40000]
    
    for size in sizes:
        if size > 40000:
            break
            
        print(f"\n{'='*50}")
        print(f"🚀 BENCHMARK MIT {size} EINTRÄGEN")
        print(f"{'='*50}")
        
        # Generiere Testdaten
        test_data = generate_test_data(size)
        
        # Zeitmessung
        start_time = time.time()
        
        # Führe Duplikat-Suche durch
        duplicates = find_hash_duplicates(test_data, hash_threshold=5)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"⏱️  Zeit: {duration:.2f} Sekunden")
        print(f"📊 Gefundene Duplikat-Gruppen: {len(duplicates)}")
        print(f"🔄 Vergleiche pro Sekunde: {(size * (size-1) / 2) / duration:.0f}")
        
        # Hochrechnung für 40k falls noch nicht erreicht
        if size < 40000:
            # O(n²) Komplexität
            estimated_40k = duration * (40000 / size) ** 2
            print(f"📈 Geschätzte Zeit für 40.000: {estimated_40k:.1f} Sekunden ({estimated_40k/60:.1f} Minuten)")

def quick_40k_estimate():
    """Schnelle Schätzung nur für 40k"""
    print("🔍 SCHNELLE 40K SCHÄTZUNG")
    print("="*30)
    
    # Teste mit kleinerer Menge für Hochrechnung
    test_size = 2000
    print(f"Teste mit {test_size} Einträgen für Hochrechnung...")
    
    test_data = generate_test_data(test_size)
    
    start_time = time.time()
    duplicates = find_hash_duplicates(test_data, hash_threshold=5)
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"⏱️  Zeit für {test_size}: {duration:.2f} Sekunden")
    
    # Hochrechnung auf 40k (O(n²))
    factor = (40000 / test_size) ** 2
    estimated_40k = duration * factor
    
    print(f"📈 Geschätzte Zeit für 40.000 Einträge:")
    print(f"   🕐 {estimated_40k:.1f} Sekunden")
    print(f"   🕐 {estimated_40k/60:.1f} Minuten")
    print(f"   🕐 {estimated_40k/3600:.1f} Stunden")

if __name__ == "__main__":
    choice = input("Vollständiger Benchmark (v) oder schnelle 40k Schätzung (s)? [s]: ").lower()
    
    if choice == 'v':
        benchmark_performance()
    else:
        quick_40k_estimate()
