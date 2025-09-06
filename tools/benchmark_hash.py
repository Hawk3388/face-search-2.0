import time
import random
from clean_db import find_hash_duplicates

def generate_test_data(count=40000):
    """Generates test data with real phash values"""
    print(f"📝 Generating {count} test entries...")
    
    test_data = []
    
    # Generate various base hashes
    base_hashes = []
    for i in range(min(1000, count // 40)):  # About 40 similar images per base hash
        # Create a random 64-bit hash
        random_hash = ''.join(random.choices('0123456789abcdef', k=16))
        base_hashes.append(random_hash)
    
    for i in range(count):
        if i % 5000 == 0:
            print(f"Progress: {i}/{count}")
        
        # Choose a base hash
        base_hash = random.choice(base_hashes)
        
        # Create slight variations (for duplicates) or completely new hashes
        if random.random() < 0.1:  # 10% chance of duplicate
            # Slight variation of base hash (1-3 bit difference)
            hash_int = int(base_hash, 16)
            # Flip 1-3 random bits
            for _ in range(random.randint(1, 3)):
                bit_pos = random.randint(0, 63)
                hash_int ^= (1 << bit_pos)
            varied_hash = f"{hash_int:016x}"
        else:
            # Completely new hash
            varied_hash = ''.join(random.choices('0123456789abcdef', k=16))
        
        entry = {
            "image_url": f"https://example.com/image_{i}.jpg",
            "page_url": f"https://example.com/page_{i}",
            "phash": varied_hash,
            "embedding": [random.random() for _ in range(128)]  # Dummy embedding
        }
        test_data.append(entry)
    
    print(f"✅ {count} test entries generated")
    return test_data

def benchmark_performance():
    """Performs performance tests"""
    sizes = [1000, 5000, 10000, 20000, 40000]
    
    for size in sizes:
        if size > 40000:
            break
            
        print(f"\n{'='*50}")
        print(f"🚀 BENCHMARK WITH {size} ENTRIES")
        print(f"{'='*50}")
        
        # Generate test data
        test_data = generate_test_data(size)
        
        # Time measurement
        start_time = time.time()
        
        # Perform duplicate search
        duplicates = find_hash_duplicates(test_data, hash_threshold=5)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"⏱️  Time: {duration:.2f} seconds")
        print(f"📊 Found duplicate groups: {len(duplicates)}")
        print(f"🔄 Comparisons per second: {(size * (size-1) / 2) / duration:.0f}")
        
        # Projection for 40k if not yet reached
        if size < 40000:
            # O(n²) complexity
            estimated_40k = duration * (40000 / size) ** 2
            print(f"📈 Estimated time for 40,000: {estimated_40k:.1f} seconds ({estimated_40k/60:.1f} minutes)")

def quick_40k_estimate():
    """Quick estimation for 40k only"""
    print("🔍 QUICK 40K ESTIMATION")
    print("="*30)
    
    # Test with smaller amount for projection
    test_size = 2000
    print(f"Testing with {test_size} entries for projection...")
    
    test_data = generate_test_data(test_size)
    
    start_time = time.time()
    duplicates = find_hash_duplicates(test_data, hash_threshold=5)
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"⏱️  Time for {test_size}: {duration:.2f} seconds")
    
    # Projection to 40k (O(n²))
    factor = (40000 / test_size) ** 2
    estimated_40k = duration * factor
    
    print(f"📈 Estimated time for 40,000 entries:")
    print(f"   🕐 {estimated_40k:.1f} seconds")
    print(f"   🕐 {estimated_40k/60:.1f} minutes")
    print(f"   🕐 {estimated_40k/3600:.1f} hours")

if __name__ == "__main__":
    choice = input("Full benchmark (f) or quick 40k estimation (q)? [q]: ").lower()
    
    if choice == 'f':
        benchmark_performance()
    else:
        quick_40k_estimate()
