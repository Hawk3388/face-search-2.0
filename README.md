# Face Search 2.0 🔍👤

An advanced face recognition system that crawls websites (primarily Wikipedia) to build a database of face embeddings and provides a web interface for searching similar faces.

## Features ✨

- **Multi-Source Web Crawling**: Specialized crawlers for Wikipedia, general URLs, and Common Crawl data
- **Advanced Face Recognition**: Uses dlib's state-of-the-art face recognition with 128-dimensional embeddings
- **Duplicate Detection**: Implements perceptual hashing (pHash) to identify and remove duplicate images
- **Interactive Search Interface**: Streamlit-based web app for uploading images and finding similar faces
- **Ethical Crawling**: Respects robots.txt, implements rate limiting, and follows best practices
- **Resume Functionality**: Continue crawling from where you left off after interruptions
- **Automatic Backups**: Periodic database backups to prevent data loss
- **Database Management**: Tools for cleaning and deduplicating the face database

## How It Works 🔧

1. **Web Crawling**: Automatically crawls websites (primarily Wikipedia) to collect face images
2. **Face Detection**: Locates faces in images using dlib
3. **Feature Extraction**: Generates 128-dimensional face embeddings for each detected face
4. **Duplicate Removal**: Uses perceptual hashing to identify and remove duplicate images
5. **Database Storage**: Saves embeddings with metadata to JSON database
6. **Face Search**: Upload an image to find similar faces using Euclidean distance matching

## Installation 📦

Clone this repository:

```bash
git clone https://github.com/yourusername/face-search-2.0.git
cd face-search-2.0
```

Install all requirements:

```bash
pip install -r requirements.txt
```

## Usage 🚀

### Web Interface

Launch the face search web application:

```bash
streamlit run main.py
```

Navigate to `http://localhost:8501` and upload an image to search for similar faces in the database.

### Building the Database

#### Wikipedia Crawler (Recommended)

Crawls Wikipedia's "Living People" category:

```bash
python crawler_wiki.py
```

#### General Website Crawler

Crawl any website for face images:

```bash
python crawler_url.py
```

#### Common Crawl Crawler

Use archived web data from Common Crawl:

```bash
python common_crawler.py
```

### Database Management

#### Clean Duplicates

Remove duplicate images using perceptual hashing:

```bash
python clean_db.py
```

#### Performance Testing

Benchmark duplicate detection performance:

```bash
python benchmark_hash.py
```

## Supported Image Formats 📸

- `.jpg`, `.jpeg`
- `.png`
- `.webp`
- `.gif`
- `.bmp`
- `.tiff`

## Technical Details 🔬

### Dependencies

- **Python 3.8+**: Required for all libraries
- **face_recognition**: Face detection and encoding
- **Streamlit**: Web interface
- **PIL/Pillow**: Image processing
- **requests**: Web crawling
- **imagehash**: Perceptual hashing for duplicates

### Database Format

The face database stores entries in this format:

```json
[
  {
    "image_url": "https://example.com/image.jpg",
    "page_url": "https://example.com/person",
    "embedding": [0.1, -0.2, 0.3, ...],
    "phash": "a1b2c3d4e5f6g7h8"
  }
]
```

### Algorithm Overview

- **Face Recognition**: Uses dlib's 128-dimensional face encodings
- **Similarity Matching**: Euclidean distance comparison with configurable tolerance
- **Duplicate Detection**: Perceptual hashing with configurable thresholds

## Privacy & Ethics ⚠️

This tool is designed for legitimate research and educational purposes. Please:

- Respect privacy and obtain proper consent when searching for people
- Follow applicable laws and regulations in your jurisdiction
- Use responsibly and ethically
- Only use publicly available images
- Be aware that results may include false positives

The project implements responsible web crawling:

- **Robots.txt Compliance**: Checks and respects robots.txt files
- **Rate Limiting**: 1-second delays between requests
- **User-Agent**: Proper bot identification
- **Educational Purpose**: Designed for research and learning

## Project Structure 📁

```sh
face-search-2.0/
├── main.py                 # Streamlit web interface
├── crawler_wiki.py         # Wikipedia "Living People" crawler
├── crawler_url.py          # General website crawler
├── common_crawler.py       # Common Crawl data crawler
├── clean_db.py            # Database cleaning and deduplication
├── benchmark_hash.py      # Performance testing for duplicate detection
├── face_embeddings.json   # Main face database
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── images/               # Sample images for testing
```

## Configuration ⚙️

### Crawler Settings

- `CRAWL_DELAY = 1.0`: Delay between requests (seconds)
- `MAX_CONSECUTIVE_429 = 10`: Stop after consecutive rate limit errors
- `model="large"`: Use high-accuracy face recognition model

### Duplicate Detection

- `threshold=5`: Similarity threshold for perceptual hashing (0-64)
- `strategy="first"`: Which duplicate to keep ("first" or "last")

## Contributing 🤝

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Use of AI

This project was built with AI assistance, including code generation and parts of this README file.

## License 📄

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

## Credits 🙏

- **face_recognition**: Adam Geitgey's face recognition library
- **dlib**: Machine learning library for face detection
- **Streamlit**: Web app framework
- **Pillow**: Python Imaging Library
