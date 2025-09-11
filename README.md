# Face Search 🔍👤

**🚀 Live App:** [face-search.streamlit.app](https://face-search.streamlit.app)

Search for similar faces in a large database! Upload an image and find people who look alike.

## ✨ What the App Can Do

### 🎯 Face Recognition

- **Automatic Face Detection** - Finds all faces in your image
- **Multiple Faces** - If there are multiple people in the image, you can choose which one to search for
- **Smart Labeling** - Faces are marked with red boxes and labels

### 🔍 Face Search

- **Instant Results** - See similar faces in seconds
- **Similarity Score** - Shows how similar the found faces are
- **Image Gallery** - Shows the found faces as previews

### 🛡️ Security & Privacy

- **Secure Connection** - All data is transmitted encrypted
- **Local Mode** - You can also run the app offline on your computer
- **No Storage** - Your uploaded images are not stored

## 🚀 Quick Start

### Use Online (Recommended)

1. Go to [face-search.streamlit.app](https://face-search.streamlit.app)
2. Upload an image with a face (all uploaded images are deleted immediately after usage)
3. Wait for automatic face detection
4. Click "Start Search"
5. Look at the similar faces!

### Install Locally

```bash
# Download repository
git clone https://github.com/Hawk3388/face-search-2.0.git
cd face-search-2.0

# Install dependencies
pip install -r requirements.txt

# Start app
streamlit run main.py
```

## 📸 Supported Formats

- JPG, JPEG, PNG, WebP

## 🎮 How It Works

### Step 1: Upload Image

Upload a photo that shows at least one face.

### Step 2: Face Detection

The app automatically detects all faces in the image and marks them with red boxes.

### Step 3: Select Face (if multiple)

If multiple faces are found, you can choose which one to search for.

### Step 4: Start Search

The app searches the database for similar faces.

### Step 5: View Results

You see all found similar faces with similarity scores.

## 🏗️ Technical Database

The app uses a database with thousands of faces from public sources (mainly Wikipedia). The database for the web app contains **only living people** and is **updated daily** through:

- **Automatic Crawling** - Searches for new faces on the internet
- **Duplicate Check** - Prevents duplicate entries
- **Quality Control** - Only clear, well-recognizable faces are stored
- **Customizable** in the `crawler` folder you can see some examples to create your own database

**Note:** The database may not be complete, due to the changing of the category and the early stage of the project.

## 🤖 Custom Trained Model

This project uses a **custom trained face recognition model** that has been specifically trained for better accuracy with diverse face types and lighting conditions.

### Download Pre-trained Model

You can download the latest pre-trained model from the [GitHub Releases](https://github.com/Hawk3388/face-search-2.0/releases):

1. Go to the Releases page
2. Download the latest model file (`tinyfacenet_best.pth`)
3. Copy the downloaded file to the project root directory
4. The app will automatically use the custom model for better face recognition (PyTorch is required for this process)

### Model Benefits

- **Higher Accuracy** - Better recognition of diverse faces
- **Optimized Performance** - Faster processing while maintaining quality (GPU recommended for best speed)
- **Regular Updates** - New model versions released with improvements
- **CPU Compatible** - Works on CPU but may be slower than GPU version

**Note:** For optimal performance, use a GPU. The model will run on CPU but processing may be slower.

## ⚠️ Important Notes

### Privacy

- Only use images of people who have consented
- Respect the privacy of others
- The app is intended for education and research only

## 🤝 Contribute

Have ideas for improvement? Great!

1. Create an issue on GitHub
2. Describe your idea or the problem
3. We'll look at it together

## 📄 License

This project is licensed under the Apache License 2.0.

## 🙏 Thanks to

- **face_recognition** - For face recognition
- **Streamlit** - For the simple web app
- **Wikipedia Community** - For the public data
- **PyTorch** - For the custom trained model

---

**Made by Hawk3388** | **Crawler active since September 11, 2025**
