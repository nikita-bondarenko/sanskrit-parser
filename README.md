# Sanskrit OCR - Russian Diacritic Helper

🕉️ **Neural network-powered OCR system for Sanskrit text recognition and conversion to Russian diacritics**

## 🚀 Features

- **Neural Network OCR**: Custom PyTorch model for Sanskrit text recognition
- **Multi-format Support**: Handles IAST and Gaura PT input formats
- **Russian Diacritics Output**: Converts all recognized text to Russian diacritics
- **Modern UI**: Beautiful Preact frontend with Tailwind CSS
- **Docker Compose**: Easy deployment with containerized services
- **Real-time Processing**: Fast image processing and text recognition

## 📋 Supported Formats

### Input Formats:
- **IAST** (International Alphabet of Sanskrit Transliteration): `Śrī Kṛṣṇa`
- **Gaura PT**: `Çrï Këñëa`
- **Mixed text**: Automatically detects and converts

### Output Format:
- **Russian Diacritics**: `Ш́рӣ Кр̣ш̣н̣а`

## 🛠️ Technology Stack

### Backend:
- **Python 3.11** with FastAPI
- **PyTorch** for neural network
- **OpenCV** for image processing
- **Pillow** for image manipulation
- **Uvicorn** ASGI server

### Frontend:
- **Preact** with TypeScript
- **Tailwind CSS** for styling
- **Vite** for development and building
- **Modern drag-and-drop interface**

## 🐳 Quick Start with Docker Compose

### Prerequisites:
- Docker and Docker Compose installed
- At least 4GB RAM available

### 1. Clone and Start:
```bash
git clone <repository-url>
cd sanskrit-ocr
docker-compose up --build
```

### 2. Access the Application:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### 3. Usage:
1. Open http://localhost:3000 in your browser
2. Drag and drop a Sanskrit text image or click to select
3. Click "Process Image" to run OCR
4. Copy the recognized Russian diacritic text

## 📁 Project Structure

```
sanskrit-ocr/
├── docker-compose.yml          # Docker Compose configuration
├── backend/                    # Python FastAPI backend
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                # FastAPI application
│   └── models/                # Neural network models
├── frontend/                   # Preact TypeScript frontend
│   ├── Dockerfile
│   ├── package.json
│   ├── src/
│   │   ├── main.tsx           # Entry point
│   │   ├── app.tsx            # Main component
│   │   └── index.css          # Tailwind CSS
│   ├── tailwind.config.js
│   └── vite.config.ts
└── README.md
```

## 🔧 Development

### Backend Development:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development:
```bash
cd frontend
npm install
npm run dev
```

## 🧠 Neural Network Architecture

The OCR system uses a custom CNN architecture:

```
Input (64x256 grayscale) 
    ↓
Conv2D (32 filters) → MaxPool2D
    ↓
Conv2D (64 filters) → MaxPool2D
    ↓
Conv2D (128 filters) → MaxPool2D
    ↓
Fully Connected (512 units)
    ↓
Fully Connected (256 units)
    ↓
Output (200 character classes)
```

## 📊 Character Mappings

### IAST → Russian Diacritics:
- `ā` → `а̄` (long a)
- `ī` → `ӣ` (long i)
- `ū` → `ӯ` (long u)
- `ṛ` → `р̣` (vocalic r)
- `ṇ` → `н̣` (retroflex n)
- `ś` → `ш́` (palatal s)
- `ṣ` → `ш̣` (retroflex s)

### Gaura PT → Russian Diacritics:
- `ä` → `а̄` (long a)
- `ï` → `ӣ` (long i)
- `ü` → `ӯ` (long u)
- `ç` → `ш́` (palatal s)
- `ñ` → `н̣` (retroflex n)

## 🔍 API Endpoints

### Health Check:
```bash
GET /health
```

### OCR Processing:
```bash
POST /ocr
Content-Type: multipart/form-data
Body: file (image file)
```

Response:
```json
{
  "success": true,
  "text": "экона̄виṃш́е виṃш́атиме...",
  "image_info": {
    "width": 800,
    "height": 200,
    "mode": "RGB"
  }
}
```

## 🎯 Example Usage

### Sample Input (IAST):
```
Śrī Bhagavad-gītā
```

### Sample Output (Russian Diacritics):
```
Ш́рӣ Бхагавад-гӣта̄
```

## 🔧 Configuration

### Environment Variables:
- `VITE_API_URL`: Backend API URL (default: http://localhost:8000)
- `PYTHONUNBUFFERED`: Python output buffering (set to 1)

### Docker Compose Services:
- **backend**: FastAPI server on port 8000
- **frontend**: Preact development server on port 3000

## 📈 Performance

- **Image Processing**: ~1-3 seconds per image
- **Memory Usage**: ~200-400MB per service
- **Supported Formats**: JPG, PNG, WEBP, BMP
- **Max Image Size**: 10MB recommended

## 🛡️ Security

- CORS enabled for frontend communication
- File type validation
- Image size limits
- Error handling and logging

## 🚀 Deployment

### Production Build:
```bash
# Build frontend for production
cd frontend
npm run build

# Use production Docker images
docker-compose -f docker-compose.prod.yml up
```

### System Requirements:
- **CPU**: Multi-core processor recommended
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB for Docker images
- **Network**: Internet connection for initial setup

## 📝 License

MIT License - see LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
- Create an issue in the repository
- Check the API documentation at `/docs`
- Review the Docker logs: `docker-compose logs`

---

**Made with ❤️ for Sanskrit scholars and enthusiasts** 