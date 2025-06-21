# Production Line Quality Control System
https://github.com/ahmed-saleh111/Factory-Production-Line/blob/main/bottles.gif
## Overview

This system provides automated quality control for production line inspection using computer vision and AI analysis. It detects bottles on a production line, tracks them, and uses Google's Gemini AI to analyze whether labels are present and check for damage.

## Features

- **Real-time Object Detection**: Uses YOLOv8 for detecting and tracking bottles
- **ROI-based Processing**: Only processes objects that cross a specific region of interest
- **AI-powered Analysis**: Leverages Google Gemini AI to analyze bottle labels and detect damage
- **Automated Cropping**: Saves cropped images of detected bottles for analysis
- **Structured Reporting**: Generates analysis reports in both visual and text formats
- **Multi-threading**: Processes analysis in separate threads for better performance

## System Requirements

- Python 3.8 or higher
- OpenCV-capable system
- Internet connection for Google Gemini AI API
- Sufficient disk space for cropped images and reports

## Setup Environment

### 1. Clone or Download the Project
```bash
# If using git
git clone <repository-url>
cd production_line

# Or download and extract the files to production_line folder
```

### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv production_env

# Activate virtual environment
# On Windows:
production_env\Scripts\activate
# On macOS/Linux:
source production_env/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install opencv-python
pip install ultralytics
pip install cvzone
pip install langchain-google-genai
pip install python-dotenv
pip install numpy
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root directory:
```bash
GOOGLE_API_KEY="your_google_api_key_here"
```

**To get Google API Key:**
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the API key and paste it in the `.env` file

### 5. Required Files
Ensure these files are present in the project directory:
- `best.pt` - YOLOv8 trained model file
- `vid4.mp4` - Input video file for processing
- `.env` - Environment variables file

## Usage

### Basic Usage
```bash
python main.py
```

### Program Flow
1. **Initialization**: Loads YOLO model and opens video file
2. **Detection**: Processes each frame to detect bottles
3. **Tracking**: Tracks bottles across frames using unique IDs
4. **ROI Analysis**: Only processes bottles that cross the detection line (x=491)
5. **Cropping**: Saves cropped images of detected bottles
6. **AI Analysis**: Sends crops to Google Gemini for label and damage analysis
7. **Reporting**: Saves analysis results as text files

### Output Structure
```
production_line/
├── crop_YYYY-MM-DD/           # Daily folder for cropped images
│   ├── trackID_timestamp.jpg  # Cropped bottle images
│   └── trackID_timestamp.txt  # Analysis reports
├── main.py
├── best.pt
├── vid4.mp4
└── .env
```

### Sample Analysis Report
Each processed bottle generates a report file containing:
```
Track ID: 15
Date: 2025-06-19 14:30:25

| Label Present | Damage |
|--------------|--------|
| Yes          | No     |
```

## Configuration

### ROI Detection Line
- **Location**: x-coordinate 491 (vertical line)
- **Offset**: ±8 pixels tolerance
- **Purpose**: Only bottles crossing this line are processed

### Modify Detection Parameters
```python
# Change detection line position
cx1 = 491  # x-coordinate of detection line

# Change tolerance
offset = 8  # pixels
```

### Video Input
Replace `vid4.mp4` with your video file:
```python
cap = cv2.VideoCapture('your_video_file.mp4')
```

## Controls

- **'q' key**: Quit the application
- **ESC**: Close video window
- **Red line**: Visual indicator of the detection zone

## Troubleshooting

### Common Issues

1. **"Error: Could not open video file"**
   - Ensure `vid4.mp4` exists in the project directory
   - Check video file format compatibility

2. **Google API Key Error**
   - Verify API key is correct in `.env` file
   - Ensure Google Generative AI API is enabled
   - Check internet connection

3. **YOLO Model Error**
   - Ensure `best.pt` model file exists
   - Verify model file is not corrupted

4. **Dependencies Error**
   - Install all required packages using pip
   - Activate virtual environment before running

### Performance Tips

- Use SSD storage for faster I/O operations
- Ensure sufficient RAM for video processing
- Close unnecessary applications during processing
- Use GPU-enabled PyTorch for better YOLO performance

## File Descriptions

- **main.py**: Main application script
- **best.pt**: Pre-trained YOLOv8 model for bottle detection
- **vid4.mp4**: Input video file for processing
- **.env**: Environment variables (API keys)
- **crop_YYYY-MM-DD/**: Daily folders containing processed images and reports

## API Integration

The system uses Google Gemini 1.5 Flash model for image analysis. The analysis prompt checks for:
1. Label presence (Yes/No)
2. Damage detection (Yes/No)

Results are formatted in a structured table format for easy parsing and reporting.

## License

This project is for production line quality control purposes. Ensure compliance with your organization's data handling and AI usage policies.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure all required files are present
4. Check Google API key validity
