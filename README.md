1️⃣ Project title
# Vehicle Speed Detection and Overspeeding Identification

2️⃣ Project description
This project detects vehicles from a video, tracks their movement, estimates their speed, 
and identifies overspeeding vehicles using OpenCV and Dlib. 
For overspeeding vehicles, number plate recognition is performed using an external API.

3️⃣ Features
## Features
- Vehicle detection using Haar Cascade
- Multi-vehicle tracking using Dlib correlation tracker
- Speed estimation in km/hr
- Overspeeding vehicle identification
- Automatic vehicle image capture
- Number plate recognition using Plate Recognizer API

4️⃣ Technologies used
## Technologies Used
- Python 3
- OpenCV
- Dlib
- Haar Cascade Classifier
- Plate Recognizer API
- VS Code

5️⃣ Project folder structure
## Project Structure

hackthon/
│
├── overspeed.py
├── myhaar.xml
├── short-test.mp4
├── images/
├── list.txt
├── requirements.txt
└── README.md

6️⃣ Installation steps
## Installation

1. Install Python 3.x from https://www.python.org  
2. Open the project folder in VS Code  
3. Open terminal in VS Code (Ctrl + `)  
4. Install required libraries:

pip install -r requirements.txt


If dlib fails on Windows:

pip install cmake
pip install dlib-bin

7️⃣ How to run the project
## How to Run

1. Make sure `short-test.mp4` and `myhaar.xml` are in the project folder  
2. Open `overspeed.py`  
3. Select Python interpreter:
   - Ctrl + Shift + P → Python: Select Interpreter → Python 3.x  
4. Run the file:

python overspeed.py

5. Press ESC to stop the video

8️⃣ Configuration
## Configuration

- SPEED_LIMIT can be changed in the code
- Video path can be updated for different input videos
- API_KEY must be valid for number plate recognition

9️⃣ Output
## Output

- Vehicles are detected with bounding boxes
- Speed is displayed on video
- Overspeeding vehicles are identified
- Cropped images of vehicles are stored in `images/` folder

🔟 Future enhancements
## Future Enhancements
- Use YOLO for better detection accuracy
- Live CCTV camera integration
- Database storage for violations
- Helmet and seatbelt detection
