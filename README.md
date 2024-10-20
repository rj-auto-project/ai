# Description
Flipkart Grid 6.0

## How to use this Repo
1. Clone the repo
    ```bash
    git clone https://github.com/your-username/your-repository.git
    ```
2. Navigate to the project directory
    ```bash
    cd your-repository
    ```
3. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```
4. Run MediaMTX
    ```bash
    mediamtx
    ```
5. Open index.html in Chrome Browser.
6. inference.py is cofigured to use webcam by default, if you wan to use any other camera through rtsp or test it on a video, give the location of video in inference.py line number 131. A set video is also availbale in data directory of this project dir.
7. Run the Backend and streaming
    ```bash
    python inference.py
    ```
