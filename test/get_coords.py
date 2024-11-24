import pyautogui
import webbrowser
import time
from pynput.mouse import Listener

# URL to open
url = "https://sso.rajasthan.gov.in/signin?encq=m0ZUFHLqc4t+0vQu27K7jl5cOBbodS7JFafFdflRFZs="

# Open the URL in the default browser
webbrowser.open(url)
print("Navigate to the browser window and click anywhere to get the coordinates.")
print("Press Ctrl+C to stop the program.\n")

# Function to be called when a mouse click is detected
def on_click(x, y, button, pressed):
    if pressed:  # When the mouse button is pressed
        print(f"Mouse clicked at: ({x}, {y})")
        
# Set up the listener
with Listener(on_click=on_click) as listener:
    listener.join()  # Keep the listener running
