import time
import pyautogui
import easyocr
import cv2
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from dotenv import load_dotenv
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
from selenium.common.exceptions import TimeoutException

# Load environment variables (username, password, and receiver_id)
load_dotenv()

# Coordinates for captcha screenshot
CAPTCHA_COORDS = (1321, 499, 1657, 577)  # (x1, y1, x2, y2)

# Webpage details and credentials from environment variables
URL = "https://sso.rajasthan.gov.in/signin?encq=m0ZUFHLqc4t+0vQu27K7jl5cOBbodS7JFafFdflRFZs="
USERNAME = "ashokmeena88.doit"
PASSWORD = os.getenv("password")
MPIN = os.getenv("MPIN")
print(USERNAME)
RECEIVER_IDS = os.getenv("reciever_id").split(",")  # Assuming receiver_id contains comma-separated values

# Element IDs on the webpage
USERNAME_FIELD_ID = "cpBodyMain_cpBody_txt_Data1"
PASSWORD_FIELD_ID = "cpBodyMain_cpBody_txt_Data2"
CAPTCHA_FIELD_ID = "cpBodyMain_cpBody_ssoCaptcha_txtCaptcha"
SUBMIT_BUTTON_ID = "cpBodyMain_cpBody_btn_Login"

# Path to save the screenshot
CAPTCHA_IMAGE_PATH = 'D:/test/data/otp.png'

# Initialize the Selenium WebDriver
driver = webdriver.Chrome()
driver.maximize_window()
driver.get(URL)
time.sleep(2)

# Capture screenshot for captcha and save it
def capture_screenshot(x1, y1, x2, y2, save_path):
    width = x2 - x1
    height = y2 - y1
    screenshot = pyautogui.screenshot(region=(x1, y1, width, height))
    screenshot.save(save_path)
    print(f"Screenshot saved at {save_path}")

def read_captcha(image_path="D:/test/data/otp.png"):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (323, 93))
    reader = easyocr.Reader(['en'])
    result = reader.readtext(img, threshold=0.2)
    captcha_p = ''.join([line[1] for line in result])
    captcha_text = re.sub(r"/D","",captcha_p)
    return captcha_text


captcha = ""
def get_captcha():
    global captcha
    while True:
        capture_screenshot(*CAPTCHA_COORDS,save_path=CAPTCHA_IMAGE_PATH)
        captcha = read_captcha("D:/test/data/otp.png").replace(" ","")
        print(captcha)
        if len(captcha) == 6 and captcha.isdigit():
            break
        else:
            driver.refresh()
            time.sleep(1)

def login():
    print(f"Logging in with username: {USERNAME}")
    get_captcha()
    initial_url = driver.current_url
    driver.find_element(By.ID, USERNAME_FIELD_ID).clear()
    driver.find_element(By.ID, USERNAME_FIELD_ID).send_keys(USERNAME)
    driver.find_element(By.ID, PASSWORD_FIELD_ID).clear()
    driver.find_element(By.ID, PASSWORD_FIELD_ID).send_keys(PASSWORD)
    driver.find_element(By.ID, CAPTCHA_FIELD_ID).clear()
    driver.find_element(By.ID, CAPTCHA_FIELD_ID).send_keys(captcha)
    print(USERNAME,PASSWORD)
    driver.find_element(By.ID, SUBMIT_BUTTON_ID).click()
    if "https://sso.rajasthan.gov.in/signin?encq=" in driver.current_url:
        driver.refresh()
        login()

if __name__ == "__main__":
    # 1st Login
    try:
        WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.ID, USERNAME_FIELD_ID)))
        login()
        time.sleep(2)
        print("login successfully!")
    except TimeoutException:
        print("Page did not load within the expected time.")

    # 2nd Login
    try:
        WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.ID, "cpBodyMain_cpBody_cbx_newsession")))
        if driver.current_url == "https://sso.rajasthan.gov.in/signin?ru=EOFFICE":
            driver.find_element(By.ID, "cpBodyMain_cpBody_cbx_newsession").click()
            driver.find_element(By.ID, "cpBodyMain_cpBody_txt_Data2").clear()
            driver.find_element(By.ID, "cpBodyMain_cpBody_txt_Data2").send_keys(PASSWORD)
        driver.find_element(By.ID, "cpBodyMain_cpBody_btn_Login").click()
        print("2nd login done successfully!")
    except TimeoutException:
        print("Page did not load within the expected time.")

    # MPIN auth
    try:
        WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.ID, "j_password")))
        if "https://rajeoffice.rajasthan.gov.in/dualAuthentication.zul" in driver.current_url:
            driver.find_element(By.ID, "j_password").send_keys(MPIN)
            driver.find_element(By.ID, "grad").click()
        else:
            print("wrong url")
    except TimeoutException:
        print("Page did not load within the expected time.")
    
    while True:
        pass