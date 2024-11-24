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
from selenium.common.exceptions import TimeoutException

# Load environment variables (username, password, and receiver_id)
load_dotenv()

# Coordinates for captcha screenshot
CAPTCHA_COORDS = (1321, 499, 1657, 577)  # (x1, y1, x2, y2)

# Webpage details and credentials from environment variables
URL = "https://sso.rajasthan.gov.in/signin?encq=m0ZUFHLqc4t+0vQu27K7jl5cOBbodS7JFafFdflRFZs="
USERNAME = "RJJP201619037610"
PASSWORD = os.getenv("password")
MPIN = os.getenv("mpin")
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
    captcha_text = ''.join([char for char in captcha_p if char.isdigit()])
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
    if WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, CAPTCHA_FIELD_ID))):
        driver.refresh()
        login()

def navigate_to_table():
    e_file_btn = driver.find_element(By.ID, "zk_comp_405")
    e_file_btn.click()
    time.sleep(2)  # Add a slight wait to allow the table to load
    print("Navigated to file table.")

def select_file():
    try:
        table_div = driver.find_element("css selector",'td.z-cell')
        if table_div:
            table_div.click()
        else:
            while True:
                pass
    except:
        print("Automantion Stpped. Go Manual")
        while True:
            pass

def select_reciever(target_index):
    try:
        target_id = RECEIVER_IDS[target_index]
        print(target_id)
        row = driver.find_element("xpath", f"//tr[td/div/span[text()='{target_id}']]")
        radio_button = row.find_element("xpath", ".//input[@type='radio']")
        driver.execute_script("arguments[0].click()",radio_button)
    except:
        print("failed at selecting reviever")

def full_login():
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
            print(MPIN)
            driver.find_element(By.ID, "j_password").send_keys(MPIN)
            driver.find_element(By.ID, "grad").click()
            print("MPIN passed")
        else:
            print("wrong url")
    except TimeoutException:
        print("Page did not load within the expected time.")

full_login()

# file transmizzion start
receiver_index = 0
while True:
    # if error or login failed then refresh
    if "https://sso.rajasthan.gov.in/signin?" in driver.current_url:
        driver.refresh()
        time.sleep(2)
        full_login()
    time.sleep(2)
    # load data btn
    try:
        WebDriverWait(driver, 60).until(EC.text_to_be_present_in_element((By.XPATH, "//*"), " Load Data"))
        driver.find_element("xpath", f"//*[contains(text(), ' Load Data')]").click()                           # load data btn in efiles page
    except TimeoutException:
        print("Page did not load within the expected time.")

    # load file
    try:
        WebDriverWait(driver, 60).until(EC.text_to_be_present_in_element((By.XPATH, "//*"), 'E - File'))
        parent_container = driver.find_element(By.ID,"zk_comp_376")
        e_file_text = parent_container.find_elements("xpath", f"//*[contains(text(), 'E - File')]")
        file_num_element = e_file_text[1].find_element(By.XPATH,"following-sibling::*[3]")
        driver.execute_script("arguments[0].click()",file_num_element)
        print(file_num_element.tag_name)
    except TimeoutException:
        print("Page did not load within the expected time.")

    time.sleep(3)

    # select file and move
    try:
        WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'td.z-cell')))
        select_file()
        WebDriverWait(driver, 60).until(EC.text_to_be_present_in_element((By.XPATH, "//*"), 'Send To Anyone'))
        time.sleep(2)
        send_to_anyon_btn = driver.find_elements("xpath", f"//*[contains(text(), 'Send To Anyone')]")
        driver.execute_script("arguments[0].click()",send_to_anyon_btn[1])
    except TimeoutException:
        print("Page did not load within the expected time.")

    # select reciever  and send it
    try:
        time.sleep(2)
        if receiver_index >= len(RECEIVER_IDS):
            receiver_index = 0
        select_reciever(receiver_index)
        driver.find_element("xpath", f"//*[contains(text(), 'Proceed')]").click()
        driver.find_element("xpath", f"//*[contains(text(), 'Proceed without DSC/e Sign')]").click()
        final_ok_btn= driver.find_element(By.ID, "zk_comp_1036")
        driver.execute_script("arguments[0].click()",final_ok_btn)
        receiver_index = receiver_index + 1
    except TimeoutException:
        print("failed at sending file step")