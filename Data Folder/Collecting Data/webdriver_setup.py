## webdriver_setup.py
## webdriver 여는 코드입니다.
  
from selenium import webdriver

def setup_driver():
    driver = webdriver.Chrome()
    return driver
