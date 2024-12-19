## post_scraper.py
## 포스트 내에서 제목과 내용 추출하는 코드입니다.
  
from selenium.webdriver.common.by import By
import time

## 제목과 내용 부분 크콜링하는 함수
def get_post_data(driver, url):
    try:
        driver.get(url)
        time.sleep(2)
        title = driver.find_element(By.CLASS_NAME, 'tit_h3').text
        content = driver.find_element(By.ID, 'viewTypeSelector').text
        return {"title": title, "content": content}
    except Exception as e:
        print(f"Error crawling post {url}: {e}")
        return None
