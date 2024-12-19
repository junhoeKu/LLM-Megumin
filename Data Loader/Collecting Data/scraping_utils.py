## scraping_utils.py
## selenium 동적 크롤링 관련 코드입니다.

from selenium.webdriver.common.by import By
import time

## 스크롤을 아래로 끝까지 내리는 함수
def scroll_to_bottom(driver):
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height


## 각 포스트의 url을 추출하는 함수
def extract_post_urls(driver):
    posts = driver.find_elements(By.CLASS_NAME, 'card__reUkU')
    post_urls = []
    for post in posts:
        try:
            url = post.find_element(By.TAG_NAME, 'a').get_attribute('href')
            post_urls.append(url)
        except Exception as e:
            print(f"Error extracting URL from post: {e}")
    return post_urls
