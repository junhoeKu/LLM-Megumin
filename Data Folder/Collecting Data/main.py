## main.py
## 실제 크롤링 수행 및 json 저장 코드입니다.

from webdriver_setup import setup_driver
from scraping_utils import scroll_to_bottom, extract_post_urls
from post_scraper import get_post_data
from file_utils import save_to_json
from tqdm import tqdm
  
def main():
    driver = setup_driver()
    
    ## blog url에 접속
    blog_url = 'https://m.blog.naver.com/PostList.nhn?blogId=znlrnldnpr&categoryName=%EC%BD%94%EB%85%B8%EC%8A%A4%EB%B0%94%21&categoryNo=12&listStyle=card&logCode=0&tab=1'
    driver.get(blog_url)
    
    ## 아래로 쭉 스크롤내리는 함수
    scroll_to_bottom(driver)

    ## 각 포스트의 url만 추출
    post_urls = extract_post_urls(driver)
    driver.quit()

    ## url 추출 후 실제 내용 크롤링을 위해 재접속
    driver = setup_driver()
    post_data = []
    
    ## url 순서대로 접속해서 제목, 내용 크롤링
    for url in tqdm(post_urls, desc="Crawling posts"):
        post_data.append(get_post_data(driver, url))
    post_data.reverse()
    driver.quit()

    ## json 파일 형태로 저장
    save_to_json(post_data, "코노스바_텍본.json")

if __name__ == "__main__":
    main()
