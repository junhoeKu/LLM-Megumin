## file_utils.py
## 리스트를 json 파일 형태로 저장하는 코드입니다.

import json

## json 파일 형태로 저장하는 함수
def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=0)
