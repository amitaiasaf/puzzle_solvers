import re
import requests


class WebsiteInterface:
    def __init__(self, url: str) -> None:
        self.url = url
        self.session = requests.session()
        response = self.session.get(self.url)
        self.task = re.search(r"var task = '([^']*)'", response.text).group(1)
        self.width = int(re.search(r'name="w" value="(\d+)"', response.text).group(1))
        self.height = int(re.search(r'name="h" value="(\d+)"', response.text).group(1))
        self.param = re.search(r'name="param" value="([^"]*)"', response.text).group(1)
        self.size = re.search(r'name="size" value="([^"]*)"', response.text).group(1)
        self.b = re.search(r'name="b" value="([^"]*)"', response.text).group(1)

    def submit_solution(self, solution: str):
        data = {
            "ansH": solution,
            "robot": 1,
            "b": self.b,
            "size": self.size,
            "param": self.param,
            "ready": "+++Done+++",
        }
        response = self.session.post(self.url, data)
        result = re.search(r'<div id="ajaxResponse"><p class="[^"]*">([^<]*)</p>', response.text).group(1)
        print(result)
