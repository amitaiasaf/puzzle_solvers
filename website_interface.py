import re
from urllib.parse import urlparse
import requests


class WrongSolutionException(Exception):
    pass

class WebsiteInterface:
    def __init__(self, url: str, params: dict = None) -> None:
        self.url = url
        self.session = requests.session()
        self.initial_response_text = self.session.post(self.url, params=params).text
        self.task = re.search(r"var task = '([^']*)'", self.initial_response_text).group(1)
        self.width = int(re.search(r'name="w" value="(\d+)"', self.initial_response_text).group(1))
        self.height = int(re.search(r'name="h" value="(\d+)"', self.initial_response_text).group(1))
        self.param = re.search(r'name="param" value="([^"]*)"', self.initial_response_text).group(1)
        self.size = re.search(r'name="size" value="([^"]*)"', self.initial_response_text).group(1)
        self.b = re.search(r'name="b" value="([^"]*)"', self.initial_response_text).group(1)

    def submit_solution(self, solution: str, email_address: str = None):
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
        if not result.startswith("Congratulations!"):
            raise WrongSolutionException(result)

        print(result)

        try:
            if not email_address:
                with open(".email_address") as file_obj:
                    email_address = file_obj.read()

            solparams = re.search(r'name="solparams" value="([^"]*)"', response.text).group(1)
            data = {
                "email": email_address,
                "robot": 1,
                "submitscore": 1,
                "solparams": solparams,
            }

            hall_submit_url = urlparse(self.url)._replace(path="hallsubmit.php").geturl()
            response = self.session.post(hall_submit_url, data)
        except:
            print("Failed to submit score")
