import re
from typing import Optional
from urllib.parse import urlparse
import requests


class WrongSolutionException(Exception):
    pass


class ParsingException(Exception):
    pass


class WebsiteInterface:
    def __init__(self, url: str, params: Optional[dict] = None) -> None:
        self.url = url
        self.session = requests.session()
        self.initial_response_text = self.session.post(self.url, params=params).text
        self.task = self.get_field_by_regex(r"var task = '([^']*)'")
        self.width = int(self.get_field_by_regex(r'name="w" value="(\d+)"'))
        self.height = int(self.get_field_by_regex(r'name="h" value="(\d+)"'))
        self.param = self.get_field_by_regex(r'name="param" value="([^"]*)"')
        self.size = self.get_field_by_regex(r'name="size" value="([^"]*)"')
        self.b = self.get_field_by_regex(r'name="b" value="([^"]*)"')
        try:
            self.puzzle_id = self.get_field_by_regex(r'<span id="puzzleID">([0-9,]+)</span>')
        except ParsingException:
            self.puzzle_id = None

    def get_field_by_regex(self, regex: str) -> str:
        return self.extract_by_regex_from_text(regex, self.initial_response_text)

    @staticmethod
    def extract_by_regex_from_text(regex: str, text: str) -> str:
        result = re.search(regex, text)
        if result:
            return result.group(1)
        raise ParsingException()

    def submit_solution(self, solution: str, email_address: Optional[str] = None):
        data = {
            "ansH": solution,
            "robot": 1,
            "b": self.b,
            "size": self.size,
            "param": self.param,
            "ready": "+++Done+++",
        }
        response = self.session.post(self.url, data)
        result = self.extract_by_regex_from_text(r'<div id="ajaxResponse"><p class="[^"]*">([^<]*)</p>', response.text)
        if not result.startswith("Congratulations!"):
            raise WrongSolutionException(result)

        print(result)

        try:
            if not email_address:
                with open(".email_address") as file_obj:
                    email_address = file_obj.read()

            solparams = self.extract_by_regex_from_text(r'name="solparams" value="([^"]*)"', response.text)
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
