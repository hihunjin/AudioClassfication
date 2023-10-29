import os
import glob
from typing import Union

import json
import click
import urllib.request

from utils.io import save_json


JSON_NAME = "_0.0.1.json"
NEW_JSON_NAME = "_0.0.1_simple.json"
URL = "https://tadmins.tunegem.io/voai/{JSON_NAME}?page={page}"  # page: 1 ~ 70


def load_json(
    json_path: str = None,
    json_url: str = None,
) -> Union[dict, list]:
    try:
        data = urllib.request.urlopen(json_url)
    except (AttributeError, FileNotFoundError):
        data = open(json_path)
    return json.load(data)


def add_goldenset(d: dict, golden_set: dict):
    d["golden_set"] = {key: value == "y" for key, value in golden_set.items()}
    return d

def main():
    json_path = glob.glob("/mnt/ebs/data/kpf/all_231022.json")[0]
    data = load_json(json_path=json_path)

    for i in range(1, 100):
        try:
            new_data = load_json(json_url=URL.format(JSON_NAME=JSON_NAME, page=i))
        except:
            break
        for key in new_data.keys():
            if key in data.keys():
                data[key] = add_goldenset(data[key], new_data[key]["golden_set"])
                if True in data[key]["golden_set"].values():
                    print(key)

    save_json("/mnt/ebs/data/kpf/all_231022_golden.json", data)

if __name__ == "__main__":
    main()
