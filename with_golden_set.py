import os
import glob
from typing import Union

import json
import click
import urllib.request

from utils.io import save_json


def load_json(
    json_path: str = None,
    json_url: str = None,
) -> Union[dict, list]:
    try:
        data = urllib.request.urlopen(json_url)
    except (AttributeError, FileNotFoundError):
        data = open(json_path)
    return json.load(data)


def add_key(data: dict, key_name: str = "golden_set"):
    for key, val in data.items():
        val[key_name] = {"level": False,"fmso": False}


def main():
    jsons = glob.glob("/mnt/ebs/data/kpf/page*/_0.0.1_simple.json")
    for json_path in jsons:
        data = load_json(json_path=json_path)
        add_key(data)
        save_json(json_path, data)


if __name__ == "__main__":
    main()
