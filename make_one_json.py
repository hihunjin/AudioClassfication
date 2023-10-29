import os
import glob
from typing import Union

import json
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


def main():
    all_data = {}
    jsons = glob.glob("/mnt/ebs/data/kpf*/page*/_0.0.2_simple.json")
    jsons.append("/mnt/ebs/data/kpf/all_231022_golden.json")
    jsons.sort(key=os.path.getmtime)
    for json_path in jsons:
        print("json_path", json_path)
        data = load_json(json_path=json_path)
        all_data.update(data)
        print(len(all_data))
    
    save_json("/mnt/ebs/data/all_231027.json", all_data)


if __name__ == "__main__":
    main()
