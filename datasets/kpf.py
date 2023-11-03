import os
from typing import Union

import json
import click
import urllib.request

from utils.helper_funcs import CategoryConverter


def load_json(
    json_path: str = None,
    json_url: str = None,
) -> Union[dict, list]:
    try:
        data = urllib.request.urlopen(json_url)
    except (AttributeError, FileNotFoundError):
        data = open(json_path)
    return json.load(data)


def convert_mp4_to_wav_save(mp4_path, save_directory):
    name = mp4_path.split(os.sep)[-1]
    out_name = os.path.join(save_directory, name) + ".wav"
    if not os.path.exists(out_name):
        os.system(
            'ffmpeg -i "{}" -hide_banner -loglevel error -acodec pcm_s16le -ar 16000 "{}"'.format(
                mp4_path, out_name
            )
        )
        print("saved location: ", out_name)
    else:
        print(f"File already exists : {out_name}")
    return out_name


def oxn2trn(_v: str):
    if _v == "O":
        return True
    elif _v == "X":
        return False
    elif _v == "N":
        return None
    else:
        return None


JSON_NAME = "_{version}.json"
NEW_JSON_NAME = "_{version}_simple.json"
URL = "https://tadmins.tunegem.io/voai/{JSON_NAME}?page={page}"  # page: 1 ~ 70


@click.command()
@click.option("--page", default=1, help="page number")
@click.option("--version", default="0.0.1", help="page number")
@click.option(
    "--save_directory",
    default="datasets/kpf",
    help="directory to save wav files",
)
def main(
    page: int,
    save_directory: str,
    version: str = "0.0.1",
):
    global JSON_NAME, NEW_JSON_NAME
    JSON_NAME = JSON_NAME.format(version=version)
    NEW_JSON_NAME = NEW_JSON_NAME.format(version=version)
    save_directory_page = os.path.join(save_directory, f"page{page}")
    os.makedirs(os.path.join(save_directory_page, "wav"), exist_ok=True)
    json_path = None
    data = load_json(json_path=json_path, json_url=URL.format(JSON_NAME=JSON_NAME, page=page))
    # get dictionary and reformat
    df_list = {}
    for key, val in data.items():
        # download data if not exists
        # if 62636 >= int(key):
        #     break
        saved_loc = convert_mp4_to_wav_save(
            val["mp4url"],
            os.path.join(save_directory_page, "wav"),
        )
        df_list[key] = {
            "key": int(key),
            "path": saved_loc,
            "fmso": list(val["time_interval"].keys())[0],  # FIXME
            "level": val["level"],
            "time_interval": val["time_interval"],
            "golden_set": {key: value == "y" for key, value in val["golden_set"].items()},
            "properties": {CategoryConverter.to_num(_k): oxn2trn(_v)for _k, _v in val["properties"].items()},
        }
    with open(os.path.join(save_directory_page, NEW_JSON_NAME), "w") as outfile:
        json.dump(df_list, outfile, indent=4, sort_keys=True)
    
    print(f"<<< saved at: {os.path.join(save_directory_page, NEW_JSON_NAME)} >>>")


if __name__ == "__main__":
    main()
