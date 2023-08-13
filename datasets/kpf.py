import os
from typing import Union

import json
import click
import urllib.request


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


JSON_NAME = "_0.0.1.json"
NEW_JSON_NAME = "_0.0.1_simple.json"
URL = "https://tadmins.tunegem.io/voai/{JSON_NAME}?page={page}"  # page: 1 ~ 70


@click.command()
@click.option("--page", default=1, help="page number")
@click.option(
    "--save_directory",
    default="datasets/kpf",
    help="directory to save wav files",
)
def main(page: int, save_directory: str):
    save_directory_page = os.path.join(save_directory, f"page{page}")
    os.makedirs(os.path.join(save_directory_page, "wav"), exist_ok=True)
    json_path = None
    data = load_json(json_path=json_path, json_url=URL.format(JSON_NAME=JSON_NAME, page=page))
    # get dictionary and reformat
    df_list = {}
    for key, val in data.items():
        # download data if not exists
        saved_loc = convert_mp4_to_wav_save(
            val["mp4url"],
            os.path.join(save_directory_page, "wav"),
        )
        df_list[key] = {
            "path": saved_loc,
            "fmso": list(val["time_interval"].keys())[0],  # FIXME
            "level": val["level"],
            "time_interval": val["time_interval"],
        }
    with open(os.path.join(save_directory, NEW_JSON_NAME), "w") as outfile:
        json.dump(df_list, outfile, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()
