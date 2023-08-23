import os
import random
import datetime

import torch
from torch.utils.data import Dataset, ConcatDataset, random_split
import librosa
import pandas as pd
import torch.nn.functional as F

from utils.io import load_json, load_wav
from datasets.audio_augs import AudioAugs


def time_string_to_frame_number(string, sr=16000):
    m, s = string.split(":")
    return (int(m) * 60 + int(s)) * sr


def time_string_to_seconds(timestring):
    """xx:xx:xx or xx:xx -> yyy seconds"""
    if len(timestring) < 6:
        timestring = "00:" + timestring
    try:
        x = datetime.datetime.strptime(timestring, "%H:%M:%S")
    except:
        print("timestring", timestring)
        return 0
    return datetime.timedelta(
        hours=x.hour, minutes=x.minute, seconds=x.second
    ).total_seconds()


class KpfDatasetPage(Dataset):
    def __init__(
        self,
        root: str,
        page: int,
        segment_length: int,
        sampling_rate: int,
        n_classes: int,
        transforms=None,
        crop: bool = True,
    ):
        mother_path = os.path.join(root, f"page{page}")
        json_path = os.path.join(mother_path, "_0.0.1_simple.json")
        self.dataframe = self.make_dataframe(json_path)
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate
        self.n_classes = n_classes
        self.transforms = transforms
        self.crop = crop

    def make_dataframe(self, json_path) -> pd.DataFrame:
        data_infos = load_json(json_path)
        rows = []
        for id, val in data_infos.items():
            rows.append(val)
        dataframe = pd.DataFrame(rows).set_index("path")
        return dataframe

    def __len__(self):
        return len(self.dataframe)

    def fit_size(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.shape[0] >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start : audio_start + self.segment_length]
        else:
            audio = F.pad(audio, (0, self.segment_length - audio.size(0)), "constant").data
        return audio

    def choose_random_interval(self, intervals: dict) -> dict:
        """
        Args:
            intervals (_type_):
            ```
            {
                "HM": [
                    {
                        "end": "00:00:27",
                        "start": "00:00:07"
                    },
                    {
                        "end": "00:01:07",
                        "start": "00:00:37"
                    },
                ],
                "LO": [
                    {
                        "end": "00:01:18",
                        "start": "00:01:07"
                    }
                ],
                ...,
            }
            ```

        Returns:
            tuple:
            ```
            "HM", {
                "end": "00:00:27",
                "start": "00:00:07"
            }
            ```
        """
        import random

        # random.seed(0)

        fmso = random.choice(list(intervals.keys()))
        interval = random.choice(intervals[fmso])

        return fmso, interval

    def load_with_start_end(self, fname, time_start: int = None, time_end: int = None):
        """_summary_

        Args:
            fname (_type_): _description_
            time_start (int, optional): _description_. Defaults to None.
            time_end (int, optional): _description_. Defaults to None.

        Returns:
            _type_: torch.Size([n])
        """
        if time_end is not None:
            duration = time_end - time_start
        else:
            duration = None
        audio, sampling_rate = librosa.core.load(
            fname,
            offset=time_start,
            duration=duration,
            sr=None,
            mono=True,
        )
        # frame_start, frame_end = list(map(time_string_to_frame_number, time_section.split("-")))
        # audio = audio[frame_start: frame_end]
        audio = 0.95 * librosa.util.normalize(audio)
        audio = torch.from_numpy(audio).float()

        assert (
            "sampling rate of the file is not as configured in dataset, will cause slow fetch {}".format(
                sampling_rate != self.sampling_rate
            )
        )

        return audio

    def basic_load(self, fname) -> torch.Tensor:
        """_summary_

        Args:
            fname (_type_): _description_

        Returns:
            torch.Tensor: torch.Size([n])
        """
        audio, sampling_rate = load_wav(fname)
        # frame_start, frame_end = list(map(time_string_to_frame_number, time_section.split("-")))
        # audio = audio[frame_start: frame_end]
        audio = 0.95 * librosa.util.normalize(audio)
        audio = torch.from_numpy(audio).float()

        assert (
            "sampling rate of the file is not as configured in dataset, will cause slow fetch {}".format(
                sampling_rate != self.sampling_rate
            )
        )

        return audio

    def __getitem__(self, index):
        row: pd.series = self.dataframe.iloc[index]
        path = row.name
        row = row.to_dict()

        if self.crop is True:
            fmso, interval = self.choose_random_interval(row["time_interval"])
            audio = self.load_with_start_end(
                path,
                time_string_to_seconds(interval["start"]),
                time_string_to_seconds(interval["end"]),
            )
            if row["fmso"] != fmso:
                row["fmso"] = fmso
            audio = self.fit_size(audio)
        else:
            audio = self.basic_load(path)
        if self.transforms is not None:
            audio = AudioAugs(self.transforms, self.sampling_rate, p=0.5)(audio)

        return audio.unsqueeze(0), row


def KpfDataset(
    root: str,
    segment_length: int,
    sampling_rate: int,
    n_classes: int,
    num_pages: int,
    transforms=None,
    crop: bool = True,
    split: bool = False,
    seed: int = 0,
):
    concat_dataset = ConcatDataset(
        [
            KpfDatasetPage(
                root=root,
                page=page,
                segment_length=segment_length,
                sampling_rate=sampling_rate,
                n_classes=n_classes,
                transforms=transforms,
                crop=crop,
            )
            for page in range(1, num_pages + 1)
        ]
    )
    if split is True:
        train_count = int(0.7 * len(concat_dataset))
        test_count = len(concat_dataset) - train_count
        generator = torch.Generator().manual_seed(seed)
        train_dataset, test_dataset = random_split(
            concat_dataset, [train_count, test_count], generator
        )
        test_dataset.transforms = None
        return train_dataset, test_dataset
    else:
        return concat_dataset
