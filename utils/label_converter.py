from typing import Union

import torch

from utils.helper_funcs import CategoryConverter


class LabelConverter:
    fmso_to_model_output = {"LF": 0, "LM": 1, "LS": 2, "LO": 3, "HF": 4, "HM": 5, "HS": 6, "HO": 7, "RP": 0}

    def __init__(self, task: str = "level"):
        assert task in [
            "level",
            "level_half",
            "fmso",
        ] + list(CategoryConverter.to_num_dict.keys())
        self.task = task

    def convert_label_to_model_output(self, label: Union[str, int]):
        if self.task == "level":
            return self._convert_level_label_to_model_output(label)
        elif self.task == "fmso":
            return self._convert_fmso_label_to_model_output(label)
        elif self.task == "level_half":
            return self._convert_level_half_label_to_model_output(label)

    def _convert_fmso_label_to_model_output(self, label: Union[str, int]):
        return self.fmso_to_model_output[label]

    def _convert_level_label_to_model_output(self, label: Union[str, int]):
        label = int(label) if isinstance(label, str) else label
        return label - 1

    def _convert_level_half_label_to_model_output(self, label: Union[str, int]):
        label = int(label) if isinstance(label, str) else label
        return (label + 1) // 2 - 1

    def convert_model_output_to_label(self, model_output: int):
        if self.task == "fmso":
            return self._convert_fmso_model_output_to_label(model_output)
        elif self.task == "level":
            return self._convert_model_output_to_level_label(model_output)
        elif self.task == "level_half":
            return self._convert_model_output_to_level_half_label(model_output)

    def _convert_fmso_model_output_to_label(self, model_output: int):
        return list(self.fmso_to_model_output.keys())[model_output]

    def _convert_model_output_to_level_label(self, model_output: int):
        return model_output + 1

    def _convert_model_output_to_level_half_label(self, model_output: int):
        return model_output + 1

    def get_target(self, y, dataset):
        if dataset == "kpf":
            if self.task == "level":
                # TODO
                return (torch.Tensor(list(map(int, y["level"]))) - 1).type(torch.int64)
            elif self.task == "fmso":
                return torch.Tensor(
                    list(
                        map(
                            self.convert_label_to_model_output,
                            y["fmso"],
                        )
                    )
                ).type(torch.int64)
            elif self.task in CategoryConverter.to_num_dict.keys():
                if self.task not in ["완성도", "표현력"]:
                    return torch.Tensor(
                        [i[CategoryConverter.to_num(self.task)] for i in y["properties"]]
                    ).type(torch.int64)
                else:
                    return torch.Tensor(
                        [i[CategoryConverter.to_num(self.task)] == True for i in y["properties"]]
                    ).type(torch.int64)
        else:
            return y


if __name__ == "__main__":
    # converter = LabelConverter(task="level")
    # print(converter.convert_label_to_model_output(1))
    # print(converter.convert_model_output_to_label(0))
    converter = LabelConverter(task="level_half")
    print(converter.convert_label_to_model_output(1))
    print(converter.convert_label_to_model_output(2))
    print(converter.convert_label_to_model_output(3))
    print(converter.convert_label_to_model_output(4))
    print(converter.convert_model_output_to_label(0))
    print(converter.convert_model_output_to_label(1))
    print(converter.convert_model_output_to_label(2))
    print(converter.convert_model_output_to_label(3))
    # converter = LabelConverter(task="fmso")
    # print(converter.convert_label_to_model_output("FL"))
    # print(converter.convert_model_output_to_label(0))