from datasets.kpf_dataset import KpfDataset
from torch.utils.data import DataLoader
from utils.helper_funcs import build_sampler, collate_fn_keep_dict


if __name__ == "__main__":
    root = r'/mnt/ebs/data/kpf'
    # page = 1
    segment_length = 114688
    sampling_rate = 22050
    n_classes = 8
    task = "level"

    train_dataset, test_dataset = KpfDataset(
        root=root,
        segment_length=segment_length,
        sampling_rate=sampling_rate,
        n_classes=n_classes,
        num_pages=20,
        split=True,
    )

    train_sampler = build_sampler(
        dataset=train_dataset,
        use_balanced_sampler=True,
        task=task,
    )
    dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=16,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn_keep_dict,
        sampler=train_sampler,
    )
    for i, data in enumerate(dataloader):
        print(data[1][task])
        if i > 10:
            break
