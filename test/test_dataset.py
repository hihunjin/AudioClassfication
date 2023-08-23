from datasets.kpf_dataset import KpfDataset, KpfDatasetPage


if __name__ == "__main__":
    root = r'/mnt/ebs/data/kpf'
    # page = 1
    segment_length = 114688
    sampling_rate = 22050
    n_classes = 8

    train_dataset, test_dataset = KpfDataset(
        root=root,
        segment_length=segment_length,
        sampling_rate=sampling_rate,
        n_classes=n_classes,
        num_pages=20,
        split=True,
    )

    for i in range(100):
        train_dataset[i]
