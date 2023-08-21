from datasets.kpf_dataset import KpfDataset, KpfDatasetPage



root = r'/mnt/ebs/data/kpf'
# page = 1
segment_length = 114688
sampling_rate = 22050
n_classes = 8

out = KpfDataset(
    root=root,
    segment_length=segment_length,
    sampling_rate=sampling_rate,
    n_classes=n_classes,
    num_pages=20,
    split=True,
)

out