import yaml
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.io import save_json
from utils.helper_funcs import accuracy, count_parameters, mAP, measure_inference_time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f_res", default=None, type=Path)
    parser.add_argument("--add_noise", default=False, type=bool)
    parser.add_argument("--infer_json_path", default=Path("/mnt/ebs/data/all_231027.json"), type=Path)
    args = parser.parse_args()
    return args


def run():
    _args = parse_args()
    f_res = _args.f_res
    add_noise = _args.add_noise
    with (f_res / Path("args.yml")).open() as f:
        args = yaml.load(f, Loader=yaml.Loader)
    try:
        args = vars(args)
    except:
        if 'net' in args.keys():
            del args['net']
        args_orig = args
        args = {}
        for k, v in args_orig.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    args[kk] = vv
            else:
                args[k] = v
    args['f_res'] = f_res
    args['add_noise'] = add_noise
    args["infer_json_path"] = _args.infer_json_path
    with open(args['f_res'] / "args.yml", "w") as f:
        yaml.dump(args, f)
    print(args)
    #######################
    # Load PyTorch Models #
    #######################
    from modules.soundnet import SoundNetRaw as SoundNet
    ds_fac = np.prod(np.array(args['ds_factors'])) * 4
    net = SoundNet(nf=args['nf'],
                   dim_feedforward=args['dim_feedforward'],
                   clip_length=args['seq_len'] // ds_fac,
                   embed_dim=args['emb_dim'],
                   n_layers=args['n_layers'],
                   nhead=args['n_head'],
                   n_classes=args['n_classes'],
                   factors=args['ds_factors'],
                   )

    print('***********************************************')
    print("#params: {}M".format(count_parameters(net)/1e6))
    if torch.cuda.is_available() and device == torch.device("cuda"):
        t_b1 = measure_inference_time(net, torch.randn(1, 1, args['seq_len']))[0]
        print('inference time batch=1: {:.2f}[ms]'.format(t_b1))
        # t_b32 = measure_inference_time(net, torch.randn(32, 1, args['seq_len']))[0]
        # print('inference time batch=32: {:.2f}[ms]'.format(t_b32))
        print('***********************************************')

    if (f_res / Path("chkpnt.pt")).is_file():
        chkpnt = torch.load(f_res / "chkpnt.pt", map_location=torch.device(device))
        model = chkpnt['model_dict']
    else:
        raise ValueError

    if 'use_dp' in args.keys() and args['use_dp']:
        from collections import OrderedDict
        state_dict = OrderedDict()
        for k, v in model.items():
            name = k.replace('module.', '')
            state_dict[name] = v
        net.load_state_dict(state_dict, strict=True)
    else:
        net.load_state_dict(model, strict=True)
    net.to(device)
    if torch.cuda.device_count() > 1:
        from utils.helper_funcs import parse_gpu_ids
        args['gpu_ids'] = [i for i in range(torch.cuda.device_count())]
        net = torch.nn.DataParallel(net, device_ids=args['gpu_ids'])
        net.to('cuda:0')
    net.eval()
    #######################
    # Create data loaders #
    #######################
    if args['dataset'] == 'esc50':
        from datasets.esc_dataset import ESCDataset as SoundDataset
        data_set = SoundDataset(args['data_path'],
                                mode='test',
                                segment_length=args['seq_len'],
                                sampling_rate=args['sampling_rate'],
                                fold_id=args['fold_id'],
                                transforms=None)

    elif args["dataset"] == "kpf":
        from datasets.kpf_dataset import KpfDatasetPath as SoundDataset

        # FIXME
        infer_json_path = args["infer_json_path"]
        data_set = SoundDataset(
            infer_json_path,
            segment_length=args["seq_len"],
            sampling_rate=args["sampling_rate"],
            n_classes=args["n_classes"],
            transforms=args["augs_signal"] + args["augs_noise"],
        )
        data_set.filter({"golden_set": {args["task"]: True}})

    elif args['dataset'] == 'speechcommands':
        from datasets.speechcommand_dataset import SpeechCommandsDataset as SoundDataset
        data_set = SoundDataset(args['data_path'],
                                mode='test',
                                segment_length=args['seq_len'],
                                sampling_rate=args['sampling_rate'],
                                transforms=None)

    elif args['dataset'] == 'urban8k':
        from datasets.urban8K_dataset import Urban8KDataset as SoundDataset
        data_set = SoundDataset(args['data_path'],
                                mode='test',
                                segment_length=args['seq_len'],
                                sampling_rate=args['sampling_rate'],
                                transforms=None,
                                fold_id=args['fold_id'])

    elif args['dataset'] == 'audioset':
        from datasets.audioset_dataset import AudioSetDataset as SoundDataset
        data_set = SoundDataset(
            args['data_path'],
            'test',
            data_subtype=None,
            segment_length=args['seq_len'],
            sampling_rate=args['sampling_rate'],
            transforms=None
        )

    else:
        raise ValueError

    if args['dataset'] != 'audioset':
        inference_single_label(net=net, data_set=data_set, args=args)
    elif args['dataset'] == 'audioset':
        inference_multi_label(net=net, data_set=data_set, args=args)
    else:
        raise ValueError("check args dataset")

def inference_single_label(net, data_set, args):
    from utils.helper_funcs import collate_fn_keep_dict

    data_loader = DataLoader(
        data_set,
        batch_size=args["batch_size"],
        num_workers=8,
        pin_memory=True if torch.cuda.is_available() else False,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn_keep_dict,
    )

    # create label converter
    from utils.label_converter import LabelConverter

    converter = LabelConverter(task=args["task"])

    labels = torch.zeros(len(data_loader.dataset), dtype=torch.float32).float()
    preds = torch.zeros(len(data_loader.dataset), args['n_classes'], dtype=torch.float32).float()
    # confusion_matrix = torch.zeros(args['n_classes'], args['n_classes'], dtype=torch.int)
    confusion_matrix = torch.zeros(args['n_classes'], args['n_classes'], dtype=torch.int)
    idx_start = 0
    results = []
    for i, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.to(device)
            y_target = converter.get_target(y, args["dataset"])
            y_target = y_target.to(device)
            pred = net(x)
            _, y_est = torch.max(pred, 1)
            idx_end = idx_start + y_target.shape[0]
            preds[idx_start:idx_end, :] = pred
            labels[idx_start:idx_end] = y_target
            for t, p in zip(y_target.view(-1), y_est.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            print("{}/{}".format(i, len(data_loader)))
            results.extend(
                [
                    {
                        "path": _path,
                        "label": converter.convert_model_output_to_label(_y),
                        "pred": converter.convert_model_output_to_label(_pred),
                        "time_interval": _time_interval,
                    } for _path, _y, _pred, _time_interval in zip(y["path"], y_target.tolist(), y_est.tolist(), y["time_interval"])
                ]
            )
        idx_start = idx_end
    pd.DataFrame(results).to_csv(args['f_res'] / f"results_{args['task']}.csv", index=False)
    acc_av = accuracy(preds.detach(), labels.detach(), [1, ])[0]

    res = {
        "acc": acc_av.item(),
        "preds": preds.tolist(),
        "labels": labels.view(-1).tolist(),
        "confusion_matrix": confusion_matrix.tolist(),
    }
    save_json(args['f_res'] / "res.json", res)
    print(f"result saved at:{args['f_res'] / 'res.json'}")

    print("acc:{}".format(np.round(acc_av*100)/100))
    print("cm:{}".format(confusion_matrix.diag().sum() / len(data_loader.dataset)))
    print('***************************************')
    # bad_labels = []
    # for i, c in enumerate(confusion_matrix):
    #     i_est = c.argmax(-1)
    #     if i != i_est:
    #         print('{} {} {}-->{}'.format(i, i_est.item(), data_set.labels[i], data_set.labels[i_est]))
    #         bad_labels.append([i, i_est])
    # print(bad_labels)
    print("confusion_matrix: \n", *confusion_matrix.tolist(), sep="\n")


def inference_multi_label(net, data_set, args):
    from utils.helper_funcs import collate_fn
    data_loader = DataLoader(data_set,
                             batch_size=128,
                             num_workers=8,
                             pin_memory=True if torch.cuda.is_available() else False,
                             shuffle=False,
                             collate_fn=collate_fn)

    labels = torch.zeros(len(data_loader.dataset), args['n_classes'], dtype=torch.float32).float()
    preds = torch.zeros(len(data_loader.dataset), args['n_classes'], dtype=torch.float32).float()
    confusion_matrix = torch.zeros(args['n_classes'], args['n_classes'], dtype=torch.int)
    idx_start = 0
    for i, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.to('cuda:0')
            y = [F.one_hot(torch.Tensor(y_i).long(), args['n_classes']).sum(dim=0).float() for y_i in y]
            y = torch.stack(y, dim=0).contiguous().to('cuda:0')
            pred = net(x)
            idx_end = idx_start + y.shape[0]
            preds[idx_start:idx_end, :] = torch.sigmoid(pred)
            labels[idx_start:idx_end, :] = y
            print("{}/{}".format(i, len(data_loader)))
        idx_start = idx_end
    mAP_av = mAP(labels.detach().cpu().numpy(), preds.detach().cpu().numpy())
    res = {
        "mAP": mAP_av,
        "preds": preds,
        "labels": labels.view(-1),
        "confusion_matrix": confusion_matrix
    }
    torch.save(res, args['f_res'] / "res.pt")
    # torch.save(net.state_dict(), "net.pt")
    print("mAP:{}".format(np.round(mAP_av*100)/100))


if __name__ == '__main__':
    run()
