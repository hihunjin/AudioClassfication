#!/bin/bash
for i in {1..81}
do
    echo $i
    sudo -H -E PYTHONPATH=$PYTHONPATH:$pwd python datasets/kpf.py --page $i --save_directory /mnt/ebs/data/kpf0.0.2 --version 0.0.2
done
