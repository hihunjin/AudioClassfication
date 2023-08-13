#!/bin/bash
for i in {1..70}
do
    echo $i
    sudo -H python datasets/kpf.py --page $i --save_directory /mnt/ebs/data/kpf
done
