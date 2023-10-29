#!/bin/bash

export version="0.0.2"

for i in {2..70}
do
    echo $i
    sudo -H python datasets/kpf.py --page $i --save_directory /mnt/ebs/data/kpf$version --version $version
done
