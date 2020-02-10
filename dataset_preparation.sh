#!/bin/sh

./download.sh ./dataset

cd dataset/multiview-pouring
for split in train test val; do
    for f in ./videos/$split/*; do
    name=${f##*/}
    name_sprit=${name%*.*}
    output_name="./frames/"$split"/"$name_sprit"/%06d.jpg"
    mkdir -p "./frames/"$split"/"$name_sprit

    echo "input:"$f
    echo "output:"$output_name
    ffmpeg -i ${f} -vf scale=224:224,transpose=1 -q:v 1 -f image2 $output_name
    done
done