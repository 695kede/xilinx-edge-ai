#!/bin/bash
export DATAPATH=$CAFFE_ROOT/data/VOC0712 #DB
#export DATAPATH=~/ML/SSD/VOC/data/VOC0712
root_dir="$DATAPATH/VOCdevkit"
sub_dir=ImageSets/Main
bash_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" #DB
#bash_dir="$CAFFE_ROOT" #DB
for dataset in trainval test
do
  dst_file=$bash_dir/$dataset.txt #DB
  #dst_file=$DATAPATH/$dataset.txt #DB
  if [ -f $dst_file ]
  then
    rm -f $dst_file
  fi
  for name in VOC2007 VOC2012
  do
    if [[ $dataset == "test" && $name == "VOC2012" ]]
    then
      continue
    fi
    echo "Create list for $name $dataset..."
    dataset_file=$root_dir/$name/$sub_dir/$dataset.txt

    img_file=$bash_dir/$dataset"_img.txt"
    cp $dataset_file $img_file
    sed -i "s/^/$name\/JPEGImages\//g" $img_file
    sed -i "s/$/.jpg/g" $img_file

    label_file=$bash_dir/$dataset"_label.txt"
    cp $dataset_file $label_file
    sed -i "s/^/$name\/Annotations\//g" $label_file
    sed -i "s/$/.xml/g" $label_file

    paste -d' ' $img_file $label_file >> $dst_file

    rm -f $label_file
    rm -f $img_file
  done

  # Generate image name and size infomation.
  if [ $dataset == "test" ]
  then
      #$bash_dir/../../build/tools/get_image_size $root_dir $dst_file $bash_dir/$dataset"_name_size.txt" #DB
      $CAFFE_ROOT/build/tools/get_image_size $root_dir $dst_file $bash_dir/$dataset"_name_size.txt" #DB
  fi

  # Shuffle trainval file.
  if [ $dataset == "trainval" ]
  then
    rand_file=$dst_file.random
    cat $dst_file | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' > $rand_file
    mv $rand_file $dst_file
  fi
done
