#export DATAPATH=/data2/datasets #DB
export DATAPATH=$CAFFE_ROOT/data/VOC0712 #DB
#export DATAPATH=~/ML/SSD/VOC/data/VOC0712
cur_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )
#root_dir=$cur_dir/../.. #DB
root_dir=$CAFFE_ROOT

#cd $root_dir

redo=1
data_root_dir="$DATAPATH/VOCdevkit"
dataset_name="VOC0712"
#mapfile="$root_dir/data/$dataset_name/labelmap_voc.prototxt" #DB
mapfile="$DATAPATH/labelmap_voc.prototxt"
anno_type="detection"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0

extra_cmd="--encode-type=jpg --encoded"
if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi
for subset in test trainval
do
#  python $root_dir/scripts/create_annoset.py --anno-type=$anno_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $root_dir/data/$dataset_name/$subset.txt $data_root_dir/$dataset_name/$db/$dataset_name"_"$subset"_"$db examples/$dataset_name #DB
	python $root_dir/scripts/create_annoset.py --anno-type=$anno_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $DATAPATH/$subset.txt $data_root_dir/$dataset_name/$db/$dataset_name"_"$subset"_"$db examples/$dataset_name
done
