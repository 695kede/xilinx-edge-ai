# Declare $PATH_TO_DATASET_DIR and $PATH_TO_DATASET_LIST

#../../build/tools/convert_driving_data /home/chengming/chengming/VPGNet/TSD /home/chengming/chengming/VPGNet/TSD/train_new_TSD.txt LMDB_train_new_TSD
#../../build/tools/compute_driving_mean LMDB_train_new_TSD ./driving_mean_train_new_TSD.binaryproto lmdb
#../../build/tools/convert_driving_data /home/chengming/chengming/VPGNet/TSD /home/chengming/chengming/VPGNet/TSD/test_new_TSD.txt LMDB_test_new_TSD

#../../build/tools/convert_driving_data /home/chengming/chengming/SCNN/data/CULane /home/chengming/chengming/SCNN/data/CULane/train_SCNN_vpgnet.txt LMDB_SCNN_train

#../../build/tools/compute_driving_mean LMDB_SCNN_train ./driving_mean_SCNN.binaryproto lmdb

#../../build/tools/convert_driving_data /home/chengming/chengming/SCNN/data/CULane /home/chengming/chengming/SCNN/data/CULane/test_SCNN_vpgnet.txt LMDB_SCNN_test


./build/tools/convert_driving_data /home/chengming/chengming/VPGNet/ /home/chengming/chengming/VPGNet/train_caltech.txt LMDB_train_compress
./build/tools/compute_driving_mean LMDB_train_compress ./driving_mean_train.binaryproto lmdb
./build/tools/convert_driving_data /home/chengming/chengming/VPGNet/ /home/chengming/chengming/VPGNet/test_caltech.txt LMDB_test_compress

