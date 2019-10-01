./build/tools/caffe train --solver=./VPGNet_model_weight/solver_caltech.prototxt -gpu=0,1,2,3 2>&1 | tee ./output/output_VGG_GSTiling.log
