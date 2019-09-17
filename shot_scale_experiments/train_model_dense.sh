SEED=$1
pwd
cd 3D-ResNets-PyTorch
python main.py --root_path ~/rekall_experiments/shot_scale_experiments \
--video_path shot_scale/images --annotation_path shot_scale/shot_scale_labels_and_rekall_accuracy_val_test.pkl \
--result_path results_dense_$SEED --dataset shot_scale_dense --n_classes 400 --n_finetune_classes 3 \
--pretrain_path models/resnet-34-kinetics.pth --ft_begin_index 4 \
--model resnet --model_depth 34 --resnet_shortcut A --batch_size 32 --n_threads 16 --checkpoint 5 \
--no_val --manual_seed $SEED
cd ..
