SEED=$1
pwd
cd 3D-ResNets-PyTorch
python main.py --root_path ~/rekall_experiments/commercial_experiments \
--video_path commercials/images --annotation_path commercials/data \
--result_path results_$SEED --dataset commercials --n_classes 400 --n_finetune_classes 2 \
--pretrain_path models/resnet-34-kinetics.pth --ft_begin_index 4 \
--model resnet --model_depth 34 --resnet_shortcut A --batch_size 32 --n_threads 16 --checkpoint 5 \
--no_val --manual_seed $SEED
