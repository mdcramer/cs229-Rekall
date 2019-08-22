SEED=$1
pwd
cd 3D-ResNets-PyTorch
python main.py --root_path ~/rekall_experiments/interview_experiments \
--video_path bernie_interviews_30sec/images --annotation_path bernie_interviews_30sec/data \
--result_path results_30sec_balanced_$SEED --dataset interviews --n_classes 400 --n_finetune_classes 2 \
--pretrain_path models/resnet-34-kinetics.pth --ft_begin_index 4 \
--model resnet --model_depth 34 --resnet_shortcut A --batch_size 32 --n_threads 16 --checkpoint 5 \
--no_val --balance_classes --manual_seed $SEED
