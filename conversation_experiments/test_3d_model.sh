SEED=$1
pwd
cd 3D-ResNets-PyTorch
python main.py --root_path ~/rekall_experiments/conversation_experiments \
--video_path conversation_export/images --annotation_path conversation_export/data \
--result_path results_$SEED --dataset conversations --n_classes 2 --n_finetune_classes 2 \
--resume_path results_$SEED/save_200.pth --ft_begin_index 4 \
--model resnet --model_depth 34 --resnet_shortcut A --batch_size 32 --n_threads 16 --checkpoint 5 \
--no_val --no_train --test --test_subset test

