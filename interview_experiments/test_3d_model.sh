pwd
cd 3D-ResNets-PyTorch
python main.py --root_path ~/rekall_experiments/interview_experiments \
--video_path bernie_interviews/images --annotation_path bernie_interviews/data \
--result_path results --dataset interviews --n_classes 2 --n_finetune_classes 2 \
--resume_path results/save_5.pth --ft_begin_index 4 \
--model resnet --model_depth 34 --resnet_shortcut A --batch_size 32 --n_threads 16 --checkpoint 5 \
--no_val --no_train --test --test_subset test

