pwd
cd 3D-ResNets-PyTorch
python main.py --root_path ~/rekall_experiments/shot_scale_experiments \
--video_path shot_scale/images --annotation_path shot_scale/shot_scale_labels_and_rekall_accuracy_val_test.pkl \
--result_path results --dataset shot_scale --n_classes 3 --n_finetune_classes 3 \
--resume_path results/save_5.pth --ft_begin_index 4 \
--model resnet --model_depth 34 --resnet_shortcut A --batch_size 32 --n_threads 16 --checkpoint 5 \
--no_val --no_train --test --test_subset test

