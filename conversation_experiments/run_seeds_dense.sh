SEED=$1
echo $SEED

mkdir -p results_dense_$SEED

./train_3d_model_dense.sh $SEED
./test_3d_model_dense.sh $SEED

# Evaluate results...
python process_results.py $SEED >> seed_results_dense.log

