SEED=$1
echo $SEED

mkdir -p results_$SEED

./train_3d_model.sh $SEED
./test_3d_model.sh $SEED

# Evaluate results...
python process_results.py $SEED >> seed_results.log

