pytest tpu_graphs/test_pt_loader.py -k "test_loads_batch" -s

python tiles_train.py --model=EarlyJoinSAGE
