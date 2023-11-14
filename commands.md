pytest tpu_graphs/test_pt_loader.py -k "test_loads_batch" -s

python tiles_train.py --model=EarlyJoinSAGE model_kwargs_json='{"num_gnns": 10, "final_mlp_layers": 4, "hidden_dim": 256}'


python tiles_train.py --model=EarlyJoinSAGE --epochs=200 --batch=1 --model_kwargs_json='{"num_gnns": 10, "final_mlp_layers": 4, "hidden_dim": 256}'

python tiles_train.py --model=EarlyJoinSAGE --epochs=800 --batch=1 --configs=3
python tiles_train.py --model=EarlyJoinSAGE --epochs=200 --batch=4 --configs=4

python tiles_train.py --model=EarlyJoinSAGE --epochs=800 --batch=4 --configs=4 --lr=1e-4
