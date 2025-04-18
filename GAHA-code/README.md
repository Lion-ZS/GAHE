# GAHE: Geometry-Aware Embedding for Hyper-Relational Knowledge Graph Representation


## Requirements
```setup
python 3.7.4
pytorch 1.13
```

## Running a model

To train (and evaluate) the model in the paper, run this command:

>📋 The ary append for FB-AUTO is 
```
CUDA_VISIBLE_DEVICES=1 python main.py --dataset FB-AUTO --num_iterations 200 --batch_size 256  --c 0.01 --rate 0.3 --lr 0.005 --dr 0.995 --K 10 --rdim 50 --m 2 --drop_role 0.2 --drop_ent 0.4 --eval_step 1 --valid_patience 10 -ary 2 -ary 4 -ary 5
```

>📋 The ary append for WikiPeople is 
```
CUDA_VISIBLE_DEVICES=1 python main.py --dataset WikiPeople --num_iterations 200 --batch_size 64 --c 0.1 --rate 0.2 --lr 0.005 --dr 0.995 --K 10 --rdim 50 --m 2 --drop_role 0.2 --drop_ent 0.4 --eval_step 1 --valid_patience 10 -ary 2 -ary 3 -ary 4 -ary 5 -ary 6 -ary 7 -ary 8 -ary 9
```
>📋 The ary append for JF17K is 
```
CUDA_VISIBLE_DEVICES=0 python main.py --dataset JF17K --num_iterations 200 --batch_size 256 --c 0.1 --rate 0.2 --lr 0.005 --dr 0.995 --K 10 --rdim 50 --m 2 --drop_role 0.2 --drop_ent 0.4 --eval_step 1 --valid_patience 10 -ary 2 -ary 3 -ary 4 -ary 5 -ary 6
```

