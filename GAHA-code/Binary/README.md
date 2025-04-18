
### 1. Preprocess the Datasets
To preprocess the datasets, run the following commands.

```shell script
unzip src_data.zip
cd code
python process_datasets.py
```

Now, the processed datasets are in the `data` directory.

### 2. Reproduce the Results 

```shell script
CUDA_VISIBLE_DEVICES=2 python learn.py --dataset WN18RR --model GAHE --rank 2000 --optimizer Adagrad --learning_rate 1e-1 --batch_size 100 --reg 1e-1 --max_epochs 50 --valid 5 -train -id 0 -save -weight


CUDA_VISIBLE_DEVICES=5 python learn.py --dataset FB237 --model GAHE --rank 2000 --optimizer Adagrad --learning_rate 1e-1 --batch_size 100 --reg 5e-2 --max_epochs 200 --valid 5 -train -id 0 -save


CUDA_VISIBLE_DEVICES=3 python learn.py --dataset YAGO3-10 --model GAHE --rank 1000 --optimizer Adagrad --learning_rate 1e-1 --batch_size 1000  --reg 5e-3 --max_epochs 200 --valid 5 -train -id 0 -save
```