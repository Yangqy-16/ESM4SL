# ESM4SL

This is the codes of **ESM4SL**, which performs synthetic lethality (SL) prediction based on [ESM-2](https://github.com/facebookresearch/esm).

## Usage 

### Environment
First, you can create the environment required for running our codes by
```
bash environment.sh
```

### Preprocess

All the codes for data preprocessing is in `data_preprocess/`.

### Main
A simple example of running our program is
```
bash script/attn/train.sh
```
After the program finishes, you can see the outputs in `output/attn/` by
```
tensorboard --logdir output/attn/<name_of_your_run>
```

If you would like to run multiple cell lines or scenes in one program, you can refer to `script/attn/specific_all_in_one.py`.

If you would like to run **ESM-2+MLP** instead of **ESM4SL**, you can refer to the same files in `output/mlp/`.

## References

Our codes is based on [coach-pl](https://github.com/DuskNgai/coach-pl). We thank the authors for their great foundational work.

Some of our codes are credited to [ESM](https://github.com/facebookresearch/esm).
