For starting evaluation, please put the dataset in the `datasets` folder, and download the pre-trained weight from 16 respositories / papers and put into the `pretrained-weight` forlder.

For the methods: SBI, MAT, ICT, Rossler, ForgeryNet, Capsule Forensics, CADDM, CCViT, ADD, MCX, LGrad, Effb4Net, please use the following code for testing:

```
CUDA_VISIBLE_DEVICES="0" python test.py  --model-name <method> --batch-size 128
```

Where the ` <method> ` is selected from (`selfblended`, `mat`, `ict`, `rossler`, `forgerynet`, `capsule`, `caddm`, `ccvit`, `add`, `mcx`, `lgrad`, `effb4att`)

For the methods:  FTCN, AltFreexing, LipForensics, LRNet, CLRNet, please refer to their respective folder and run the `test.py` or `test.ipynb`.

For Lgrad method: Please learn how to pre-process facial deepfake image datasets to gradient datasets and download pretrained GAN networks from their official repository before run the `test.py`.
