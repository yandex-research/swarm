# Replication of experiments on compression-aware architectures

To replicate the experiments, there are three steps:
1. Install the [fairseq](https://github.com/facebookresearch/fairseq) code located in [fairseq](./fairseq)
2. Preprocess the data
3. Run the [training commands](./commands.txt)

# Installing Fairseq

Follow instructions of the original [fairseq repo](https://github.com/pytorch/fairseq).

# Processing the data

1. Download the [OpenWebText corpus](https://skylion007.github.io/OpenWebTextCorpus/)
2. Preprocess according to the instructions for [GPT-2 BPE preprocessing](https://github.com/pytorch/fairseq/blob/main/examples/roberta/README.pretraining.md) in fairseq

# Runing the commands

The commands are in commands.txt. Some of the paths need to be changed to the dataset location or the checkpoint location specified by `--save-dir`.

# Maxout and bottleneck compression code

The code of the compression layers can be found in the path `./fairseq/fairseq/modules/transformer_layer.py`. Search for "bottleneck" and "maxout".
