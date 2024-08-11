# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

import os

wandb_log = False
# wandb_project = 'owt'
# wandb_run_name='gpt2-124M'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 4
block_size = 512 # 1024 # context length
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 300B
#max_iters = 600000
max_iters = 4000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1


# New
n_layer = 6
n_head = 8
n_embd = 128

compile = False
device = "cuda:4"

dataset = 'wiki2' # for data_dir
data_dir = '/data/ryan/other/wiki2' # specified in train file
out_dir = '/data/ryan/other/nanoGPT/out'
chkpt_dir = os.path.join(out_dir, 'ckpts')
emb_dir = os.path.join(out_dir, 'emb')
os.makedirs(out_dir, exist_ok=True)

