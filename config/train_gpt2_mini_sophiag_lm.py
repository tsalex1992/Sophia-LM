wandb_log = True
wandb_project = "sophia"
wandb_run_name = "gpt2-mini-sophiag-lm-10k"

# 8 batch size * 512 block size * 6 gradaccum * 1 GPUs = 24576
batch_size = 8
block_size = 512
gradient_accumulation_steps = 6
total_bs = batch_size

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False

damping_factor = 1.0
damping_increment = 2.0
damping_decrement = 3.0
use_conditioned_step = False
# this makes total number of tokens be 300B
max_iters = 10000
lr_decay_iters = 10000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# optimizer
optimizer_name = "sophiaglm"
learning_rate = 3e-4  # max learning rate
weight_decay = 2e-1
beta1 = 0.965
beta2 = 0.99
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 20  # how many steps to warm up for
min_lr = 1.5e-5
rho = 0.03
interval = 10

compile = False

out_dir = "out_mini_sophiag_lm_10k"
