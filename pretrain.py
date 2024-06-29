# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

import os
import math
import time
from datetime import datetime
import argparse
import torch
import torchinfo
from hellaswag import render_example, iterate_examples, get_most_likely_row
from fineweb.fineweb_dataloader import DataLoaderLite
import tiktoken

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from language_models import TransformerLM,  configure_optimizers
# from gpt2_model import GPT, GPTConfig

# Create argument parser
parser = argparse.ArgumentParser(description='Pretrain script for transformer_residual_stream')

# Add arguments for logging and checkpointing
parser.add_argument('--wandb_log', type=int, default=1, help='Enable wandb logging')
parser.add_argument('--wandb_project', type=str, default='fineweb', help='Wandb project name')
parser.add_argument('--run_name', type=str, default=None, help='Name of the run')

# Add arguments for optimizer configuration
parser.add_argument('--total_batch_size', type=int, default=524_288, help='Total batch size')
parser.add_argument('--B', type=int, default=32, help='Micro batch size')
parser.add_argument('--T', type=int, default=1024, help='Sequence length')
parser.add_argument('--max_lr', type=float, default=6e-4, help='Maximum learning rate')
parser.add_argument('--min_lr', type=float, default=6e-5, help='Minimum learning rate')
parser.add_argument('--warmup_steps', type=int, default=715, help='Number of warmup steps')
parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.95), help='Betas for Adam optimizer')
parser.add_argument('--learning_rate', type=float, default=6e-4, help='Learning rate')
parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value')

# Add arguments about training procedure
parser.add_argument('--eval_interval', type=int, default=250, help='Interval for evaluation')
parser.add_argument('--val_loss_steps', type=int, default=20, help='Number of steps for validation loss')
parser.add_argument('--save_interval', type=int, default=5000, help='Interval for saving model')
parser.add_argument('--gen_interval', type=int, default=250, help='Interval for generation')
parser.add_argument('--max_steps', type=int, default=19073, help='Maximum number of steps') # TODO change to calculate dynamically as # of tokens?

# Add arguments for cuda optimizations
parser.add_argument('--compile', type=int, default=1, help='Enable torch.compile')
parser.add_argument('--use_bf16', type=int, default=1, help='Use bfloat16 for matmuls')

# Add arguments for model configuration
# default config matches GPT2-medium / GPT3-medium (124M params)
parser.add_argument('--vocab_size', type=int, default=50304, help='Number of tokens in the tokenizer')
parser.add_argument('--d_model', type=int, default=1024, help='Dimensionality of the model')
parser.add_argument('--n_layers', type=int, default=24, help='Number of layers in the model')
parser.add_argument('--n_heads', type=int, default=16, help='Number of attention heads')
parser.add_argument('--dff', type=int, default=None, help='Dimensionality of the feed-forward layer')
parser.add_argument('--activation', type=str, default='gelu', help='Activation function')
parser.add_argument('--dropout_rate', type=float, default=0.0, help='Dropout rate')
parser.add_argument('--norm_first', type=int, default=1, help='Whether to apply layer normalization before the attention layer')
parser.add_argument('--norm_type', type=str, default='layernorm', help='Type of normalization')
parser.add_argument('--max_block_size', type=int, default=1024, help='Maximum block size')
parser.add_argument('--bias', type=int, default=0, help='Whether to include bias in the model')
parser.add_argument('--pos_enc_type', type=str, default='RoPE', help='Type of positional encoding')

parser.add_argument('--gate_application', type=str, default='none', help='Ways of applying gate')
parser.add_argument('--gate_compute', type=str, default='linear-bias', help='Ways of computing gates')
parser.add_argument('--gate_activation', type=str, default='sigmoid', help='Type of positional encoding')

parser.add_argument('--seed', type=int, default=None, help='Random seed')


# -----------------------------------------------------------------------------
# Parse arguments
args = parser.parse_args()

# Logging and checkpointing configuration
wandb_log = bool(args.wandb_log)
wandb_project = args.wandb_project
run_name = args.run_name

# Optimizer configuration
total_batch_size = args.total_batch_size
micro_batch_size = args.B # micro batch size
max_seq_len = args.T # sequence length
max_lr = args.max_lr
min_lr = args.min_lr
warmup_steps = args.warmup_steps
weight_decay = args.weight_decay
betas = tuple(args.betas)
learning_rate = args.learning_rate # TODO: what is difference bw max_lr and learning_rate?
grad_clip = args.grad_clip if args.grad_clip > 0 else None
optimizer_config = dict(
    total_batch_size=total_batch_size, micro_batch_size=micro_batch_size, max_seq_len=max_seq_len, max_lr=max_lr, min_lr=min_lr, warmup_steps=warmup_steps,
    weight_decay=weight_decay, betas=betas, learning_rate=learning_rate, grad_clip=grad_clip)

# Training procedure
eval_interval = args.eval_interval
save_interval = args.save_interval
val_loss_steps = args.val_loss_steps
gen_interval = args.gen_interval
max_steps = args.max_steps
training_config = dict(eval_interval=eval_interval, save_interval=save_interval, val_loss_steps=val_loss_steps, gen_interval=gen_interval, max_steps=max_steps)


# CUDA optimizations
use_compile = bool(args.compile)
use_bf16 = bool(args.use_bf16)

# Model configuration
vocab_size = args.vocab_size
d_model = args.d_model
n_layers = args.n_layers
n_heads = args.n_heads
dff = args.dff
activation = args.activation
dropout_rate = args.dropout_rate
norm_first = bool(args.norm_first)
norm_type = args.norm_type
max_block_size = args.max_block_size
bias = bool(args.bias)
pos_enc_type = args.pos_enc_type
gate_application = args.gate_application
gate_compute = args.gate_compute
gate_activation = args.gate_activation
resgate_kwargs = dict(d_model=d_model, gate_application=gate_application, gate_compute=gate_compute, gate_activation=gate_activation)
model_config = dict(
    vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads, dff=dff, activation=activation, resgate_kwargs=resgate_kwargs,
    dropout_rate=dropout_rate, norm_first=norm_first, norm_type=norm_type, max_block_size=max_seq_len, bias=bias, pos_enc_type=pos_enc_type)

run_config = dict(optimizer_config=optimizer_config, training_config=training_config, use_compile=use_compile, use_bf16=use_bf16, model_config=model_config)

# annotate run_name with datetime
datetime_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
run_name = datetime_now if run_name is None else f'{run_name}_{datetime_now}'

# set seed, if specified
if args.seed is not None:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

# -----------------------------------------------------------------------------
# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"

if master_process and torch.cuda.is_available():
    print("CUDA is available")
    print(f"Number of available GPUs: {torch.cuda.device_count()}")
    print("GPU Types:")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    if ddp:
        print("Running in DDP mode")
else:
    print("CUDA is not available")


# tokenizer
enc = tiktoken.get_encoding("gpt2")

# calculate gradient accumulation steps
assert total_batch_size % (micro_batch_size * max_seq_len * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (micro_batch_size * max_seq_len * ddp_world_size)

if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=micro_batch_size, T=max_seq_len, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
if master_process:
    print(f"found {len(train_loader.shards)} shards for trainsplit")

val_loader = DataLoaderLite(B=micro_batch_size, T=max_seq_len, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")
if master_process:
    print(f"found {len(val_loader.shards)} shards for trainsplit")

# set up torch to use bfloat16 for matmuls
if use_bf16:
    torch.set_float32_matmul_precision('high')

# create model
model = TransformerLM(**model_config)
model.to(device)
torchinfo.summary(model)

# torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

# TODO: put this function in a separate file?
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# initialize wandb logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=run_name, config=run_config)

# optimizer
optimizer = configure_optimizers(raw_model, weight_decay=weight_decay, betas=betas, learning_rate=learning_rate, device_type=device_type)

# create the log directory we will write checkpoints to and log to
log_dir = f"log/{run_name}"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log_{run_name}.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

# region training loop utility functions

# TODO: modify to take these things as input?
def eval_val_loss():
    model.eval()
    val_loader.reset()
    with torch.no_grad():
        val_loss_accum = 0.0
        for _ in range(val_loss_steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / val_loss_steps
            val_loss_accum += loss.detach()

    if ddp:
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

    return val_loss_accum

def save_checkpoint():
    # optionally write model checkpoints
    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
    checkpoint = {
        'model': raw_model.state_dict(),
        'config': model_config,
        'step': step,
        'val_loss': val_loss_accum.item(),
        'optimizer': optimizer.state_dict(),
        'seed': args.seed
    }
    torch.save(checkpoint, checkpoint_path)

def eval_hellaswag():
    num_correct_norm = 0
    num_total = 0

    for i, example in enumerate(iterate_examples("val")):
        # only process examples where i % ddp_world_size == ddp_rank
        if i % ddp_world_size != ddp_rank:
            continue
        # render the example into tokens and labels
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        # get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(tokens)
            pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct_norm += int(pred_norm == label)

    # reduce the stats across all processes
    if ddp:
        num_total = torch.tensor(num_total, dtype=torch.long, device=device)
        num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
        dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
        num_total = num_total.item()
        num_correct_norm = num_correct_norm.item()
    acc_norm = num_correct_norm / num_total
    if master_process:
        print(f"step {step:5d} | HellaSwag acc {num_correct_norm}/{num_total}={acc_norm:.4f}")
        with open(log_file, "a") as f:
            f.write(f"step {step:5d} | HellaSwag acc {num_correct_norm}/{num_total}={acc_norm:.4f}\n")

def generate_samples():
    model.eval()
    num_return_sequences = 4 # TODO: make these configurable parameters
    max_length = 32
    tokens = enc.encode("Hello, I'm a language model,") # TODO: make this a parameter
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens.to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42 + ddp_rank)

    while xgen.size(1) < max_length:
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(xgen) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            xgen = torch.cat((xgen, xcol), dim=1)

    # print the generated text
    for i in range(num_return_sequences):
        tokens = xgen[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(f"rank {ddp_rank} sample {i}: {decoded}")

# endregion

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % eval_interval == 0 or last_step:
        val_loss_accum = eval_val_loss()

        if master_process:
            print(f"step {step:5d} | val loss {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"step {step:5d} | val loss {val_loss_accum.item():.4f}\n")

            if wandb_log:
                try:
                    wandb.log({"loss/val": val_loss_accum.item()}, step = step)
                except Exception as e:
                    print(f"logging to wandb failed: {e}")

            if step > 0 and (step % save_interval == 0 or last_step):
                save_checkpoint()

    # once in a while evaluate hellaswag
    if (step % eval_interval == 0 or last_step) and (not use_compile): # TODO: fix hellaswag issue
        eval_hellaswag()

    # # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % eval_interval == 0) or last_step) and (not use_compile):
        generate_samples()

    # do one step of the optimization
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # set up DDP syncing of accumulated gradients after final microbatch
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # note that the loss is scaled by the gradient accum steps 
        # because loss.backward() adds accumulated loss grads (not mean, which is what we want)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    # clip gradients
    if grad_clip is not None:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip) # TODO: log norms per layer

    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # take an optimization steps according to loss gradient
    optimizer.step()

    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work

    # log the training stats
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        log_string = f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f} | progress: {(step/max_steps)*100:.2f}%"
        print(log_string)
        with open(log_file, "a") as f:
            f.write(log_string + "\n")

        # log to wandb
        if wandb_log:
            try:
                wandb.log(
                    {
                        "step": step,
                        "tokens": step * grad_accum_steps * ddp_world_size * micro_batch_size * max_seq_len,
                        "loss/train": loss_accum.item(),
                        # "loss/val": val_loss_accum.item(),
                        "norm": norm, # TODO: log gradient norm separated across layers
                        "lr": lr,
                        # "mfu": running_mfu * 100,  # convert to percentage
                    }, step = step
                )
            except Exception as e:
                print(f"logging to wandb failed: {e}")

if ddp:
    destroy_process_group()

# TODO: add 'resume' option
# NOTE: is this loop better than llama2.c's training loop? or worse? or in-between?
# TODO: incorporate missing features in llama2.c's training loop into this one