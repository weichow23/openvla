# Ours

```bash
# Create and activate conda environment
conda create -n openvla python=3.10 -y
conda activate openvla

pip3 install -e .
pip3 install packaging ninja
# pip3 install --upgrade torch torchvision torchaudio
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip3 install "flash-attn==2.5.5" --no-build-isolation
```

train

```shell
export hf_token=xxx
PYTHONPATH='./' torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
  --data_root_dir '<openx-path>' \  # download from https://huggingface.co/datasets/WeiChow/VLATrainingDataset
  --run_root_dir './runs/'
```
data_root_dir 为bridge_orig/1.0.0这样的文件的父目录


# OpenVLA: An Open-Source Vision-Language-Action Model

[**Getting Started**](#getting-started) | [**Pretrained VLAs**](#pretrained-vlas) | [**Installation**](#installation) | [**Fine-Tuning OpenVLA via LoRA**](#fine-tuning-openvla-via-lora) | [**Fully Fine-Tuning OpenVLA**](#fully-fine-tuning-openvla) |
[**Training VLAs from Scratch**](#training-vlas-from-scratch) | [**Evaluating OpenVLA**](#evaluating-openvla) | [**MiniVLA Details**](#minivla) | [**Action Chunking**](#action-chunking) | [**Multi-Image**](#multi-image) | [**Project Website**](https://openvla.github.io/)


<hr style="border: 2px solid gray;"></hr>

## Fine-Tuning OpenVLA via LoRA

In this section, we discuss fine-tuning OpenVLA using Low-Rank Adaptation (LoRA) via the Hugging Face `transformers` library,
which is recommended if you do not have sufficient compute to fully fine-tune a 7B-parameter model. The main script for LoRA
fine-tuning is `vla-scripts/finetune.py`. (If you instead wish to do full fine-tuning, please see the
[Fully Fine-Tuning OpenVLA](#fully-fine-tuning-openvla) section.)

Below we show an example of how you can fine-tune the main OpenVLA checkpoint ([`openvla-7b`](https://huggingface.co/openvla/openvla-7b))
via LoRA. Here we fine-tune on [BridgeData V2](https://rail-berkeley.github.io/bridgedata/) using a single A100
GPU with 80 GB VRAM. (You can also fine-tune with a smaller GPU, as long as it has at least ~27 GB of memory,
by modifying the batch size.)

First, download the BridgeData V2 dataset:

```bash
# Change directory to your base datasets folder
cd <PATH TO BASE DATASETS DIR>

# Download the full dataset (124 GB)
wget -r -nH --cut-dirs=4 --reject="index.html*" https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/

# Rename the dataset to `bridge_orig` (NOTE: Omitting this step may lead to runtime errors later)
mv bridge_dataset bridge_orig
```

Now, launch the LoRA fine-tuning script, as shown below. Note that `--batch_size==16` with `--grad_accumulation_steps==1`
requires ~72 GB GPU memory. If you have a smaller GPU, you should reduce `--batch_size` and increase `--grad_accumulation_steps`
to maintain an effective batch size that is large enough for stable training. If you have multiple GPUs and wish to train via
PyTorch Distributed Data Parallel (DDP), simply set `--nproc-per-node` in the `torchrun` command below to the number of available GPUs.

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir <PATH TO BASE DATASETS DIR> \
  --dataset_name bridge_orig \
  --run_root_dir <PATH TO LOG/CHECKPOINT DIR> \
  --adapter_tmp_dir <PATH TO TEMPORARY DIR TO SAVE ADAPTER WEIGHTS> \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug <True or False> \
  --wandb_project <PROJECT> \
  --wandb_entity <ENTITY> \
  --save_steps <NUMBER OF GRADIENT STEPS PER CHECKPOINT SAVE>
```

Note: If you set `--image_aug==False` in the command above, you will observe nearly 100% `action_accuracy` in the training logs,
since the [`openvla-7b`](https://huggingface.co/openvla/openvla-7b) model is already pretrained (without augmentations) on a
superset of datasets that includes BridgeData V2.

To LoRA fine-tune on a different dataset, you can download the dataset from the [Open X-Embodiment (OXE)](https://robotics-transformer-x.github.io/)
mixture (see [this custom script](https://github.com/moojink/rlds_dataset_mod/blob/main/prepare_open_x.sh) for an example of how to download datasets
from OXE). Alternatively, if you have a custom dataset that is not part of OXE, you can either (a) convert the dataset to the RLDS format which is
compatible with our fine-tuning script (see [this repo](https://github.com/kpertsch/rlds_dataset_builder) for instructions on this), or (b) use your own
custom PyTorch Dataset wrapper (see comments in `vla-scripts/finetune.py` for instructions). We recommend option (a) for most users; the RLDS dataset and
dataloader are tested more extensively since we used these for all of our pretraining and fine-tuning experiments.

For option (a), after you converted your dataset to RLDS, you need to register it with our data loader, by registering a dataset
config [here](prismatic/vla/datasets/rlds/oxe/configs.py#L54) and a dataset transform function [here](prismatic/vla/datasets/rlds/oxe/transforms.py#L828).

Once you have integrated your new dataset, you can launch LoRA fine-tuning with the same `vla-scripts/finetune.py` script above. If you run into any issues,
please visit the [VLA Troubleshooting](#vla-troubleshooting) section or search for a similar issue in the [OpenVLA GitHub Issues page](https://github.com/openvla/openvla/issues?q=)
(including "Closed" issues). If you cannot find a similar issue there, feel free to create a new issue.

## Fully Fine-Tuning OpenVLA

In this section, we discuss <ins>fully fine-tuning</ins> OpenVLA (all 7.5 billion parameters) via native PyTorch Fully Sharded Data Parallel (FSDP)
using the [Prismatic VLMs](https://github.com/TRI-ML/prismatic-vlms) training script. Full fine-tuning is more advanced/involved and is only recommended
if you have sufficient compute (e.g., a full node of 8 A100 GPUs) and if LoRA fine-tuning is insufficient for your use case (e.g., if the fine-tuning distribution
varies drastically from the pretraining distribution). Otherwise, we recommend that you try parameter-efficient fine-tuning via LoRA, which is described in the
[Fine-Tuning OpenVLA via LoRA](#fine-tuning-openvla-via-lora) section.

For full fine-tuning, you will need to download [a different version of the OpenVLA model checkpoint](https://huggingface.co/openvla/openvla-7b-prismatic) that is compatible
with the Prismatic VLMs codebase, which we built on top of to develop the OpenVLA model. You can download this Prismatic-compatible OpenVLA checkpoint using the git commands below
(alternatively, you can download via the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli)):

```bash
# Change directory to your base model checkpoints folder
cd <PATH TO BASE MODEL CHECKPOINTS DIR>

# Download checkpoint (30 GB) -- may take a few minutes
git clone git@hf.co:openvla/openvla-7b-prismatic

# If the command above did not download the full checkpoint,
# manually fetch it via git Large File Storage (LFS)
# Note: You may have to configure an SSH key for this to work
cd openvla-7b-prismatic
git lfs fetch --all
```

We show how you can fully fine-tune OpenVLA on [BridgeData V2](https://rail-berkeley.github.io/bridgedata/) using a single node with 8 GPUs. If you wish to
use a different number of GPUs (or nodes), you can modify the VLA training configuration in [`prismatic/conf/vla.py`](prismatic/conf/vla.py).

Download the BridgeData V2 dataset:

```bash
# Change directory to your base datasets folder
cd <PATH TO BASE DATASETS DIR>

# Download the full dataset (124 GB)
wget -r -nH --cut-dirs=4 --reject="index.html*" https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/

# Rename the dataset to `bridge_orig` (NOTE: Omitting this step may lead to runtime errors later)
mv bridge_dataset bridge_orig
```

Next, create a [Hugging Face user access token](https://huggingface.co/docs/hub/en/security-tokens) and copy the token value (a string that starts with
`hf_...`) into a file named `.hf_token` at the root directory of this repo (`openvla/.hf_token`).

```bash
# Go to openvla root directory
cd openvla

# Copy HF token value into token file. Replace "hf_..." with your own token value!
# See: https://huggingface.co/docs/hub/en/security-tokens
echo hf_... >>> .hf_token
```

Now, launch the training script. If you wish to use a different number of nodes or GPUs, modify the VLA training configuration in
[`prismatic/conf/vla.py`](prismatic/conf/vla.py) and then change the `--nnodes` and `--nproc-per-node` arguments below accordingly.

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
  --pretrained_checkpoint <PATH TO openvla/openvla-7b-prismatic CHECKPOINT FILE: step-295000-epoch-40-loss=0.2200.pt> \
  --vla.type prism-dinosiglip-224px+mx-bridge \
  --data_root_dir <PATH TO BASE DATASETS DIR> \
  --run_root_dir <PATH TO LOG/CHECKPOINT DIR> \
  --run_id <OPTIONAL RUN ID FOR WANDB LOGGING> \
  --image_aug <True or False> \
  --wandb_project <PROJECT> \
  --wandb_entity <ENTITY> \
  --save_interval <NUMBER OF GRADIENT STEPS PER CHECKPOINT SAVE> \
  --is_resume False
```

Note that the `--is_resume` argument is set to `False` above since we are fine-tuning a pretrained checkpoint rather than resuming a paused training run.

If your training run gets paused and you wish to resume from the latest checkpoint, change `--pretrained_checkpoint` to the latest checkpoint path,
and then set `--is_resume==True` and specify `--resume_step` and `--resume_epoch` as the step and epoch number, respectively. For example, if you wish to
resume training from a checkpoint named `step-010000-epoch-20-loss=0.0160.pt`, you would set `is_resume==True`, `resume_step==10000`, and `resume_epoch==20`.

Note: If you run the BridgeData V2 fine-tuning command above, you should observe nearly 100% Action Token Accuracy in the training logs, since the
[`openvla-7b`](https://huggingface.co/openvla/openvla-7b) model is already pretrained on a superset of datasets that includes BridgeData V2.

To fully fine-tune OpenVLA on a different dataset, you can download the dataset from the [Open X-Embodiment (OXE)](https://robotics-transformer-x.github.io/)
mixture (see [this custom script](https://github.com/moojink/rlds_dataset_mod/blob/main/prepare_open_x.sh) for an example of how to download datasets from OXE).
Alternatively, if you have a custom dataset that is not part of OXE, you can convert the dataset to the RLDS format, which is compatible with our fine-tuning script
(see [this repo](https://github.com/kpertsch/rlds_dataset_builder) for instructions on this). After downloading/converting the dataset, you will need to modify the following files:

* [`prismatic/conf/vla.py`](prismatic/conf/vla.py): Add a new training configuration by creating an experiment class, and then register it in the `VLARegistry` at the bottom of the file.
  * Make sure to create a new unique `vla_id` for your fine-tuning run, and adjust some configuration variables as needed – e.g., `expected_world_size` (number of GPUs),
  `per_device_batch_size` (batch size per GPU), `global_batch_size` (total batch size), `shuffle_buffer_size` (number of samples in shuffle buffer per GPU), etc. See comments
  under the `VLAConfig` class at the top of the file to understand the purpose of each variable.
* [`prismatic/vla/datasets/rlds/oxe/mixtures.py`](prismatic/vla/datasets/rlds/oxe/mixtures.py): Define a new mixture for your fine-tuning mixture in the `OXE_NAMED_MIXTURES` dictionary.
* [`prismatic/vla/datasets/rlds/oxe/transforms.py`](prismatic/vla/datasets/rlds/oxe/transforms.py): Define a new dataset transform function for your fine-tuning dataset, and add it to the
`OXE_STANDARDIZATION_TRANSFORMS` registry at the bottom of the file.
* [`prismatic/vla/datasets/rlds/oxe/configs.py`](prismatic/vla/datasets/rlds/oxe/configs.py): Add a new configuration specifying your fine-tuning dataset's observation and action spaces
to the `OXE_DATASET_CONFIGS` dictionary.

After completing the steps above, you can start full fine-tuning using the `vla-scripts/train.py` script. Make sure to set the `--vla.type` argument to the new `vla_id` that you added in `prismatic/conf/vla.py`.

When you are finished with fine-tuning, you will need to convert the final model checkpoint to a version that is
compatible with the Hugging Face `transformers` library. See the [Converting Prismatic Models to Hugging Face](#converting-prismatic-models-to-hugging-face) section for instructions.

If you run into any issues, please visit the [VLA Troubleshooting](#vla-troubleshooting) section or search for a similar issue in the
[OpenVLA GitHub Issues page](https://github.com/openvla/openvla/issues?q=) (including "Closed" issues). If you cannot find a similar issue there, feel free to create a new issue.

### Converting Prismatic Models to Hugging Face

**NOTE: Converting and deploying MiniVLA models and VQ / multi image is not supported yet!**

If you have used the Prismatic VLMs codebase to train your model (e.g., if you did full fine-tuning of OpenVLA on a
new dataset), you will need to convert the final checkpoint to a version that is compatible with Hugging Face
`transformers` AutoClasses. We discuss how to do so in this section.

Let's say your training run directory is `PRISMATIC_RUN_DIR` (e.g., `prism-dinosiglip-224px+mx-oxe-magic-soup-plus+n8+b32+x7`).
Inside this directory, there should be a directory called `checkpoints` which contains saved model checkpoints (e.g.,
`step-295000-epoch-40-loss=0.2200.pt`). The Prismatic-to-Hugging-Face conversion script
([convert_openvla_weights_to_hf.py](vla-scripts/extern/convert_openvla_weights_to_hf.py)) expects a checkpoint file
named `latest-checkpoint.pt`. Therefore, you should first create a symbolic link called `latest-checkpoint.pt` that
points to the checkpoint file that you wish to convert:

```bash
# Go to your Prismatic training run's `checkpoints` directory
cd PRISMATIC_RUN_DIR/checkpoints

# Create symbolic link pointing to your checkpoint file
ln -s <YOUR CHECKPOINT FILENAME> latest-checkpoint.pt
```

Then, launch the conversion script to convert the checkpoint from the Prismatic VLMs format to the Hugging Face format:

```bash
python vla-scripts/extern/convert_openvla_weights_to_hf.py \
    --openvla_model_path_or_id <PRISMATIC_RUN_DIR> \
    --output_hf_model_local_path <OUTPUT DIR FOR CONVERTED CHECKPOINT>
```

The command above will save the HF-compatible checkpoint in `output_hf_model_local_path`. Now you can load the checkpoint
with HF AutoClasses as normal, as shown below. Note that there is an additional necessary step to register the OpenVLA model
to HF AutoClasses before loading it because you are loading a locally saved checkpoint rather than one that is pushed to the
HF Hub (see [here](https://huggingface.co/docs/transformers/en/custom_models#registering-a-model-with-custom-code-to-the-auto-classes)
for details).

```python
import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Register OpenVLA model to HF AutoClasses (not needed if you pushed model to HF Hub)
AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

# Load Processor & VLA
processor = AutoProcessor.from_pretrained("<PATH TO CONVERTED CHECKPOINT DIR>", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "<PATH TO CONVERTED CHECKPOINT DIR>",
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).to("cuda:0")

...
```

## Training VLAs from Scratch

We provide full instructions and configurations for training VLA models on (arbitrary subsets of) the
[Open X-Embodiment (OXE) Dataset](https://robotics-transformer-x.github.io/). If you run in to any issues with
the following, see [VLA Troubleshooting](#vla-troubleshooting) below (or file a GitHub Issue).

### VLA Pretraining Datasets

We download and preprocess individual datasets from Open X-Embodiment in [RLDS format](https://github.com/google-research/rlds) following
[this custom script](https://github.com/moojink/rlds_dataset_mod/blob/main/prepare_open_x.sh). See
[mixtures.py](./prismatic/vla/datasets/rlds/oxe/mixtures.py) for the full list of component datasets (and mixture
weights) we use to train `openvla-7b`.
- **Important**: For the BridgeData V2 component, the version in OXE is out of date (as of 12/20/2023). Instead,
  you should download the dataset from the [official website](https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/) and place it under the subdirectory `bridge_orig/`.
  Replace any reference to `bridge` in the OXE code with `bridge_orig`.

### VLA Configuration & Training Script

The entry point for VLA training is [`vla-scripts/train.py`](vla-scripts/train.py). We use
[`draccus`](https://pypi.org/project/draccus) to provide a modular, dataclass-based interface for specifying VLA
training configurations; existing VLA configurations are in [`prismatic/conf/vla.py`](prismatic/conf/vla.py). You can
add your own training configuration and refer to it using the `--vla.type` command line argument.

We use PyTorch Fully Sharded Data Parallel (FSDP) to distribute training across GPUs. Launch training via `torchrun`:

```bash
# Train VLA on BridgeData V2 with the Prismatic DINO-SigLIP 224px Backbone on a Single Node (w/ 8 GPUs)
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
  --vla.type "prism-dinosiglip-224px+mx-bridge" \
  --data_root_dir <PATH TO OXE DATA ROOT> \
  --run_root_dir <PATH TO LOG/CHECKPOINT ROOT> \
  --wandb_project "<PROJECT>" \
  --wandb_entity "<ENTITY>"
```

### VLA Troubleshooting

The following are a list of known problems and corresponding fixes:

```bash
FileNotFoundError: Failed to construct dataset "fractal20220817_data", builder_kwargs "{'data_dir': '/path/to/processed/datasets/'}": Could not load dataset info from fractal20220817_data/0.1.0/dataset_info.json
```
- **Fix**: Downgrade `tensorflow-datasets` via `pip install tensorflow-datasets==4.9.3`.


```bash
AttributeError: 'DLataset' object has no attribute 'traj_map'. Did you mean: 'flat_map'?
```
- **Fix**: Upgrade `dlimp` to the newest version. You may have to `--force-reinstall` like so:
`pip install --no-deps --force-reinstall git+https://github.com/moojink/dlimp_openvla`

---

## MiniVLA


### Base Prismatic VLM for Qwen2.5

We have included a variant Prismatic VLM trained one the [Qwen2.5 0.5B backbone here](https://huggingface.co/Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b).
The configurations for training the Prismatic VLM can be found under `prismatic/conf/models.py` with `Qwen2.5` in the name.

The recommended Base VLM config to use for pretraining is `--model.type prism-qwen25-extra-dinosiglip-224px+0_5b`.
Here, "extra" corresponds to adding an extra 256 tokens to the Qwen Vocabulary `<extra_i> ... ` to deconflict later action prediction in MiniVLA.

Here's an example of how to train, following the prismatic guidelines with 8 GPUs:

```bash
# Run from the root of the repository
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
  --model.type "prism-qwen25-extra-dinosiglip-224px+0_5b" \
```

You can find this model [here on hugging face](https://huggingface.co/Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b)


### Training MiniVLA on your dataset

Training configs for MiniVLA using the above base VLM can be found under `prismatic/conf/vla.py` with `Qwen2.5` in the name.

IMPORTANT: All Qwen2.5 backbone VLAs should use `vla.action_tokenizer = extra_action_tokenizer`, which is the default in all the configs now.
This uses the newly added 256 extra tokens for VLA action binning, instead of the last 256 tokens in the vocabulary (`action_tokenizer` uses this).

Here's an example of how to train MiniVLA on Libero-90:

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
  --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90" \
  --data_root_dir <PATH TO LIBERO DATA ROOT> \
  --run_root_dir <PATH TO LOG/CHECKPOINT ROOT> \
  --wandb_project "<PROJECT>" \
  --wandb_entity "<ENTITY>"
```

We have [LIBERO pretrained models here](https://huggingface.co/collections/Stanford-ILIAD/minivla-675a2a9aca369ff3a6c04e33)

---

## Action Chunking

To implement action chunking, we use the VQ-Bet implementation of Residual VQ. Follow the [installation instructions here](https://github.com/jayLEE0301/vq_bet_official).

### How does it work?

Rather than naively predict action chunks by adding more output tokens, we *compress* the continuous action chunks of shape `(H x A)` using Residual VQ into a set `T` of "code words", each of which is an N dimensional latent vector. We use the unique integer index of each code work as the `bins` for action prediction. Thus we compress `(H x A)` continuous values into a set of `T` discrete bins for the model to predict. Residual VQ is especially useful because it is "autoregressively" encoded, which matches the autoregressive decoding that we do with action prediction in VLAs.

### Pretraining VQ for your dataset

To use VQ style chunking, first you should train a Residual VQ network. We provide the `vla-scripts/pretrain_vq.py` script as a wrapper around the VQ-Bet Residual VQ implementation for training from arbitrary tfds datasets and any mixtures defined in this codebase.

For example, here's how to pretrain the VQ on bridge (`data_mix=bridge_dataset`) with an action horizon of 8 (`future_action_horizon=7`), input action dimension of 7 (`action_dim=7`) and the number of codewords (i.e. the number of discrete bins for each output action) as 256 (`vqvae_n_embed=256`):

```bash
# from root dir
python vla-scripts/pretrain_vq.py --data_dir <WHERE IS BRIDGE DATA> \\
        --data_mix bridge_dataset --action_dim 7 --future_action_horizon 7 --vqvae_n_embed 256
```

By default, this will create a directory under the root called `vq/` for each of the pretrained checkpoint.

We have several existing VQs that you can use, [LIBERO VQ](https://huggingface.co/Stanford-ILIAD/pretrain_vq/tree/main/pretrain_vq%2Bmx-libero_90%2Bfach-7%2Bng-7%2Bnemb-128%2Bnlatent-512) and [Bridge VQ](https://huggingface.co/Stanford-ILIAD/pretrain_vq/tree/main/pretrain_modvq%2Bmx-bridge_dataset%2Bfach-7%2Bng-7%2Bnemb-256%2Bnlatent-512). Download them into this vq/ folder before running VLA training or inference.

### Training and Inference with VQ

Once you've downloaded an existing VQ or pretrained your own, you can train a VQ enabled model just by:
1. Specify the new action tokenizer in `ACTION_TOKENIZERS` in `prismatic/vla/action_tokenizer.py` similarly to the VQ `libero_vq_action_tokenizer`, for example if we wanted to add a new one for vq path `vq/my_vq_model` that uses the same action bins as the base `action_tokenizer`, just append:

```python
  'my_action_tokenizer_name': partial(
      VQActionTokenizer, vq_vae_path="vq/my_vq_model"
  ),
  ...
```

**NOTE**: For MiniVLA and Qwen LLM backbones, you should use the `use_extra=True` when defining tokenizers to use the extra tokens as our action bins instead of the last 256 like other models use.

2. Pass in `--vla.action_tokenizer my_action_tokenizer_name` to the train or finetune script to use this action tokenizer instead.

---

## Multi Image

We have added native support for multi images into the code base. Some examples on libero in `prismatic/conf/vla.py`:
1. `prism-qwen25-dinosiglip-224px-t2+0_5b+mx-libero-90`: History of 2
2. `prism-qwen25-dinosiglip-224px-wrist+0_5b+mx-libero-90`: Uses wrist images

Both of these can be used at launch by updating the `--vla.type ...` argument.

The implementation just uses the same visual encoder, but runs it on multiple images, then concatenates those token sequences to pass into the LLM backbone, similarly to single images.

**NOTE** Images are expected to be passed in at inference in the exact same order & number as training, since there are no "prefix" tags denoting which image is which in the token sequence that gets passed into the image, its just one big concatenated sequence of `<img1_tokens> <img2_tokens> ...` before the prompt.

**NOTE**: If you want to add your own config with multi-image, make sure to specify `image_sequence_len` to match the number of *raw images* expected (e.g., 2 for both history=2 and wrist example above). If you use wrist images, specify `vla.use_wrist_image=True` in the config, which is already done in the above config.

---

## Evaluating OpenVLA

### BridgeData V2 WidowX Evaluations

#### Setup

Clone the [BridgeData V2 WidowX controller repo](https://github.com/rail-berkeley/bridge_data_robot) and install the `widowx_envs` package:

```bash
git clone https://github.com/rail-berkeley/bridge_data_robot.git
cd bridge_data_robot
pip install -e widowx_envs
```

Additionally, install the [`edgeml`](https://github.com/youliangtan/edgeml) library:
```bash
git clone https://github.com/youliangtan/edgeml.git
cd edgeml
pip install -e .
```

Follow the instructions in the `bridge_data_robot` README to create the Bridge WidowX Docker container.

#### Launching BridgeData V2 Evaluations

There are multiple ways to run BridgeData V2 evaluations. We describe the server-client method below.

In one Terminal window (e.g., in tmux), start the WidowX Docker container:

```bash
cd bridge_data_robot
./generate_usb_config.sh
USB_CONNECTOR_CHART=$(pwd)/usb_connector_chart.yml docker compose up --build robonet
```

In a second Terminal window, run the WidowX robot server:

```bash
cd bridge_data_robot
docker compose exec robonet bash -lic "widowx_env_service --server"
```

In a third Terminal window, run the OpenVLA policy evaluation script:

```bash
cd openvla
python experiments/robot/bridge/run_bridgev2_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b
```

If you run into any problems with evaluations, please file a GitHub Issue.


### LIBERO Simulation Benchmark Evaluations

In the [updated OpenVLA paper (v2)](https://arxiv.org/abs/2406.09246), we discuss fine-tuning OpenVLA
on a simulated benchmark, [LIBERO](https://libero-project.github.io/main.html), in Appendix E.
Please see the paper for details, such as how we modify the provided demonstration datasets to
improve the overall performance of all methods.

We copy the results to the section below and then discuss how to reproduce the results for OpenVLA.

#### OpenVLA Fine-Tuning Results

| Method | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal | LIBERO-Long | Average |
|--------|----------------|---------------|-------------|-------------|---------|
| Diffusion Policy from scratch | 78.3 ± 1.1% | **92.5 ± 0.7%** | 68.3 ± 1.2% | 50.5 ± 1.3% | 72.4 ± 0.7% |
| Octo fine-tuned | 78.9 ± 1.0% | 85.7 ± 0.9% | **84.6 ± 0.9%** | 51.1 ± 1.3% | 75.1 ± 0.6% |
| OpenVLA fine-tuned (ours) | **84.7 ± 0.9%** | 88.4 ± 0.8% | 79.2 ± 1.0% | **53.7 ± 1.3%** | **76.5 ± 0.6%** |

Each success rate is the average over 3 random seeds x 500 rollouts each (10 tasks x 50 rollouts per task).

#### LIBERO Setup

Clone and install the [LIBERO repo](https://github.com/Lifelong-Robot-Learning/LIBERO):

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
```

Additionally, install other required packages:
```bash
cd openvla
pip install -r experiments/robot/libero/libero_requirements.txt
```

(Optional) To download the modified versions of the LIBERO datasets that we used in our fine-tuning
experiments, run the command below. This will download the LIBERO-Spatial, LIBERO-Object, LIBERO-Goal,
and LIBERO-10 datasets in RLDS data format (~10 GB total). You can use these to fine-tune OpenVLA or
train other methods. This step is optional since we provide pretrained OpenVLA checkpoints below.
(Also, you can find the script we used to generate the modified datasets in raw HDF5 format
[here](experiments/robot/libero/regenerate_libero_dataset.py) and the code we used to convert these
datasets to the RLDS format [here](https://github.com/moojink/rlds_dataset_builder).)
```bash
git clone git@hf.co:datasets/openvla/modified_libero_rlds
```

#### Launching LIBERO Evaluations

We fine-tuned OpenVLA via LoRA (r=32) on four LIBERO task suites independently: LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-10 (also called LIBERO-Long).
The four checkpoints are available on Hugging Face:
* [openvla/openvla-7b-finetuned-libero-spatial](https://huggingface.co/openvla/openvla-7b-finetuned-libero-spatial)
* [openvla/openvla-7b-finetuned-libero-object](https://huggingface.co/openvla/openvla-7b-finetuned-libero-object)
* [openvla/openvla-7b-finetuned-libero-goal](https://huggingface.co/openvla/openvla-7b-finetuned-libero-goal)
* [openvla/openvla-7b-finetuned-libero-10](https://huggingface.co/openvla/openvla-7b-finetuned-libero-10)

To start evaluation with one of these checkpoints, run one of the commands below. Each will automatically download the appropriate checkpoint listed above.

```bash
# Launch LIBERO-Spatial evals
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True

# Launch LIBERO-Object evals
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-object \
  --task_suite_name libero_object \
  --center_crop True

# Launch LIBERO-Goal evals
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --center_crop True

# Launch LIBERO-10 (LIBERO-Long) evals
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \
  --task_suite_name libero_10 \
  --center_crop True
```

Notes:
* The evaluation script will run 500 trials by default (10 tasks x 500 episodes each). You can modify the number of
  trials per task by setting `--num_trials_per_task`. You can also change the random seed via `--seed`.
* **NOTE: Setting `--center_crop True` is important** because we fine-tuned OpenVLA with random crop augmentations
  (we took a random crop with 90% area in every training sample, so at test time we simply take the center 90% crop).
* The evaluation script logs results locally. You can also log results in Weights & Biases
  by setting `--use_wandb True` and specifying `--wandb_project <PROJECT>` and `--wandb_entity <ENTITY>`.
* The results reported in our paper were obtained using **Python 3.10.13, PyTorch 2.2.0, transformers 4.40.1, and
  flash-attn 2.5.5** on an **NVIDIA A100 GPU**, averaged over three random seeds. Please stick to these package versions.
  Note that results may vary slightly if you use a different GPU for evaluation due to GPU nondeterminism in large models
  (though we have tested that results were consistent across different machines with A100 GPUs).

Please file a GitHub Issue if you run into any problems.

---

## Repository Structure

High-level overview of repository/project file-tree:

+ `prismatic` - Package source; provides core utilities for model loading, training, data preprocessing, etc.
+ `vla-scripts/` - Core scripts for training, fine-tuning, and deploying VLAs.
+ `experiments/` - Code for evaluating OpenVLA policies in robot environments.
+ `LICENSE` - All code is made available under the MIT License; happy hacking!
+ `Makefile` - Top-level Makefile (by default, supports linting - checking & auto-fix); extend as needed.
+ `pyproject.toml` - Full project configuration details (including dependencies), as well as tool configurations.
+ `README.md` - You are here!

---


# VLA Performance Troubleshooting

In this section we cover best practices for debugging poor VLA performance after fine-tuning on your target domain robot dataset.

**Note**: OpenVLA typically requires fine-tuning on a small demonstration dataset (~100 demos) from your target domain robot. Out-of-the-box, it only works well on domains from the training dataset.

**Sanity checks**:
- replay the actions from a demonstration from your fine-tuning dataset and make sure that the robot can execute the task successfully (this ensures that your data collection pipeline is correct)
- once you fine-tuned a model, load the model in your inference pipeline (as if you would run it to control the robot), but feed images from the fine-tuning dataset into the model (pretending they come from the robot) and verify that you can reproduce the token accuracies / L1 errors from training (this ensures that your inference pipeline is correct)

**Best practices for fine-tuning data collection**:
If your setup passed the above two sanity checks, the issue may not be in model training, but in the data you fine-tuned the model with. Some best practices for data collection:
- *Collect at a control frequency around 5-10Hz.* OpenVLA is not trained with action chunking, empirically the model struggles with high-frequency data. If your robot setup uses a high-frequency controller (eg 50 Hz), consider downsampling your actions to 5Hz. Verify first that your robot can still solve the task when using 5Hz actions (ie repeat sanity check (1) above with 5Hz actions)
- *Avoid pauses / small actions during data collection.* Because OpenVLA is trained without action chunking, the model can be sensitive to idle actions in the fine-tuning data. If your data contains steps in which the robot barely moves, the model may "get stuck" in these steps at inference time. Try to collect fine-tuning demonstrations with continuous, slow movement.
- *Ensure sufficient data coverage.* If you plan to test the model with some variation, e.g. different initial positions of objects, make sure that your fine-tuning data contains sufficient diversity of such conditions as well, e.g. shows demonstrations with diverse initial conditions.
- *Use consistent task strategies during data collection.* This is not a hard constraint, but may make your life easier. Try to demonstrate tasks in consistent ways, e.g. approach objects from the same side, perform sub-steps in the same order even if they could be performed in arbitrary sequences. Being consistent gives you a less multi-modal fine-tuning dataset, which makes the modeling problem easier
