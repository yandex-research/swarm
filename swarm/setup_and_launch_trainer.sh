#!/usr/bin/env bash
set -e
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O install_miniconda.sh
bash install_miniconda.sh -b -p /opt/conda
export PATH="/opt/conda/bin:${{PATH}}"
conda install python~=3.8.0 pip
conda install pytorch cudatoolkit=11.2 -c pytorch -c nvidia

pip install --no-cache-dir -i https://test.pypi.org/simple/ bitsandbytes-cuda111==0.0.23
pip install --no-cache-dir transformers sentencepiece datasets torch_optimizer nltk zstandard wandb
sed -i '1d' /opt/conda/lib/python3.8/site-packages/bitsandbytes/optim/lamb.py
sed -i '33d' /opt/conda/lib/python3.8/site-packages/bitsandbytes/optim/lamb.py
cd ..

git clone https://github.com/yandex-research/swarm.git
cd swarm/swarm/pipeline
pip install -e .
cd ..
export PATH="/opt/conda/bin:${{PATH}}"
ulimit -n 8192
for ind in $(seq 0 3); do
  (WANDB_DISABLED=true python train_pipeline.py \
  --grid_size 128 --output_dir temp  --no_cuda --max_steps 125000000 --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 --dataloader_num_workers 1 --logging_steps 10 --experiment_prefix swarm \
  --initial_peers '{init_peer}' 2>&1 | tee -a trainer_stderr_$ind.log &);
done
