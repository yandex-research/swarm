#!/usr/bin/env bash
set -e
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O install_miniconda.sh
bash install_miniconda.sh -b -p /opt/conda
export PATH="/opt/conda/bin:${{PATH}}"
conda install python~=3.8.0 pip
conda install pytorch cudatoolkit=11.1 -c pytorch -c nvidia

pip install --no-cache-dir -i https://test.pypi.org/simple/ bitsandbytes-cuda111==0.0.23
pip install --no-cache-dir transformers sentencepiece datasets torch_optimizer nltk
sed -i '1d' /opt/conda/lib/python3.8/site-packages/bitsandbytes/optim/lamb.py
sed -i '33d' /opt/conda/lib/python3.8/site-packages/bitsandbytes/optim/lamb.py

git clone https://github.com/yandex-research/swarm.git
cd swarm/swarm/pipeline
pip install -e .

export PATH="/opt/conda/bin:${{PATH}}"
cd /home/root
ip addr show eth0 | grep "inet .* scope global" | awk '{{print $2}}' | cut -d/ -f1 > ip_address
export IPADDR=$(cat ip_address)
ulimit -n 8192

src-server --num_experts 1 \
--expert_pattern head.0.[0:127] --expert_cls lm_head --hidden_dim 4096 --num_handlers 64 \
--scheduler linear --fp16 --stats_report_interval 60 \
--num_warmup_steps 3125 --num_total_steps 15000 --clip_grad_norm 1.0 --compression BLOCKWISE_8BIT \
--averaging_target_batch_size 4096 --averaging_expiration 60 --averaging_timeout 700 --metadata_expiration 700 \
--min_batch_size 1 --max_batch_size 1 --offload \
--device cuda:0 --listen_on $IPADDR:* --dht_listen_on ip4/$IPADDR \
--initial_peers '{init_peer}' 2>&1 | tee -a server_stderr_head_0.log;
