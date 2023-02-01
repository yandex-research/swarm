# Large-scale experiments on preemptible instances

This folder contains the scripts that are necessary to launch large-scale language modeling experiments using SWARM Parallelism.
We use a modified version of [Hivemind](https://github.com/learning-at-home/hivemind) to run our experiments.

## To reproduce

1. On a dedicated stable (yet possibly low-performance) server, install the code with `pipeline/setup.py` and
   run `python start_monitor.py`;
2. Run `setup_and_launch_server.sh` on GPU-enabled servers,
   adjusting `'{init_peer}', --min_batch_size, --max_batch_size` if needed;
3. Run `setup_and_launch_trainer.sh` on CPU-only nodes, also adjusting `'{init_peer}'` to use the same libp2p ID as
   given by the `start_monitor` script;
4. To measure throughput, you can use [`nload`](https://github.com/rolandriegel/nload).
