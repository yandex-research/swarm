# SWARM Parallelism: Training Large Models Can Be Surprisingly Communication-Efficient

![Illustration of SWARM parallelism](swarm.png)

This repository contains the code to replicate experiments of
["SWARM Parallelism: Training Large Models Can Be Surprisingly Communication-Efficient"](https://arxiv.org/abs/2301.11913).

**Note:** this codebase, as well as the project itself, is a work in progress: 
certain features (e.g., rebalancing between pipeline stages) are not yet added to the repository, expect the paper to get updated as well.
In the meantime, you can watch this repository or visit the [repository](https://github.com/bigscience-workshop/petals)
of [Petals](https://petals.ml/) — a similar project for *inference* of large language models that was inspired by SWARM
and shares portions of codebase with it.

# Large-scale experiments and throughput estimation

Instructions to replicate the experiments on large-scale language model pretraining and throughput estimation on
multiple preemptible nodes, as well as the prototype implementation of SWARM, are located in
the [swarm](./swarm) subfolder.

# Bottleneck experiments

Instructions to replicate the compression-aware architecture experiments can be found
in [bottleneck/README.md](bottleneck/README.md).

# Contacts

Feel free to ask any questions about this work [by email](mailto:mryabinin0@gmail.com).

# References

```
@inproceedings{ryabinin2023swarm,
  title = 	 {{SWARM} Parallelism: Training Large Models Can Be Surprisingly Communication-Efficient},
  author =       {Ryabinin, Max and Dettmers, Tim and Diskin, Michael and Borzunov, Alexander},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {29416--29440},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v202/ryabinin23a/ryabinin23a.pdf},
  url = 	 {https://proceedings.mlr.press/v202/ryabinin23a.html},
}
```
