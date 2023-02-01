# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import bitsandbytes as bnb
from fairseq.optim import LegacyFairseqOptimizer, register_optimizer


@register_optimizer("lamb")
class FairseqLAMB(LegacyFairseqOptimizer):
    """LAMB optimizer."""

    def __init__(self, args, params):
        super().__init__(args)
        if args.use_bnb:
            print('using bnb lamb')
            self._optimizer = bnb.optim.LAMB8bit(params, **self.optimizer_config, max_unorm=args.max_unorm, block_wise=False)
        else:
            try:
                from apex.optimizers import FusedLAMB

                print('using apex lamb')
                self._optimizer = FusedLAMB(params, **self.optimizer_config, use_nvlamb=True)
            except ImportError:
                raise ImportError("Please install apex to use LAMB optimizer")

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--lamb-betas', default='(0.9, 0.999)', metavar='B',
                            help='betas for LAMB optimizer')
        parser.add_argument('--lamb-eps', type=float, default=1e-8, metavar='D',
                            help='epsilon for LAMB optimizer')
        parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                            help='weight decay')
        parser.add_argument('--use-bnb', action='store_true',
                            help='bnb')
        parser.add_argument('--max-unorm', default=1.0, type=float,
                            help='bnb')
        # fmt: on

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            "lr": self.args.lr[0],
            "betas": eval(self.args.lamb_betas),
            "eps": self.args.lamb_eps,
            "weight_decay": self.args.weight_decay,
        }

    @property
    def supports_flat_params(self):
        return False
