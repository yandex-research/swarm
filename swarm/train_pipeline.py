import weakref
from collections import defaultdict
from functools import wraps
from socket import gethostname

import torch
from src import BalancedRemoteExpert, DHT
from src.dht.crypto import RSASignatureValidator
from src.utils import get_logger, use_src_log_handler, get_dht_time
from torch.optim.lr_scheduler import _LRScheduler
from transformers import DataCollatorForLanguageModeling, HfArgumentParser, GPT2TokenizerFast, TrainingArguments, \
    Trainer, TrainerState, TrainerControl, TrainerCallback, set_seed

from arguments import CollaborationArguments
from pile_streaming import get_pile_dataset
from utils import LocalMetrics

use_src_log_handler("in_root_logger")
logger = get_logger(__name__)

TIMEOUT = 30  # averaging timeout + offset
BACKWARD_TIMEOUT = TIMEOUT * 2.5


class TrainerModel(torch.nn.Module):
    def __init__(self, grid_size, dht):
        super().__init__()

        self.dummy = torch.nn.Parameter(torch.randn(1, 1), requires_grad=True)

        self.head = BalancedRemoteExpert(grid_size=grid_size, dht=dht,
                                         forward_timeout=TIMEOUT, backward_timeout=BACKWARD_TIMEOUT,
                                         uid_prefix='head.')

        self.body1 = BalancedRemoteExpert(grid_size=grid_size, dht=dht,
                                          forward_timeout=TIMEOUT, backward_timeout=BACKWARD_TIMEOUT,
                                          uid_prefix='body1.')

        self.body2 = BalancedRemoteExpert(grid_size=grid_size, dht=dht,
                                          forward_timeout=TIMEOUT, backward_timeout=BACKWARD_TIMEOUT,
                                          uid_prefix='body2.')

        self.tail = BalancedRemoteExpert(grid_size=grid_size, dht=dht,
                                         forward_timeout=TIMEOUT, backward_timeout=BACKWARD_TIMEOUT,
                                         uid_prefix='tail.')

    def forward(self, input_ids, **kwargs):
        hidden = self.head(input_ids)
        hidden = self.body1(hidden)
        hidden = self.body2(hidden)
        loss = self.tail(hidden, input_ids)

        loss = loss.mean()

        return loss, None


class NoOpOptimizer(torch.optim.Optimizer):
    def __init__(self, params, defaults):
        torch._C._log_api_usage_once("python.optimizer")
        self.defaults = defaults

        self._hook_for_profile()

        if isinstance(params, torch.Tensor):
            raise TypeError("params argument given to the optimizer should be "
                            "an iterable of Tensors or dicts, but got " +
                            torch.typename(params))

        self.state = defaultdict(dict)
        self.param_groups = []

        param_groups = list(params)
        if len(param_groups) != 0:
            if not isinstance(param_groups[0], dict):
                param_groups = [{'params': param_groups}]

            for param_group in param_groups:
                self.add_param_group(param_group)

    def step(self, **kwargs):
        r"""Performs a single optimization step (parameter update).
        Args:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        .. note::
            Unless otherwise specified, this function should not modify the
            ``.grad`` field of the parameters.
        """
        pass


class NoOpScheduler(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1, verbose=False):

        # Attach optimizer
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        # Initialize epoch and base learning rates
        self.base_lrs = [0 for group in optimizer.param_groups]
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0
        self._step_count = 0
        self.verbose = verbose

        self.step()

    def get_lr(self):
        if self.optimizer.param_groups:
            return [0 for _ in self.optimizer.param_groups]
        else:
            return [0]

    def print_lr(self, *args, **kwargs):
        if self.optimizer.scheduler:
            return self.optimizer.scheduler.print_lr(*args, **kwargs)

    def step(self):
        self._last_lr = self.get_lr()

    def state_dict(self):
        return {}

    def load_state_dict(self, *args, **kwargs):
        pass


class CollaborativeCallback(TrainerCallback):
    """
    This callback monitors and reports collaborative training progress.
    In case of a catastrophic failure, it can also revert training to a backup.
    """

    def __init__(
        self,
        dht: DHT,
        model: torch.nn.Module,
        local_public_key: bytes,
        experiment_prefix: str,
        statistics_expiration: float,
    ):
        super().__init__()
        self.model = model
        self.dht = dht
        self.local_public_key = local_public_key
        self.experiment_prefix = experiment_prefix
        self.statistics_expiration = statistics_expiration
        self.last_reported_collaboration_step = -1

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        control.should_log = True

        if state.log_history:
            loss = state.log_history[-1]["loss"]
            if state.global_step != self.last_reported_collaboration_step:
                self.last_reported_collaboration_step = state.global_step

                statistics = LocalMetrics(
                    step=state.global_step,
                    loss=loss,
                )

                self.dht.store(
                    key=self.experiment_prefix + "_metrics",
                    subkey=self.local_public_key,
                    value=statistics.dict(),
                    expiration_time=get_dht_time() + self.statistics_expiration,
                    return_future=True,
                )

        return control


def main(training_args, collaboration_args, args):
    signature_validator = RSASignatureValidator()

    GPT2TokenizerFast.max_model_input_sizes['gpt2'] = 1e20
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    tokens_to_add = 128 - (len(tokenizer) % 128)
    tokenizer.add_special_tokens({'additional_special_tokens': [f'〈special{i}〉' for i in range(tokens_to_add)]})

    set_seed(training_args.seed)

    dataset = get_pile_dataset(seed=int(hash(signature_validator.local_public_key) % 100000), shards_to_choose=1)

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=2048)

    grid_size = (1, args.grid_size)
    dht = DHT(
        start=True,
        initial_peers=collaboration_args.initial_peers,
        client_mode=True,
        host_maddrs=collaboration_args.host_maddrs,
        announce_maddrs=collaboration_args.announce_maddrs,
    )

    hostname = gethostname()
    training_args.run_name = hostname

    model = TrainerModel(grid_size, dht)

    optimizer = NoOpOptimizer(model.parameters(), defaults=dict())

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        optimizers=(optimizer, NoOpScheduler(optimizer)),
        callbacks=[
            CollaborativeCallback(
                dht,
                model,
                hostname,
                collaboration_args.experiment_prefix,
                collaboration_args.statistics_expiration,
            )
        ],
    )

    trainer.train()


if __name__ == '__main__':
    parser = HfArgumentParser((TrainingArguments, CollaborationArguments))
    parser.add_argument('--grid_size', type=int, required=True)

    training_args, collaboration_args, args = parser.parse_args_into_dataclasses()
    main(training_args, collaboration_args, args)
