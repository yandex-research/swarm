import random
from collections import defaultdict

from datasets import load_dataset, interleave_datasets, disable_progress_bar
from transformers import GPT2TokenizerFast

MAX_SEQ_LENGTH = 2048

disable_progress_bar()

GPT2TokenizerFast.max_model_input_sizes['gpt2'] = 1e20
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '<PAD>'})
tokens_to_add = 128 - (len(tokenizer) % 128)
tokenizer.add_special_tokens({'additional_special_tokens': [f'〈special{i}〉' for i in range(tokens_to_add)]})


def split_list(l, n):
    # splits list/string into n size chunks
    return (l[i:i + n] for i in range(0, len(l), n))


def process_instance(text, max_seq_length):
    tokenized_text = tokenizer.encode(text) + [tokenizer.eos_token_id]

    for chunk in split_list(tokenized_text, max_seq_length):
        yield chunk


def examples_from_documents(documents, max_seq_length=MAX_SEQ_LENGTH):
    texts = (text for text in documents["text"] if len(text) > 0 and not text.isspace())

    new_examples = defaultdict(list)

    for text in texts:
        instances = process_instance(text, max_seq_length)

        for instance in instances:
            new_examples['input_ids'].append(instance)

    return new_examples


def get_pile_dataset(seed, shards_to_choose):
    shards = random.Random(seed).choices(range(30), k=shards_to_choose)

    dsets = [
        load_dataset("json", data_files=f"https://the-eye.eu/public/AI/pile/train/{shard:02}.jsonl.zst",
                     streaming=True, split="train") for shard in shards
    ]

    pile = interleave_datasets(dsets)
    shuffled_pile = pile.shuffle(buffer_size=100, seed=seed)
    tokenized_pile = shuffled_pile.map(examples_from_documents, batched=True, batch_size=4)
    tokenized_pile = tokenized_pile.with_format('torch')
    return tokenized_pile
