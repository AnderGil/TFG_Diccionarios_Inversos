import torch
from torch.utils.data import IterableDataset
from functools import partial
from torch.utils.data import DataLoader

first_batch = True


def blocks(files, size=65536):
    while True:
        b = files.read(size)
        if not b:
            break
        yield b


def count_lines(input_path: str) -> int:
    with open(input_path, "r", encoding="utf8") as f:
        return sum(bl.count("\n") for bl in blocks(f))


class FileDataset(IterableDataset):
    def __init__(
        self,
        filename: str,
        tokenizer,
    ):

        self.filename = filename
        self.tokenizer = tokenizer
        self.num_lines = count_lines(filename)
        print(f"Number of lines in {filename}: {self.num_lines}")

    def prepare_sentence(self, line: str) -> (torch.tensor, torch.tensor, int, str):
        sentence, pos = line.rstrip().strip().split("|||")

        sentence = sentence.split(" ")
        gold = sentence[int(pos)]
        sentence[int(pos)] = self.tokenizer.mask_token
        sentence = " ".join(sentence)

        return (
            sentence,
            [
                self.tokenizer.encode(
                    gold, add_special_tokens=False, return_tensors="pt", max_length=512, truncation=True
                )[0][0]
            ],
            int(pos.strip()) + 1,  # +1 because we add the start of sentence token
            gold,
        )

    def __iter__(self):
        file = open(self.filename, "r", encoding="utf8")

        mapped_itr = map(
            self.prepare_sentence,
            file,
        )

        return mapped_itr

    def __getitem__(self, index):
        pass

    def __len__(self):
        return self.num_lines


def collate_fn_pad(tokenizer, batch):
    global first_batch
    sentences, golds_ids, positions, gold_words = zip(*batch)

    if first_batch:
        print(f"Sentence sample: {sentences}")
        first_batch = False

    sentence_tokens = tokenizer(
        list(sentences), return_tensors="pt", add_special_tokens=True, padding=True
    )

    return sentence_tokens, torch.tensor(golds_ids), list(positions), list(gold_words)


def get_dataloader(filename: str, tokenizer, batch_size: int, pin_memory: bool):

    collate_fn = partial(collate_fn_pad, tokenizer)

    return DataLoader(
        FileDataset(
            filename=filename,
            tokenizer=tokenizer,
        ),
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
