"""Shared WordPiece encoding while preserving highlight sentence boundaries."""
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class SpecialTokens:
    pad_id: int
    bos_id: int
    eos_id: int
    nl_id: int


class SummaryTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.special = SpecialTokens(tokenizer.pad_token_id, tokenizer.cls_token_id,
                                     tokenizer.sep_token_id, tokenizer.convert_tokens_to_ids("[unused1]"))

    def encode_article(self, text: str, max_length: int) -> List[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=max_length)
        return ids or [self.special.eos_id]

    def encode_summary(self, text: str, max_length: int) -> List[int]:
        if max_length < 2: raise ValueError("max summary length must allow BOS and EOS")
        ids = []
        for i, sentence in enumerate(text.split("\n")):
            if i: ids.append(self.special.nl_id)
            ids.extend(self.tokenizer.encode(sentence, add_special_tokens=False))
        content = ids[:max_length - 2]
        return [self.special.bos_id] + content + [self.special.eos_id]

    def decode_summary(self, ids: List[int]) -> str:
        sentences, current = [], []
        for token_id in ids:
            if token_id in (self.special.eos_id, self.special.pad_id): break
            if token_id == self.special.bos_id: continue
            if token_id == self.special.nl_id:
                sentences.append(self.tokenizer.decode(current, skip_special_tokens=True).strip()); current=[]
            else: current.append(token_id)
        sentences.append(self.tokenizer.decode(current, skip_special_tokens=True).strip())
        return "\n".join(s for s in sentences if s)
