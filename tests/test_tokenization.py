from summarizer.tokenization import SummaryTokenizer

class MockTokenizer:
    pad_token_id=0; cls_token_id=1; sep_token_id=2
    def convert_tokens_to_ids(self,x): return 3
    def encode(self,text,add_special_tokens=False,**kw): return [ord(c)%20+4 for c in text]
    def decode(self,ids,skip_special_tokens=True): return ''.join(chr((x-4)%20+97) for x in ids)

def test_summary_sentence_split_keeps_nl():
    tok=SummaryTokenizer(MockTokenizer()); ids=tok.encode_summary('ab\ncd',20)
    assert tok.special.nl_id in ids and ids[0]==1 and ids[-1]==2
    assert tok.encode_summary('abcdefgh',5)[-1]==2
