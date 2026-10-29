"""ROUGE and corpus diagnostics."""
def score(references,predictions):
    from rouge_score import rouge_scorer
    scorer=rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL','rougeLsum'],use_stemmer=True)
    totals={k:0. for k in ('rouge1','rouge2','rougeL','rougeLsum')}
    for ref,pred in zip(references,predictions):
        s=scorer.score(ref,pred)
        for k in totals: totals[k]+=100*s[k].fmeasure
    return {k:totals[k]/max(len(references),1) for k in totals}

def diagnostics(article,reference,prediction):
    import re
    def words(s): return re.findall(r"\b\w+\b",s.lower())
    p=words(prediction); ref=words(reference); src=set(words(article))
    bigrams=list(zip(p,p[1:])); repeated=(len(bigrams)-len(set(bigrams)))/max(len(bigrams),1)
    novel=[g for g in bigrams if g[0] not in src or g[1] not in src]
    numbers=re.findall(r"\b\d+(?:\.\d+)?\b",prediction); src_numbers=set(re.findall(r"\b\d+(?:\.\d+)?\b",article))
    return {'prediction_words':len(p),'reference_words':len(ref),'repeated_bigram_rate':repeated,'unsupported_number_rate':sum(n not in src_numbers for n in numbers)/max(len(numbers),1),'novel_bigram_percent':100*len(novel)/max(len(bigrams),1)}
