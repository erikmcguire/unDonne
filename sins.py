#! /usr/bin/env python

from lode import *
import gramarye
from nltk.corpus import wordnet as wn, sentiwordnet as swn

def get_syn(line: list = [], o: bool = False, s: bool = False) -> float:
    """Return affect, polarity of line.
       Modified via: condor.depaul.edu/ntomuro/courses/NLP594s18
    """
    subsum, ssum = 0.0, 0.0
    to_wnt = (lambda t:  t[0].lower()
                         if t[0] != "J" else "a")
    for (w, t) in gramarye.tag_list(line):
        if t[0] in "RVNJR": # Technically Middleton...
            w = "not" if w == "n't" else w.lower()
            lemma = wn.morphy(w, to_wnt(t))
            if not lemma:
                continue
            synsets = wn.synsets(lemma, to_wnt(t))
            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())
            subsum += (1 - swn_synset.obj_score()) # Subjectivity
            ssum += (swn_synset.pos_score() - swn_synset.neg_score())
        else:
            continue
    if o: # orientation: positive or negative
        return ssum
    elif s:
        return synset.name()
    else: # 'subjectivity'
        return (subsum/(len(line) or 1.0))

def word_model(sins: list):
    """Return sentence lists, trained model."""
    try:
        model = Word2Vec.load('data/donnemod')
    except:
        model = Word2Vec(sins)
        model.save('data/donnemod')
    return model

def k_model(sins: list, mn: int = 3, mxn: int = 6):
    """Use FastText for char-level flexibility."""
    try:
        model = FastText.load('data/kmod')
    except:
        # word_ngrams = 1 (uses subword features) by default
        model = FastText(sins, min_count=1, min_n=mn, max_n=mxn)
        model.save('data/kmod')
    return model

def phrase_model(sins: list):
    """Trigram-based features."""
    try:
        model = Word2Vec.load('data/phrasesmod')
    except:
        frasier = Phrases(sins, delimiter=b' ', threshold=0, common_terms=stopwords, scoring="npmi")
        bigram_trans = Phraser(frasier)
        trigram = Phrases(bigram_trans[sins],
                          common_terms=stopwords,
                          delimiter=b' ', threshold=0,
                          scoring="npmi")
        trigram_trans = Phraser(trigram)
        model = Word2Vec(trigram_trans[sins])
        model.save('data/phrasesmod')
    return model

def nsim(m: str = "words", seed: str = "Sonnet", n: int = 20) -> list:
    """Return n similar words."""
    if m == "words":
        train = wmodel
    elif m == "phrases":
        train = pmodel
    else:
        train = kmodel
    return train.most_similar(seed, topn=n)

def get_wms(line1: str, line2: str) -> float:
    """Get similarity via Word Mover's Distance.
       Calculation from WmdSimilarity."""
    # Perhaps not a good measure with archaic, small corpus.
    return 1.0 / (1.0 + model.wmdistance(line1, line2))

wmodel = word_model(get_reader(TreebankWordTokenizer(), 'utf-8').sents())
pmodel = phrase_model(get_reader(TreebankWordTokenizer(), 'utf-8').sents())
kmodel = k_model(get_reader(TreebankWordTokenizer(), 'utf-8').sents())
