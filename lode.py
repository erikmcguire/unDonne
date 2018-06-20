#! /usr/bin/env python

import warnings # To suppress Windows "chunkize" aliasing alert:
warnings.filterwarnings(action='ignore',
                        category=UserWarning,
                        module='gensim')
import os, string, pickle, re
import nltk, gensim
from collections import *
from gensim.models import Word2Vec, Phrases, FastText
from gensim.models.phrases import Phraser
from nltk.corpus import cmudict, stopwords, PlaintextCorpusReader
from nltk.corpus import lin_thesaurus as thes
from nltk.tokenize import word_tokenize, sent_tokenize, TreebankWordTokenizer
# from nltk.tag import StanfordNERTagger, StanfordPOSTagger

stopwords = stopwords.words('english')
stopwords = list(filter(lambda x:
                            x not in ['not', 'no', "nor"],
                        stopwords))
prond = cmudict.dict()

def load_v(f: str = 'data/all_donne.txt') -> list:
    """Open main Donne corpus; filter and return types."""
    types, tokens, tokens_oc = set(), [], dict()
    with open(f) as poems:
        for line in nltk.sent_tokenize(poems.read()):
            for ix, word in enumerate(nltk.word_tokenize(line.strip())):
                if word.isalpha() and word not in string.punctuation:
                    # Rudimentary attempt at preserving arbitrary capitalization.
                    if line.split()[0] == word or word.isupper():
                        tokens_oc[word.lower()] = word.lower()
                    else:
                        tokens_oc[word.lower()] = word
                    tokens_oc["i"] = "I"
                    tokens.append(word.lower())
    types = set(tokens)
    return types, tokens, tokens_oc

def get_reader(tok, enc, fname = 'all_donne.txt'):
    """Return pair of PlaintextCorpusReaders."""
    all = PlaintextCorpusReader(r'data/', fname, word_tokenizer=tok, encoding=enc)
    return all

def load_sonnets(f: str  = 'data/titled_sonnets.txt') -> dict:
    """Create dictionary of Donne's sonnets."""
    with open(f) as sons:
        titles, sonnets = [], dict()
        sonnet = []
        for line in sons.readlines():
            if line[0] == '#':
                sonnet = []
                title = line[1:-1]
                titles.append(title)
            elif not line[0].isdigit():
                if len(sonnet) < 15:
                    sonnet.append(line)
                if len(sonnet) == 15:
                    sonnets[title] = ''.join(sonnet)
    return titles, sonnets

def parse_sonnet(ix: int = 12) -> list:
    """Collect sonnet's types, tokens, and sentences.
       Default is HSDeath."""
    sonnet = sonnets[titles[ix]]
    sents = sonnet.strip().split('\n')
    # "Send me some token, that my hope may live... "
    tokens = [w for sent in sents
                for w in nltk.word_tokenize(sent)
                if w not in string.punctuation.replace("'", "")]
    types = set(tokens)
    return [tokens, types, sents]

def get_sonnet_sents() -> list:
    """Get all sonnet sentences."""
    all_sents = []
    for i in range(len(titles)):
        sents = parse_sonnet(i)[2]
        all_sents.extend(sents)
    return all_sents

def sonnets_ends() -> dict:
    """Collect rhyme words based on rhyme scheme."""
    rd = defaultdict(list)
    for ix in range(len(sonnets)):
        sonnet = sonnets[titles[ix]]
        lines = sonnet.strip().split('\n')
        lines = [line.split() for line in lines]
        ends = [line[-1] if line[-1][-1] not in string.punctuation
                         else line[-1][:-1]
                         for line in lines]
        rd["a"].extend([(ends[0], ends[3]), (ends[4], ends[7])])
        rd["b"].extend([(ends[1], ends[2]), (ends[5], ends[6])])
        if ((ends[8][-1] == ends[11][-1]) or
            (ends[9][-1] == ends[10][-1])):
            rd["c"].extend([(ends[8], ends[11])])
            rd["d"].extend([(ends[9], ends[10])])
        else:
            rd["c"].extend([(ends[8], ends[10])])
            rd["d"].extend([(ends[9], ends[11])])
        rd["e"].extend([(ends[12], ends[13])])
    return rd

def sylc(w: str, d: dict = prond) -> int:
    """Approximate syllable count by counting numeric stress marks."""
    if not w in d.keys():
        return 0
    else:
        return min([len([pron for pron in prons
                              if pron[-1].isdigit()])
                              for prons in d[w]])

titles, sonnets = load_sonnets()
types, tokens, tokens_oc = load_v()
rhyme_words = sonnets_ends()
