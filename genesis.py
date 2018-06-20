#! /usr/bin/env python

from gramarye import *
import time

def generate_model(cfdist, word: str, hsl: list, ix: int, n: int = 30) -> str:
    """Generate constrained sentences from conditional distribution."""
    line = []
    for i in range(n):
        # "Walking here, Two shadowes went..."
        word = cfdist[word].generate()
        if line_sylc(line, prond) + sylc(word) == 10:
            # Simply matches 'ur-sonnet' rhymes.
            while not rhyme_match([word], [filter_sline(hsl[ix])[-1]]):
                word = cfdist[word].generate()
        if not line or word not in line[:-2]:
            line.append(tokens_oc[word])
        if line_sylc(line, prond) == 10 and len(line) <= max_line_len(hsl):
            if get_syn(line=line) >= 0.02:
                sline = ' '.join(line)
                return sline
        else:
            continue
    return 'xxx'

def gen_from_cfd(punk: list, cpd: dict, seed: str, hsl: list) -> str:
    """Use conditional probabilities and algorithmic constraints to produce sonnet."""
    genson = []
    for typ in types:
        if len(genson) == 0:
            typ = seed
        if len(genson) < 14:
            if len(genson) >= 1:
                prev = genson[-1].split()[-1]
                if prev in prond.keys():
                    typ = prev
            gend = generate_model(cpd, typ, hsl, len(genson))
            if gend != 'xxx':
                line = mimick(gend, punk[len(genson)]).replace(" i ", " I ")
                if line[-2:] == " i":
                    line = line[:-2]
                line = ''.join(filter(lambda x: x in string.printable, line))
                genson.append(line)
    return '\n'.join(genson)

def gensen(st: list, n: int = 20, d: int = 4, m: int = 13) -> list:
    """Induce stochastic CFG from sonnet 'treebank'.
       Use modified NLTK functions to probabilistically generate new sentences."""
    gensents = []
    sontag = [gram(sent) for sent in st]
    prodsp = [p for _, t in sontag
                       for p in t.productions()]
    grammarp = induce_pcfg(Nonterminal("S"), prodsp)
    for sentence in generated(grammarp, n=n, depth=d):
        if len(sentence) <= m:
            gensents.append(' '.join(sentence))
    return gensents

def gen_pcfg(p: list = None, l: int = 13, n: int = 20,
             d: int = 4, m: int = 13, to: int = 45) -> str:
    """Call PCFG-based generator and return sonnet."""
    songen = []
    ix = 0
    all_sents = get_sonnet_sents()
    start = time.time()
    while len(songen) < l:
        if int(time.time() - start) >= to:
            print("Timed out after ~{}s.\n Results:".format(to))
            break
        sents = gensen(all_sents, n, d, m)
        if sents and ix < l:
            sent = random.choice(sents)
            sent = sent.replace(" 's", "'s")
            songen.append(mimick(sent, p[ix]).replace(" i ", " I "))
            ix += 1
        else:
            print("No results.")
            break
    return '\n'.join(songen)
