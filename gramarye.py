#! /usr/bin/env python

import sys, itertools, random
from scansion import *
from typing import Tuple, Iterator
from nltk import tree, Tree, PCFG, induce_pcfg, Nonterminal

# For Stanford taggers...
# os.environ["JAVAHOME"] = 'C:\\Program Files\\Java\\jre1.8.0_101\\bin'
# os.environ["CLASSPATH"] = 'data'

def tag_list(l: list) -> list:
    """Filter and tag wordlist."""
    l = [word for word in l if word not in string.punctuation]
    tagged_l = nltk.pos_tag(l)
    return tagged_l

def generated(grammar, start=None, depth=None, n=None) -> Iterator[list]:
    """
    Generates an iterator of all sentences from a PCFG.
    Modified from: https://www.nltk.org/_modules/nltk/parse/generate.html
    """
    if not start:
        start = grammar.start()
    if not depth:
        depth = sys.maxsize
    iter = generate_all(grammar, [start], depth)
    if n:
        iter = itertools.islice(iter, n)
    return iter

def generate_all(grammar: 'nltk.grammar.PCFG', items: list, depth: int) -> list:
    """Modified from: https://www.nltk.org/_modules/nltk/parse/generate.html"""
    if items:
        try:
            for frag1 in generate_one(grammar, items[0], depth):
                for frag2 in generate_all(grammar, items[1:], depth):
                    yield frag1 + frag2
        except RuntimeError as _error:
            if _error.message == "maximum recursion depth exceeded":
                # Helpful error message while still showing the recursion stack.
                raise RuntimeError("The grammar has rule(s) that yield infinite recursion!!")
            else:
                raise
    else:
        yield []

def generate_one(grammar: 'nltk.grammar.PCFG', item: list, depth: int) -> list:
    """Modified from: https://www.nltk.org/_modules/nltk/parse/generate.html"""
    if depth > 0:
        if isinstance(item, Nonterminal):
            p = random.random()
            for prod in grammar.productions(lhs=item):
                # nltk.org/_modules/nltk/probability.html#ProbDistI.generate
                p -= prod.prob()
                # "If that be simply perfectest
                #  Which can by no way be exprest
                #  But Negatives, my love is so."
                if p <= 0:
                    for frag in generate_all(grammar, prod.rhs(), depth-1):
                        yield frag
            if p < .0001:
                # "... arise, arise
                #  From death, you numberlesse infinities... "
                for frag in generate_all(grammar, prod.rhs(), depth-1):
                    yield frag
        else:
            yield [item]

def gram(gend: str) -> Tuple:
    """Use (ideally) stylistic grammar to create treebank."""
    grammar = r"""  NOM: {<JJ>+ <NN.*>} # "poor death..."
                         {<DT> <NN.*>} # "... a little world... "
                         {<CD> <NN.*>}
                         {<PR.*> <NN.*>}
                         {<RB.*>+ <NN.*>}
                         {<NN.*>+}
                     NP: {<NOM> <CC> <NOM>} # "Mighty and dreadful"
                         {<NOM> <PP>}
                         {<NOM>}
                     PP: {<IN> <NP>}
                     VP: {<PR.*> <VB.*>} # "i am... "
                         {<MD> <VB.*>}
                         {<EX> <VB.*>}
                         {<WDT> <VB.*>}
                         {<VB.*> <RB.*>} # "... made cunningly... "
                         {<RB.*> <VB.*>}
                         {<VB.*> <PP>}
                         {<VB.*>+}
                      S: {^<NP> <VP> [\n$]} # "... a little world made cunningly..." - Anchors seem to help speed.
                         {^<VP> <NP> [\n$]} # "batter my heart... "
                """
    cp = nltk.RegexpParser(grammar)
    gend = [word for word in nltk.word_tokenize(gend)
                 if word not in string.punctuation]
    # st_pos = StanfordPOSTagger('data/english-bidirectional-distsim.tagger', encoding='utf-8')
    gendt = nltk.pos_tag(gend)
    pt = cp.parse(gendt)
    # Awkward conversion to S-expression style analogous to Penn treebank.
    st = re.sub(r"(\b[A-Z\$a-z]+)\/([A-Z\$]+)", "(\\2 \\1)", str(pt))
    t = Tree.fromstring(st)
    t.collapse_unary()
    return gendt, t
