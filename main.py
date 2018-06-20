#! /usr/bin/env python

from karkoav import *
from gui import *

words, words_set, sents = parse_sonnet()

# Instantiate bauplan, Holy Sonnet X: 'Death be not proud... '
hsdeath = sonnets[titles[12]]
hslines = list(filter(lambda x: x, hsdeath.split('\n')))
alpha = hslines[0].split()[0]
kpunk = end_punk_scheme(hslines)

seed = hslines[0].split()[0].lower()
cfd = nltk.ConditionalFreqDist(nltk.bigrams(tokens))
cpd = nltk.ConditionalProbDist(cfd, nltk.LaplaceProbDist,
                               bins=len(types))

if __name__ == '__main__':
    gg = GenGUI()
    gg.main()
