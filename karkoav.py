#!/usr/bin/env python

from genesis import *
from svgpathtools import *
from itertools import chain

class Karkoav:
    """Inspired by Andrej Karpathy & Yoav Goldberg."""
    def __init__(self, min: int, max: int, p: list,
                 hsl: list, k: int = 5, sub1: float = 0.06,
                 sub2: float = 0.03,
                 f: str = 'data/sonnets.txt',
                 theme: str = "",
                 thresh: int = 100,
                 vmod: str = "fasttext",
                 default: bool = True):
        """
        f: input txt path, name
        min: min chars in resulting line
        max: max chars in resulting line
        order: order-k history to condition on
        p: end punctuation scheme to mimic
        sub1: minimum subjectivity for octet (first 8 lines)
        sub2: minimum subjectivity for sestet (final 6 lines)
        hsl: lines of sonnet to mimic
        seeds: starting character k-grams
        lines: resulting string of lines
        theme: arbitrate initial search of thematic vector space
        thresh: number of word embeddings from theme(s)
        vmod: word2vec model: unigrams or higher order n-grams
        default: save/load dictionary created w/ default settings if true
        """
        with open(f, 'r') as txt:
            self._imp = txt.readlines()
        # "I am a little world made cunningly
        #  Of Elements, and an angelic sprite... "
        self._hsl = hsl
        self._min = min
        self._max = max
        self._theme = tuple(["<s>"] + list(theme))
        self._k = k
        self._p = p
        self._sub1 = sub1
        self._sub2 = sub2
        self._thresh = thresh
        self._model = defaultdict(list)
        self._seeds = []
        self._pairs = self._gen_pairs()
        self._title = ""
        self._vmod = vmod
        self._ttr = 0
        self._lines = "Nothing generated yet!"
        if default:
            try:
                self._model = pickle.load(open("data/karmod_{}.p".format(k), "rb"))
                self._seeds = list(self._model.keys())
            except:
                self._train_mod()
                pickle.dump(self._model, open("data/karmod_{}.p".format(k), "wb"))
        else:
            self._train_mod()
        if self._theme[:self._k] in self._seeds:
            self._prev_seed = self._theme[:self._k]
        else:
            # "I will vent that humour then... "
            self._prev_seed = None
        self._prev_seeds = [self._prev_seed]

    def _gen_pairs(self) -> list:
        """Generate list of rhyme pairs for a sonnet."""
        def get_pairs(l: str) -> Tuple:
            return random.choice(rhyme_words[l])
        a1 = get_pairs("a") # 0, 3
        a2 = get_pairs("a") # 4, 7
        while a2 == a1:
            a2 = get_pairs("a")
        b1 = get_pairs("b") # 1, 2
        b2 = get_pairs("b") # 5, 6
        while b2 == b1:
            b2 = get_pairs("b")
        c = get_pairs("c") # 8, 10 or 8, 11
        d = get_pairs("d") # 9, 11 or 9, 10
        e = get_pairs("e") # 12, 13
        type1 = [a1[0], b1[0], b1[1], a1[1],
                 a2[0], b2[0], b2[1], a2[1],
                 c[0], d[0], d[1], c[1],
                 e[0], e[1]]
        type2 = [a1[0], b1[0], b1[1], a1[1],
                 a2[0], b2[0], b2[1], a2[1],
                 c[0], d[0], c[1], d[1],
                 e[0], e[1]]
        return random.choice([[w.lower() for w in type1],
                              [w.lower() for w in type2]])

    def _train_mod(self) -> dict:
        """Create dictionary of letter k-grams;
           Values are tallies of letters that follow k chars.
        """
        k = self._k
        model = defaultdict(Counter)
        for line in self._imp:
            line = ''.join(filter(lambda x: x in string.printable, line))
            # Start state; preserving final EOL observations to shift probabilities, allow for constraints, such as rhyme check.
            lex = ["<s>"] + list(line)
            for i in range(len(lex)-k): # Update count for distribution.
                model[tuple(lex[i:i+k])][lex[i+k]] += 1
        self._prob_dist(model)

    def _prob_dist(self, model: dict = None) -> dict:
        """Normalize letter counts via Counter to create
           probability distribution for letter chains.
           Follows same logic as NLTK DictionaryProbDist class.
        """
        for lex, c in model.items():
            for l, p in c.items():
                # "Make your returne home gracious... "
                p *= 1.0/sum(c.values())
                self._model[lex].append((l, p))
        self._seeds = list(self._model.keys())

    def _gen_one(self, lex: tuple = None) -> str:
        """Follows same logic as NLTK generate() function."""
        lpd = self._model[lex]
        p = random.random()
        for l, prob in lpd:
            # "Good seed degenerates, and oft obeyes... "
            p -= prob
            if p <= 0:
                return l

    def _gen_sim(self, sims: list, cnt: int = 0) -> str:
        """Select suitable seed from similar words or phrases."""
        if cnt < len(sims):
            cnt += 1
            for w, score in sims:
                # Exhaust possibilities.
                _sims = list(filter(lambda t: w not in t, sims))
                return w if tuple(w) in self._seeds and tuple(w) not in self._prev_seeds else self._gen_sim(_sims, cnt)
        else:
            return None

    def _seed_sim(self, flag: bool = True) -> Tuple:
        """Attempts to choose a seed thematically similar to previous."""
        query = ''.join(self._prev_seed[flag:])
        try:
            ksims = set(nsim(m=self._vmod, seed=query, n=self._thresh))
            # Experimental, combining similar character and word embeddings.
            # fasttext + phrases || words + phrases
            if self._vmod == "fasttext":
                wq = "phrases"
            else: # vmod == "words" or "phrases"
                wq = "phrases" if self._vmod == "words" else "words"
            wsims = set(nsim(m=wq, seed=query, n=self._thresh))
            sims = list(ksims | wsims)
            if sims:
                temp_seed = tuple(self._gen_sim(sims)) or None
        except:
            return False, None
        # "No man is an Iland, intire of it selfe..."
        return (True, temp_seed) if temp_seed in self._seeds else (False, None)

    def _get_seed(self) -> Tuple:
        """Return starting state of a line."""
        if self._prev_seed:
            found_sim, temp_seed = self._seed_sim(flag="<s>" in self._prev_seed)
        else:
            found_sim, temp_seed = False, None
        seed = random.choice(self._seeds) if not found_sim else temp_seed
        self._prev_seed = seed
        self._prev_seeds.append(seed)
        # "She's all States, and all Princes, I,
        #  Nothing else is."
        return seed if "<s>" in seed or found_sim else self._get_seed()

    def _gen_line(self, ix: int = None, line: tuple = None,
                  st: float = 0., n: int = None) -> list:
        """Generate a line, character by character,
           of designated properties.
        """
        line = list(self._get_seed())
        if int(time.clock() - st) >= 30:
            print("Timed out after ~{}s.".format(30))
            return ''.join(line).replace("<s>", "")
        for i in range(self._max):
            lex = tuple(list(line[-self._k:]))
            l = self._gen_one(lex)
            if not l == "\n":
                line.append(l)
            else:
                if i < self._min:
                    # Conditional EOL emission to balance w/ manual constraints.
                    # Should still roughly reflect distributional bias.
                    continue
        # END OF LINE
        if not n: # Can match indices to template sonnet.
            sub = self._sub1 if ix < 7 else self._sub2
            lc = line_check(line, self._hsl, ix, sub)
            lw = filter_sline(self._hsl[ix])[-1]
        else:
            lc, rm = True, True
        if (self._min <= len(line) <= self._max and (lc and " " in line)):
            if line[-1] == " ":
                line = ''.join(line).replace("<s>", "") + self._pairs[ix]
            elif line[-1].isalpha():
                line = ''.join(line).replace("<s>", "") + " " + self._pairs[ix]
            elif line[-1] in string.punctuation:
                line = ''.join(line[:-1]).replace("<s>", "") + " " + self._pairs[ix]
            return line
        else:
            if not n:
                # Catching line pass not implemented just yet...
                return self._gen_line(ix=ix, line=tuple(line), st=st)
            else: # Generating user-specified lines, not matching sonnet.
                return self._gen_line(line=tuple(line), st=st, n=n)

    def _mimickry(self, line: str, ix: int = None,
                  n: int = None) -> str:
        """Capitalize start and mimic template punctuation scheme.
           Always ends with newline character.
        """
        # ne = StanfordNERTagger('data/english.conll.4class.distsim.crf.ser.gz',
                                # encoding='utf-8')
        # line = ' '.join(w[:1].upper() + w[1:]
                        # if t[1] != 'O' else w
                        # for w, t in ne.tag_sents([line.split()]))
        if n:
            line = line[:1].upper() + line[1:] + "\n"
            return re.sub(r"(.* )i(\W)", r"\g<1>I\g<2>", line)
        ll = list(filter(lambda l:
                            l in string.punctuation,
                        line[-2:]))
        if ll:
            i = line.index(ll[0])
            line = "{}{}{}\n".format(line[:1].upper() + line[1:i],
                                     self._p[ix], line[i+1:])
        else:
            line = line[:1].upper() + line[1:] + self._p[ix] + "\n"
        # "'Tis but applying worme-seed to the Taile."
        line = re.sub(r"(.* )i(\W)", r"\g<1>I\g<2>", line)
        return re.sub(r"(.* )i(\W)", r"\g<1>I\g<2>", line)

    def _gen_lines(self) -> list:
        """Slide over text to create order-k model,
           generate n lines of char length m.
        """
        st = time.clock()
        # "All honour's mimick; All wealth alchemy."
        lines = [self._mimickry(self._gen_line(ix=i, st=st), ix=i)
                                for i, l in enumerate(self._hsl)]
        self.ttr = get_ttr(lines)
        self._lines = ''.join(lines)

    def _contrib_mod(self, lines: str) -> Iterable:
        """Stores counts of sonnets' word contributions to each line."""
        lines = lines.split('\n')
        dd = defaultdict(Counter)
        for j, l in enumerate(lines):
            for i in sonnets:
                for w in nltk.word_tokenize(l):
                    if w.lower() in sonnets[i].lower():
                        dd[j][i] += 1
        return dd

    def _set_title(self, mc_sonnet: str) -> None:
        """Convoluted function to obtain title from synonyms
           of the title of the sonnet contributing the most
           to the first line."""
        if " " in mc_sonnet: # "Holy Sonnet (X)" won't work well...
            try:
                mc_title = random.choice(list(filter(lambda x: len(x) > 4,  sonnets[mc_sonnet].split())))
                mc_title = re.sub(r"\W", "", mc_title)
            except: # Most likely filtered out all words somehow...
                mc_title = mc_sonnet
        else: # Remove punctuation.
            mc_title = re.sub(r"\W", "", mc_sonnet)
        try: # "s" argument pulls synset name.
            temp_title = get_syn([mc_title], s=True)
            temp_title = temp_title[:temp_title.find(".")]
            self._title = temp_title.upper()
        except:
            try:
                mc_title = random.choice(list(thes.synonyms(ngram=mc_title)[-1][1]))
                self._title = mc_title.upper()
            except:
                self._title = mc_title.upper() + " (Version)"

    def get_most_contrib(self) -> Tuple:
        """Returns the sonnet indices which contributed the most per line:
           Returns {gen_line_index1: [(source_word_index1, source_line_index1), ...]}"""
        dd = self._contrib_mod(self._lines)
        indices = defaultdict(list)
        words = []
        cnt = 0
        f = False
        ctitles = (dd[i].most_common(1)[0][0]
                   for i, _ in enumerate(list(dd.keys())))
        stitles = set([dd[i].most_common(1)[0][0]
                       for i, _ in enumerate(list(dd.keys()))])
        colors = ['orchid', 'indigo', 'teal', 'seagreen', 'cyan',
                  'darkorange', 'crimson', 'coral', 'dimgray',
                  'indianred', 'gold', 'sienna', 'magenta', 'turquoise']
        tcd = {title:random.choice(colors)
               for title in stitles}
        while f == False:
            for i, gen_line in enumerate(self._lines.split('\n')):
                try:
                    title = next(ctitles)
                    sonnet = sonnets[title]
                except:
                    sonnet = sonnets[title]
                    cnt += 1
                    if cnt >= len(dd):
                        f = True
                    pass
                for j, source_line in enumerate(sonnet.split('\n')[1:-1]):
                    for k, word in enumerate(source_line.split()):
                        if word in gen_line.split() and word not in words:
                            words.append(word)
                            indices[i] = [title, tcd[title], (k, j)]
                            f = True
                            break
                        else:
                            f = False
                    if f:
                        f = False
                        break
                if len(indices) == 14:
                    f = True
        return indices, self._title, self._lines.split('\n')[:-1]

    def _make_paths(self, dd, lines):
        mpath = (lambda i, entry:
                    parse_path("M0 {} {} {} l.5,-.15 z".format(i, *entry)))
        # dd[gen_line_ix]: [source_title, color, (source_word_ix, source_line_ix)]
        paths = [mpath(i, dd[i][2]) for i in range(len(dd.keys()))]
        colors = [dd[i][1] for i in range(len(dd.keys()))]
        return colors, paths, lines

    def gen_svg(self, dd, title, lines):
        get_expath = (lambda t:
                        os.path.join(os.getcwd(), 'output',
                                     t + '.svg'))
        colors, paths, lines = self._make_paths(dd, lines)
        nodes = list(chain.from_iterable(((p.point(0.0), p.point(1.0))
                     for p in paths)))
        nodec = list(chain.from_iterable((('plum', 'thistle')
                        for _ in paths)))
        noder = list(chain.from_iterable(((.1, .1)
                        for _ in paths)))
        disvg(paths=paths, timestamp=True, colors=colors,
              nodes=nodes, node_radii=noder, node_colors=nodec,
              text_path=[Line(start=path.point(0.0),
                              end=(path.point(0.0) + len(line)))
                         for i, (line, path) in enumerate(zip(lines, paths))],
              text=lines, font_size=.4, svg_attributes={"viewBox": "-1 -1 50 25"},
              openinbrowser=False,
              **{'filename': get_expath(title)})

    def generate(self, n: int = None) -> str:
        """Generate and return resulting lines."""
        # "Goe, and catche a falling starre..."
        if n:
            st = time.clock()
            return ''.join([self._mimickry(
                           self._gen_line(st=st, n=n), n=n)
                           for i in range(n)]).replace(" i ", " I ")
        else:
            self._gen_lines()
            self._lines = self._lines.replace(" i ", " I ")
            dd = self._contrib_mod(self._lines)
            mc_sonnet = dd[0].most_common(1)[0][0]
            self._set_title(mc_sonnet)
            return self._lines, self._title
