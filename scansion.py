#! /usr/bin/env python

from sins import *
from nltk.stem.snowball import SnowballStemmer

def filter_sline(line: str) -> list:
    """Tokenize, downcase & remove punctuation from untokenized line."""
    return [word.lower() for word in nltk.word_tokenize(line)
                                  if word not in string.punctuation]

def filter_lline(line: list) -> list:
    """Downcase and remove punctuation from tokenized line."""
    return [word.lower() for word in line
                                  if word not in string.punctuation]

def line_sylc(line: list, prond: dict = prond) -> int:
    """Tally line's words' syllables."""
    toks = filter_lline(line)
    return sum([sylc(token) for token in toks
                                if token in prond.keys()])

def lines_sylc(lines: list) -> list:
    """Collect lines' syllable counts."""
    return [line_sylc(line) for line in lines]

def line_meter(line: list) -> str:
    """Encode meter. Returns 0, 2, ... 1, etc."""
    return ''.join([re.sub(r"[^0-9]", "",
                    ''.join(prond[token.lower()][0]))
            for token in line
            if token in prond.keys()])

def meter_match(line1: list, line2: list) -> bool:
    """Evaluate whether meter matches."""
    return line_meter(line1) == line_meter(line2)

def lines_ends(lines: list) -> str:
    """Encode lines' ending pronunciations."""
    res = []
    for line in lines:
        if line:
            line = filter_sline(line)
            end = line[-1]
            if end in prond.keys():
                if not prond[end][0][-1][-1].isdigit():
                    # Avoid unhelpful endings by requiring stresses.
                    end = ''.join(prond[end][0][-2:])
                else:
                    end = prond[end][0][-1]
                res.append(end)
            else:
                # Ensure scheme matches despite no pronunciation info.
                res.append('unk')
    return res

def rhyme_match(lines1: list, lines2: list) -> bool:
    """Evaluate whether lines' endings' schemes match."""
    return lines_ends(lines1) == lines_ends(lines2)

def lines_tags(lines: list) -> list:
    """Return lines' tag patterns."""
    return [' '.join([t for w, t in tagged_line])
                 for tagged_line in [nltk.pos_tag(filter_sline(line))
                                     for line in lines]]

def tag_match(lines1: list, lines2: list) -> bool:
    """Evaluate whether lines' tag patterns match."""
    l1 = [tag[0] for tag in [tags for tags in lines_tags(lines1)]]
    l2 = [tag[0] for tag in [tags for tags in lines_tags(lines2)]]
    return l1 == l2

def punk_match(line1: str, line2: str) -> bool:
    """Evaluate whether lines end punctuation matches."""
    assert(line1[-1] in string.punctuation)
    return line1[-1] and line1[-1] == line2[-1]

def max_line_len(lines: list) -> int:
    """Return longest line length in words."""
    return len(max(lines, key=lambda x: len(x.split())).split())

def max_chars(lines: list) -> int:
    """Return longest line length in chars."""
    return len(max(lines, key=lambda x: len(list(x))))

def min_chars(lines: list) -> int:
    """Return shortest line length in chars."""
    return len(min(lines, key=lambda x: len(list(x))))

def end_punk_scheme(lines: list) -> list:
    """Collect final punctuation per line."""
    punk = [line[-1] if line[-1] in string.punctuation
            else "" for line in lines]
    return punk

def get_ttr(lines: list) -> float:
    """Return type-token ratio."""
    lines = '\n'.join(lines)
    tokens = filter_lline(nltk.word_tokenize(lines))
    stemmer = SnowballStemmer("english")
    stems = [stemmer.stem(token) for token in tokens]
    types = set(stems)
    return len(types)/len(tokens)

def mimick(line: str, punk: str) -> str:
    """Capitalize line and add required punctuation."""
    return line[:1].upper() + line[1:] + punk

def line_check(line: list, hsl: list, ix: int, pol: float) -> bool:
    """Evaluate line polarity and meter."""
    line = ''.join(line).replace("<s>", "").split()
    return (line_sylc(line, prond) in range(8, 10) and
            get_syn(line=line) >= pol)
