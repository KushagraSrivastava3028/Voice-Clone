import re
import unicodedata
from .symbols import symbols

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1])
    for x in [
        ('mrs', 'misess'),
        ('mr', 'mister'),
        ('dr', 'doctor'),
        ('st', 'saint'),
        ('co', 'company'),
        ('jr', 'junior'),
        ('maj', 'major'),
        ('gen', 'general'),
        ('drs', 'doctors'),
        ('rev', 'reverend'),
        ('lt', 'lieutenant'),
        ('hon', 'honorable'),
        ('sgt', 'sergeant'),
        ('capt', 'captain'),
        ('esq', 'esquire'),
        ('ltd', 'limited'),
        ('col', 'colonel'),
        ('ft', 'fort'),
    ]
]

def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text

def expand_numbers(text):
    return normalize_numbers(text)

def lowercase(text):
    return text.lower()

def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)

def convert_to_ascii(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

def basic_cleaners(text):
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text

def english_cleaners(text):
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    text = expand_numbers(text)
    text = collapse_whitespace(text)
    return text

def text_to_sequence(text, cleaner_names):
    sequence = []
    clean_text = _clean_text(text, cleaner_names)
    for symbol in clean_text:
        if symbol in _symbol_to_id:
            symbol_id = _symbol_to_id[symbol]
            sequence.append(symbol_id)
        else:
            # Skip unknown symbols
            continue
    return sequence

def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception('Unknown cleaner: %s' % name)
        text = cleaner(text)
    return text
