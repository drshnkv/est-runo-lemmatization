"""Recompute the article's reported figures from the files in data/ and
compare each with its printed value.

Usage, from the repository root: python analysis/recompute_results.py

Prints each figure's place (article or configuration record), description,
recomputed and printed values and agreement, and writes the same to
recomputed_results.csv in the working directory. Exit status 1 if any
figure differs or lacks a value, else 0.
"""
import csv
import difflib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

# Data location and resampling settings: one seed and the replicate counts.
DATA = Path(__file__).resolve().parent.parent / 'data'

BOOT_SEED = 20261019
N_BOOT = 10000
N_BOOT_MAIN = 400000
N_FLIP = 100000

# The seven model-selection songs, left out in the sensitivity analysis.
MODEL_SELECTION_SONGS = {
    'ars0104990010', 'era219001990005', 'era203601950002', 'era210102140008',
    'estk1501170082', 'era203203650010', 'era216102270022',
}


# The matching rule behind every agreement figure, the translation analysis
# included: remove the compound-boundary marker + and the diminutive endings
# -kene and -ke, ignore case, and accept any of the comma-separated reference
# lemmas. casefold=False serves only the case-sensitive word-level tests in
# section 7 of the configuration record.
def normalize_lemma(lemma: str) -> str:
    lemma = (lemma or '').strip().replace('+', '')
    if lemma.endswith('kene'):
        lemma = lemma[:-4]
    elif lemma.endswith('ke'):
        lemma = lemma[:-2]
    return lemma


def lemmas_match(pred: str, ref: str, casefold: bool) -> bool:
    if not (pred or '').strip() or not (ref or '').strip():
        return False
    p = pred.strip()
    refs = [g.strip() for g in ref.split(',')]
    if casefold:
        p = p.lower()
        refs = [g.lower() for g in refs]
    if any(p == g for g in refs):
        return True
    pn = normalize_lemma(p)
    for g in refs:
        gn = normalize_lemma(g)
        if pn == gn or pn == g or p == gn:
            return True
    return False


# Statistical routines: exact McNemar test, song-level bootstrap and
# sign-flip draws, percentile interval, sign-flip p-value, Holm adjustment.
def exact_mcnemar(b: int, c: int) -> float:
    if b + c == 0:
        return 1.0
    return binomtest(b, b + c, 0.5).pvalue


def draw_boot_matrix(n_songs: int, seed: int = BOOT_SEED,
                     n_boot: int = N_BOOT) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_songs, size=(n_boot, n_songs))


def draw_flip_matrix(n_songs: int, seed: int = BOOT_SEED,
                     n_flip: int = N_FLIP) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(n_flip, n_songs)) * 2 - 1


def boot_ratio(k_s: np.ndarray, n_s: np.ndarray,
               boot: np.ndarray) -> np.ndarray:
    return k_s[boot].sum(axis=1) / n_s[boot].sum(axis=1)


def percentile_ci(vals: np.ndarray):
    return tuple(np.percentile(vals, [2.5, 97.5]))


def sign_flip_p(d_s: np.ndarray, flips: np.ndarray) -> float:
    T = d_s.sum()
    T_star = flips @ d_s
    return (1 + int(np.sum(np.abs(T_star) >= abs(T)))) / (len(flips) + 1)


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, prev = {}, 0.0
    for i, (name, p) in enumerate(items):
        val = min(1.0, max(prev, (m - i) * p))
        adj[name] = val
        prev = val
    return adj


# Read the two data files, remove compound-boundary markers from the
# neural lemmatizer columns, and check that the files align row by row.
def marker_normalize(value: str) -> str:
    return value.replace('_', '').replace('=', '')


def read_table(name):
    with open(DATA / name, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


EVAL = read_table('evaluation_results.csv')
PRED = read_table('per_word_predictions.csv')
N_ROWS = len(EVAL)
assert N_ROWS == len(PRED) == 3216
for i, (e, p) in enumerate(zip(EVAL, PRED)):
    assert int(p['row']) == i
    assert p['poem_id'] == e['poem_id'] and p['original_word'] == e['original_word']

MANUAL = 'manual_lemma_standard'
EXPERT_REF = 'secondary_reference'
MARKER_COLUMNS = {'lexicon_lemmatizer_dialect', 'lexicon_lemmatizer_vabamorf',
                  'glilem', 'stanza'}


def column(name):
    table = EVAL if name in EVAL[0] else PRED
    values = [r[name] for r in table]
    if name in MARKER_COLUMNS:
        values = [marker_normalize(v) for v in values]
    return values


def correct(name, reference=MANUAL, casefold=True):
    return np.array([lemmas_match(p, g, casefold)
                     for p, g in zip(column(name), column(reference))])


# Songs are the resampling unit: word counts per song, shared draws, helpers.
SONGS = sorted({r['poem_id'] for r in EVAL})
assert len(SONGS) == 45
SONG_OF = {s: j for j, s in enumerate(SONGS)}
SONG = np.array([SONG_OF[r['poem_id']] for r in EVAL])
ALL = np.ones(N_ROWS, dtype=bool)
EXPERT = np.array([bool(r[EXPERT_REF].strip()) for r in EVAL])


def per_song(mask, song=SONG, n_songs=45):
    return np.bincount(song[mask], minlength=n_songs).astype(float)


N_ALL = per_song(ALL)
N_EXPERT = per_song(EXPERT)
BOOT = draw_boot_matrix(45, BOOT_SEED, N_BOOT_MAIN)
FLIPS = draw_flip_matrix(45, BOOT_SEED, N_FLIP)


def pct(k, n):
    return 100.0 * k / n


def accuracy(hit, universe=ALL):
    return pct(int(hit[universe].sum()), int(universe.sum()))


def interval(hit, universe=ALL):
    lo, hi = percentile_ci(boot_ratio(per_song(hit & universe),
                                      per_song(universe), BOOT))
    return 100 * lo, 100 * hi


def contrast(hit_a, hit_b, universe=ALL):
    d = per_song(hit_a & universe) - per_song(hit_b & universe)
    n = per_song(universe)
    lo, hi = percentile_ci(boot_ratio(d, n, BOOT))
    return {'points': 100 * d.sum() / n.sum(), 'tokens': d.sum(),
            'low': 100 * lo, 'high': 100 * hi, 'p': sign_flip_p(d, FLIPS),
            'songs_a': int((d > 0).sum()), 'songs_b': int((d < 0).sum()),
            'songs_tied': int((d == 0).sum())}


# V collects every recomputed quantity under a stable key.
V = {}


def put(prefix, values):
    for k, v in values.items():
        V[f'{prefix}.{k}'] = v


# The twelve systems' columns, and the subsets for translation similarity.
TABLE1 = {
    'lexical': 'lexical_levenshtein_lemma',
    'normalization': 'normalized_estnltk',
    'lexicon_dialect': 'lexicon_lemmatizer_dialect',
    'lexicon_vabamorf': 'lexicon_lemmatizer_vabamorf',
    'glilem': 'glilem',
    'stanza': 'stanza',
    'pure_estnltk': 'pure_estnltk',
    'estnltk_dictionary': 'estnltk_lemma',
    'deepseek_base': 'deepseek_standard_lemma_modern',
    'deepseek_enhanced': 'deepseek_enhanced_lemma_modern',
    'claude_base': 'claude_standard_lemma_modern',
    'claude_enhanced': 'claude_enhanced_lemma_modern',
}
NEURAL = ['normalization', 'lexicon_dialect', 'lexicon_vabamorf', 'glilem', 'stanza']
TRANSLATION = {
    'deepseek_base': 'deepseek_standard_english_translation',
    'deepseek_enhanced': 'deepseek_enhanced_english_translation',
    'claude_base': 'claude_standard_english_translation',
    'claude_enhanced': 'claude_translation_enhanced',
}

HIT = {s: correct(c) for s, c in TABLE1.items()}
HIT_EXPERT_CF = {s: correct(c, EXPERT_REF, True) for s, c in TABLE1.items()}
HIT_EXPERT_CS = {s: correct(c, EXPERT_REF, False) for s, c in TABLE1.items()}
HIT_CS = {s: correct(c, MANUAL, False) for s, c in TABLE1.items()}

# Corpus counts, and regional and subcategory shares.
V['corpus.rows'] = N_ROWS
V['corpus.songs'] = len(SONGS)
V['corpus.collection_areas'] = len({r['place_name'] for r in EVAL})
V['corpus.expert_rows'] = int(EXPERT.sum())
V['corpus.expert_rows_pct'] = pct(int(EXPERT.sum()), N_ROWS)
V['corpus.excluded_rows'] = int((~EXPERT).sum())
places = Counter(r['place_name'] for r in EVAL)
for place in ['Setumaa', 'Tõstamaa', 'Jämaja', 'Märjamaa']:
    V[f'corpus.area_entries.{place}'] = places[place]
V['corpus.setomaa_most_represented'] = places.most_common(1)[0][0] == 'Setumaa'
REGION = np.array([r['dialect_primary_merged'] for r in EVAL])
SUBCATEGORY = np.array([r['dialect_subcategory_merged'] for r in EVAL])
REGIONS = ['South Estonian', 'Western Estonian', 'North+East Estonian']
for region in REGIONS:
    V[f'corpus.region_entries.{region}'] = int((REGION == region).sum())
    V[f'corpus.region_songs.{region}'] = len({r['poem_id'] for r in EVAL
                                              if r['dialect_primary_merged'] == region})
V['corpus.south_share_pct'] = pct(V['corpus.region_entries.South Estonian'], N_ROWS)
V['corpus.seto_entries'] = int((SUBCATEGORY == 'Seto').sum())
V['corpus.seto_songs'] = len({r['poem_id'] for r in EVAL
                              if r['dialect_subcategory_merged'] == 'Seto'})
V['corpus.seto_share_pct'] = pct(V['corpus.seto_entries'], N_ROWS)
# Dictionary coverage and single- versus multi-lemma reference entries.
lexical_filled = np.array([bool(v.strip()) for v in column('lexical_levenshtein_lemma')])
V['coverage.lexical_empty'] = int((~lexical_filled).sum())
V['coverage.lexical_all_rows'] = int(lexical_filled.sum())
V['coverage.lexical_all_pct'] = pct(int(lexical_filled.sum()), N_ROWS)
multi = np.array([',' in r[EXPERT_REF] for r in EVAL]) & EXPERT
V['reference.single_lemma'] = int((EXPERT & ~multi).sum())
V['reference.single_lemma_pct'] = pct(V['reference.single_lemma'], int(EXPERT.sum()))
V['reference.multi_lemma'] = int(multi.sum())
V['reference.multi_lemma_pct'] = pct(V['reference.multi_lemma'], int(EXPERT.sum()))

# Agreement with both reference sets, and type-level agreement.
for s in TABLE1:
    V[f'manual.{s}'] = accuracy(HIT[s])
    V[f'manual_low.{s}'], V[f'manual_high.{s}'] = interval(HIT[s])
    V[f'expert.{s}'] = accuracy(HIT_EXPERT_CF[s], EXPERT)
for s in TABLE1:
    V[f'gap.{s}'] = V[f'manual.{s}'] - V['manual.claude_enhanced']
V['gap.claude_both_above_estnltk_dictionary_24'] = (
    min(V['manual.claude_base'], V['manual.claude_enhanced']) - V['manual.estnltk_dictionary'] > 24)
V['manual.lexical_frequency_ranked'] = accuracy(correct('lexical_frequency_ranked'))

# Type-level agreement: each word form counts once, as correct only if
# every occurrence agrees; the Claude Opus 5 rows average the three
# original-order runs.
types = defaultdict(list)
for i, r in enumerate(EVAL):
    w = r['original_word'].strip()
    if w:
        types[w].append(i)
V['types.count'] = len(types)


def types_correct(hit):
    return sum(1 for ii in types.values() if all(hit[j] for j in ii))


for s in TABLE1:
    V[f'types.{s}'] = pct(types_correct(HIT[s]), len(types))
for s, stem in [('opus_base', 'base_original_order'),
                ('opus_enhanced', 'enhanced_original_order')]:
    V[f'types.{s}'] = np.mean([pct(types_correct(correct(f'{stem}_run{r}')), len(types))
                               for r in (1, 2, 3)])
    V[f'expert.{s}'] = np.mean([accuracy(correct(f'{stem}_run{r}', EXPERT_REF), EXPERT)
                                for r in (1, 2, 3)])
V['types.claude_gain'] = V['types.claude_enhanced'] - V['types.claude_base']
V['types.claude_gain_forms'] = (types_correct(HIT['claude_enhanced'])
                                - types_correct(HIT['claude_base']))


# Translation similarity: difflib's SequenceMatcher ratio, asymmetric, so
# the manual translation always comes first. Entries at 0.7 or above form
# the high group; the Enhanced AI comparison is repeated at every cut-off
# from 0.5 to 0.9 where either system's grouping changes.
def clean(t):
    return re.sub('[.,;:!?()"\'-]', '', re.sub(r'\s+', ' ', str(t or '').strip().lower()))


def similarity(a, b):
    if a and b:
        if clean(a) == clean(b):
            return 1.0
        return difflib.SequenceMatcher(None, clean(a), clean(b)).ratio()
    return float(a == b)


manual_translation = column('manual_english_translation')
SIMILARITY = {}
for s, tcol in TRANSLATION.items():
    sim = np.array([similarity(m, a) for m, a in zip(manual_translation, column(tcol))])
    SIMILARITY[s] = sim
    high = sim >= 0.7
    g = HIT[s]
    V[f'translation.similarity.{s}'] = 100 * sim.mean()
    V[f'translation.high_share.{s}'] = 100 * high.mean()
    V[f'translation.agreement_high.{s}'] = 100 * g[high].mean()
    V[f'translation.agreement_low.{s}'] = 100 * g[~high].mean()
    V[f'translation.association.{s}'] = 100 * (g[high].mean() - g[~high].mean())
V['translation.claude_association_smaller'] = (
    V['translation.association.claude_enhanced'] < V['translation.association.deepseek_enhanced'])


def association_at(s, cut):
    high = SIMILARITY[s] >= cut
    return 100 * (HIT[s][high].mean() - HIT[s][~high].mean())


cutoffs = sorted({0.5, 0.9} | {float(v) for s in ('claude_enhanced', 'deepseek_enhanced')
                               for v in SIMILARITY[s] if 0.5 <= v <= 0.9})
V['translation.claude_association_smaller_all_cutoffs'] = all(
    association_at('claude_enhanced', c) < association_at('deepseek_enhanced', c) for c in cutoffs)

for column_name, label in [('claude_enhanced_lemma_modern', 'claude_enhanced'),
                           ('claude_standard_lemma_modern', 'claude_base'),
                           (MANUAL, 'manual'), (EXPERT_REF, 'expert')]:
    forms = defaultdict(list)
    for r in EVAL:
        w, v = r['original_word'].strip(), r[column_name].strip()
        if w and v:
            forms[w].append(v)
    V[f'variation.{label}'] = sum(1 for v in forms.values()
                                  if len(v) > 1 and len(set(v)) > 1)

# Original evaluation: differences with bootstrap intervals, sign-flip
# tests with Holm adjustment, and McNemar tests with and without case.
V['original.manual_gain_over_pure'] = V['manual.claude_enhanced'] - V['manual.pure_estnltk']
V['original.expert_gain_over_pure'] = V['expert.claude_enhanced'] - V['expert.pure_estnltk']
V['original.best_other_non_llm'] = max(V['manual.lexical_frequency_ranked'],
                                       *[V[f'manual.{s}'] for s in NEURAL])
put('original.claude_gain', contrast(HIT['claude_enhanced'], HIT['claude_base']))
put('original.deepseek_gain', contrast(HIT['deepseek_enhanced'], HIT['deepseek_base']))
did = ((per_song(HIT['deepseek_enhanced']) - per_song(HIT['deepseek_base']))
       - (per_song(HIT['claude_enhanced']) - per_song(HIT['claude_base'])))
lo, hi = percentile_ci(boot_ratio(did, N_ALL, BOOT))
put('original.gain_difference', {'points': 100 * did.sum() / N_ALL.sum(),
                                 'low': 100 * lo, 'high': 100 * hi,
                                 'p': sign_flip_p(did, FLIPS)})
adjusted = holm({'claude_gain': V['original.claude_gain.p'],
                 'deepseek_gain': V['original.deepseek_gain.p'],
                 'gain_difference': V['original.gain_difference.p']})
for name, p in adjusted.items():
    V[f'original.{name}.p_adjusted'] = p
V['original.deepseek_gap_to_claude'] = V['manual.claude_enhanced'] - V['manual.deepseek_enhanced']

for rule, hits in [('cf', (HIT, HIT_EXPERT_CF)), ('cs', (HIT_CS, HIT_EXPERT_CS))]:
    for universe_name, hit_map, universe in [('manual', hits[0], ALL),
                                             ('expert', hits[1], EXPERT)]:
        e = hit_map['claude_enhanced'][universe]
        b = hit_map['claude_base'][universe]
        only_e, only_b = int((e & ~b).sum()), int((~e & b).sum())
        V[f'mcnemar.{rule}.{universe_name}.enhanced_only'] = only_e
        V[f'mcnemar.{rule}.{universe_name}.base_only'] = only_b
        V[f'mcnemar.{rule}.{universe_name}.p'] = exact_mcnemar(only_e, only_b)


# Expert review: parse the expert's decision codes and count their classes
# (Table 3).
def parse_codes(raw):
    toks = [t.strip() for t in raw.strip().split(',') if t.strip()]
    return frozenset('LE' if t == 'EL' else t for t in toks)


def code_class(codes):
    if not codes:
        return 'empty'
    if codes == {'M', 'LE', 'LS'}:
        return 'all_matched'
    if codes == {'M'}:
        return 'manual_only'
    if codes == {'0'}:
        return 'none_matched'
    if codes in ({'M', 'LE'}, {'M', 'LS'}):
        return 'mixed'
    if 'M' not in codes and codes <= {'LE', 'LS'}:
        return 'ai_without_manual'
    if codes <= {'-', '?'}:
        return 'dash_or_question'
    return 'other'


codes = [parse_codes(r['expert_comparison']) for r in EVAL]
classes = Counter(code_class(c) for c in codes)
reviewed = sum(v for k, v in classes.items() if k not in ('empty', 'dash_or_question'))
V['review.cases'] = reviewed
V['review.other_codes'] = classes['other']
for name in ['ai_without_manual', 'manual_only', 'all_matched', 'none_matched', 'mixed']:
    V[f'review.{name}'] = classes[name]
    V[f'review.{name}_pct'] = pct(classes[name], reviewed)
V['review.enhanced_without_manual'] = sum(1 for c in codes if 'LE' in c and 'M' not in c)
V['review.enhanced_without_manual_pct'] = pct(V['review.enhanced_without_manual'], reviewed)
V['review.five_categories_sum'] = sum(classes[k] for k in
                                      ['ai_without_manual', 'manual_only', 'all_matched',
                                       'none_matched', 'mixed'])
V['review.manual_included_pct'] = pct(classes['manual_only'] + classes['all_matched']
                                      + classes['mixed'], reviewed)

# Switching to the secondary reference set: words each Claude system
# matches under one reference only, the effects of the exclusions and of
# the expert's decisions on Enhanced AI, and whether other systems gain less.
for s in ['claude_enhanced', 'claude_base']:
    manual_hit, expert_hit = HIT[s], HIT_EXPERT_CF[s]
    V[f'bridge.flips_up.{s}'] = int((EXPERT & expert_hit & ~manual_hit).sum())
    V[f'bridge.flips_down.{s}'] = int((EXPERT & manual_hit & ~expert_hit).sum())
V['bridge.exclusion_effect'] = (accuracy(HIT['claude_enhanced'], EXPERT)
                                - accuracy(HIT['claude_enhanced']))
V['bridge.adjudication_effect'] = (accuracy(HIT_EXPERT_CF['claude_enhanced'], EXPERT)
                                   - accuracy(HIT['claude_enhanced'], EXPERT))
claude_margin = min(V[f'expert.{s}'] - V[f'manual.{s}'] for s in ['claude_enhanced', 'claude_base'])
V['bridge.outside_systems_gain_less'] = all(
    V[f'expert.{s}'] - V[f'manual.{s}'] < claude_margin
    for s in TABLE1 if s not in ('claude_enhanced', 'claude_base'))

# Neural systems, the normalization controls, and their comparisons.
NORMALIZATION_ARMS = {'identity': 'unnormalized_estnltk', 'lowercased': 'lowercased_estnltk',
                      'repaired': 'normalized_repaired_estnltk'}
ARM = {k: correct(c) for k, c in NORMALIZATION_ARMS.items()}
for k in ARM:
    V[f'neural.{k}'] = accuracy(ARM[k])
put('neural.normalization_vs_identity', contrast(HIT['normalization'], ARM['identity']))
V['neural.normalization_below_identity'] = -V['neural.normalization_vs_identity.points']
put('neural.repaired_vs_lowercased', contrast(ARM['repaired'], ARM['lowercased']))
put('neural.lexicon_vs_estnltk_dictionary',
    contrast(HIT['lexicon_dialect'], HIT['estnltk_dictionary']))
put('neural.glilem_vs_estnltk_dictionary', contrast(HIT['glilem'], HIT['estnltk_dictionary']))
V['neural.glilem_below_estnltk_dictionary'] = -V['neural.glilem_vs_estnltk_dictionary.points']
put('neural.stanza_vs_pure', contrast(HIT['stanza'], HIT['pure_estnltk']))
V['neural.lowest'] = min(V[f'manual.{s}'] for s in NEURAL)
V['neural.highest'] = max(V[f'manual.{s}'] for s in NEURAL)
V['neural.normalization_lowest'] = V['manual.normalization'] == V['neural.lowest']
V['neural.all_below_estnltk_dictionary'] = all(
    V[f'manual.{s}'] < V['manual.estnltk_dictionary'] for s in NEURAL)

# Replication with Claude Opus 5: condition means, run-to-run spread,
# intervals, the main and seven Holm-adjusted comparisons, processing orders.
REPLICATION = {
    'enhanced': 'enhanced_original_order', 'base': 'base_original_order',
    'order2': 'enhanced_order2', 'order3': 'enhanced_order3', 'order4': 'enhanced_order4',
    'frozen': 'enhanced_frozen_store', 'no_dictionary': 'enhanced_no_dictionary',
    'no_estnltk': 'enhanced_no_estnltk', 'no_word_context': 'enhanced_no_word_context',
    'no_corpus_store': 'enhanced_no_corpus_store',
    'september_base': 'september_prompts_base',
    'september_enhanced': 'september_prompts_enhanced',
}
RUN_HITS = {c: [correct(f'{stem}_run{r}') for r in (1, 2, 3)]
            for c, stem in REPLICATION.items()}


def run_counts(condition, universe=ALL):
    return np.array([per_song(h & universe) for h in RUN_HITS[condition]])


def replication_cell(condition):
    k = run_counts(condition)
    per_run = k.sum(axis=1) / N_ALL.sum() * 100
    k_mean = k.mean(axis=0)
    lo, hi = np.percentile(k_mean[BOOT].sum(axis=1) / N_ALL[BOOT].sum(axis=1) * 100,
                           [2.5, 97.5])
    return {'mean': per_run.mean(), 'sd': per_run.std(ddof=1), 'low': lo, 'high': hi}


# Difference in points and in mean correct words, 95 per cent bootstrap
# interval, and sign-flip p-value (on summed counts, for exact ties).
def replication_contrast(a, b, universe=ALL, n_s=N_ALL, boot=BOOT, flips=FLIPS,
                         songs=slice(None)):
    d = (run_counts(a, universe).mean(axis=0) - run_counts(b, universe).mean(axis=0))[songs]
    whole = (run_counts(a, universe).sum(axis=0) - run_counts(b, universe).sum(axis=0))[songs]
    lo, hi = np.percentile(d[boot].sum(axis=1) / n_s[boot].sum(axis=1) * 100, [2.5, 97.5])
    return {'points': d.sum() / n_s.sum() * 100, 'tokens': d.sum(), 'low': lo,
            'high': hi, 'p': sign_flip_p(whole, flips)}


for c in ['enhanced', 'base', 'order2', 'order3', 'order4', 'frozen', 'no_dictionary',
          'no_estnltk', 'no_word_context', 'no_corpus_store']:
    put(f'replication.{c}', replication_cell(c))
REPLICATION_CONTRASTS = {
    'enhanced_vs_base': ('enhanced', 'base'),
    'frozen_vs_base': ('frozen', 'base'),
    'frozen_vs_no_corpus_store': ('frozen', 'no_corpus_store'),
    'enhanced_vs_frozen': ('enhanced', 'frozen'),
    'enhanced_vs_no_dictionary': ('enhanced', 'no_dictionary'),
    'enhanced_vs_no_estnltk': ('enhanced', 'no_estnltk'),
    'enhanced_vs_no_word_context': ('enhanced', 'no_word_context'),
    'enhanced_vs_no_corpus_store': ('enhanced', 'no_corpus_store'),
}
for name, (a, b) in REPLICATION_CONTRASTS.items():
    put(f'replication.{name}', replication_contrast(a, b))
adjusted = holm({name: V[f'replication.{name}.p'] for name in REPLICATION_CONTRASTS
                 if name != 'enhanced_vs_base'})
for name, p in adjusted.items():
    V[f'replication.{name}.p_adjusted'] = p
# The frozen-store gain over Base AI on the seven Seto songs and on the other songs.
SETO_SONGS = {r['poem_id'] for r in EVAL if r['dialect_subcategory_merged'] == 'Seto'}
for name, keep in [('seto', [j for j, s in enumerate(SONGS) if s in SETO_SONGS]),
                   ('without_seto', [j for j, s in enumerate(SONGS) if s not in SETO_SONGS])]:
    put(f'replication.frozen_vs_base.{name}',
        replication_contrast('frozen', 'base', n_s=N_ALL[keep], songs=keep,
                             boot=draw_boot_matrix(len(keep), BOOT_SEED, N_BOOT_MAIN),
                             flips=draw_flip_matrix(len(keep), BOOT_SEED)))
orders = ['enhanced', 'order2', 'order3', 'order4']
means = [V[f'replication.{o}.mean'] for o in orders]
V['replication.order_span'] = max(means) - min(means)
for o in orders:
    V[f'replication.order_gap.{o}'] = V[f'replication.{o}.mean'] - V['replication.base.mean']
V['replication.order_gap_min'] = min(V[f'replication.order_gap.{o}'] for o in orders)
V['replication.order_gap_max'] = max(V[f'replication.order_gap.{o}'] for o in orders)
V['replication.base_above_original'] = V['replication.base.mean'] - V['manual.claude_base']
V['replication.enhanced_above_original'] = (V['replication.enhanced.mean']
                                            - V['manual.claude_enhanced'])
# The two Claude Opus 5 rows also gain less from the change of
# reference than either reviewed Claude configuration.
V['bridge.replication_gains_less'] = all(
    V[f'expert.opus_{c}'] - V[f'replication.{c}.mean'] < claude_margin
    for c in ('base', 'enhanced'))

# Sensitivity: the main comparisons without the seven model-selection songs.
kept = [j for j, s in enumerate(SONGS) if s not in MODEL_SELECTION_SONGS]
V['sensitivity.songs_left_out'] = len(SONGS) - len(kept)
flips_kept = draw_flip_matrix(len(kept), BOOT_SEED)
n_kept = N_ALL[kept]
for name, (a, b) in [('enhanced_vs_base', ('enhanced', 'base')),
                     ('enhanced_vs_no_dictionary', ('enhanced', 'no_dictionary'))]:
    d = run_counts(a).mean(axis=0)[kept] - run_counts(b).mean(axis=0)[kept]
    whole = run_counts(a).sum(axis=0)[kept] - run_counts(b).sum(axis=0)[kept]
    V[f'sensitivity.{name}.points'] = d.sum() / n_kept.sum() * 100
    V[f'sensitivity.{name}.p'] = sign_flip_p(whole, flips_kept)

# September 2025 prompts against the July 2025 ones, on both reference sets.
prompt_differences = []
for measure, universe, n_s in [('manual', ALL, N_ALL), ('expert', EXPERT, N_EXPERT)]:
    for arm, july in [('base', 'base'), ('enhanced', 'frozen')]:
        if measure == 'expert':
            saved = dict(RUN_HITS)
            for c in (f'september_{arm}', july):
                RUN_HITS[c] = [correct(f'{REPLICATION[c]}_run{r}', EXPERT_REF)
                               for r in (1, 2, 3)]
        result = replication_contrast(f'september_{arm}', july, universe, n_s)
        if measure == 'expert':
            RUN_HITS.update(saved)
        put(f'prompts.{measure}.{arm}', result)
        prompt_differences.append((measure, result))
V['prompts.max_abs_manual'] = max(abs(r['points']) for m, r in prompt_differences
                                  if m == 'manual')
V['prompts.max_abs_expert'] = max(abs(r['points']) for m, r in prompt_differences
                                  if m == 'expert')
V['prompts.intervals_including_zero'] = sum(1 for m, r in prompt_differences
                                            if r['low'] <= 0 <= r['high'])


# Regional and subcategory results with per-region bootstrap intervals.
def regional_interval(hit, mask):
    songs_r = sorted({EVAL[i]['poem_id'] for i in np.flatnonzero(mask)})
    local = {s: j for j, s in enumerate(songs_r)}
    song_r = np.array([local.get(r['poem_id'], 0) for r in EVAL])
    k = per_song(mask & hit, song_r, len(songs_r))
    n = per_song(mask, song_r, len(songs_r))
    lo, hi = percentile_ci(boot_ratio(k, n, draw_boot_matrix(len(songs_r))))
    return 100 * lo, 100 * hi


REGIONAL = ['claude_enhanced', 'claude_base', 'deepseek_enhanced', 'estnltk_dictionary',
            'pure_estnltk', 'lexicon_dialect', 'normalization']
for region in REGIONS:
    mask = REGION == region
    for s in REGIONAL:
        V[f'region.{region}.{s}'] = accuracy(HIT[s], mask)
        low, high = regional_interval(HIT[s], mask)
        V[f'region_low.{region}.{s}'] = low
        V[f'region_high.{region}.{s}'] = high
    V[f'region.{region}.dictionary_gain'] = (V[f'region.{region}.estnltk_dictionary']
                                             - V[f'region.{region}.pure_estnltk'])
V['region.western_best'] = all(
    V[f'region.Western Estonian.{s}'] == max(V[f'region.{r}.{s}'] for r in REGIONS)
    for s in ['claude_enhanced', 'estnltk_dictionary'])
V['region.north_east_between'] = all(
    V[f'region.South Estonian.{s}'] < V[f'region.North+East Estonian.{s}']
    < V[f'region.Western Estonian.{s}'] for s in ['claude_enhanced', 'estnltk_dictionary'])
V['region.margin_largest_south'] = all(
    V[f'region.South Estonian.claude_enhanced'] - V[f'region.South Estonian.{s}']
    == max(V[f'region.{r}.claude_enhanced'] - V[f'region.{r}.{s}'] for r in REGIONS)
    for s in ['estnltk_dictionary', 'pure_estnltk', 'lexicon_dialect', 'normalization'])

SUBCATEGORIES = ['Western Northern', 'Island Western', 'Northeast Estonian',
                 'Western Southern', 'Tartu', 'Mulgi', 'Northern Central', 'Seto', 'Võro']
large = sorted(c for c, n in Counter(SUBCATEGORY).items() if n > 120)
V['subcategory.count_over_120'] = len(large)
assert sorted(SUBCATEGORIES) == large
for c in SUBCATEGORIES:
    mask = SUBCATEGORY == c
    for s in ['claude_enhanced', 'claude_base', 'estnltk_dictionary', 'pure_estnltk',
              'lexical', 'lexicon_dialect', 'normalization']:
        V[f'subcategory.{c}.{s}'] = accuracy(HIT[s], mask)
for s in ['claude_enhanced', 'estnltk_dictionary']:
    values = [V[f'subcategory.{c}.{s}'] for c in SUBCATEGORIES]
    V[f'subcategory.spread.{s}'] = max(values) - min(values)
V['subcategory.voro_songs'] = len({r['poem_id'] for r in EVAL
                                   if r['dialect_subcategory_merged'] == 'Võro'})
# 'About one word in five': rounds to 20 per cent at the nearest 5.
V['subcategory.seto_voro_lexicon_one_in_five'] = all(
    round(V[f'subcategory.{c}.lexicon_dialect'] / 5) * 5 == 20 for c in ['Seto', 'Võro'])
V['subcategory.enhanced_above_rule_based'] = sum(
    1 for c in SUBCATEGORIES
    if V[f'subcategory.{c}.claude_enhanced'] > max(V[f'subcategory.{c}.{s}'] for s in
                                                    ['estnltk_dictionary', 'pure_estnltk',
                                                     'lexical', 'lexicon_dialect',
                                                     'normalization']))
# Share of entries where Enhanced AI departs from each reference.
V['divergence.manual'] = 100 - V['manual.claude_enhanced']
V['divergence.expert'] = 100 - V['expert.claude_enhanced']

# Corpus store at the start of the July 2025 runs: evaluated words whose form,
# in lower case, has a store entry from one of the seven texts processed before
# the evaluated songs (entries 1 to 7 of processed_texts).
with open(DATA / 'corpus_database.json', encoding='utf-8') as f:
    STORE = json.load(f)
EARLY = set(STORE['processed_texts'][:7])
assert not EARLY & set(SONGS)
EARLY_FORMS = {w.lower() for w, e in STORE['words'].items() if EARLY & set(e['sources'])}
COVERED = np.array([r['original_word'].lower() in EARLY_FORMS for r in EVAL])
for name, mask in [('covered', COVERED), ('other', ~COVERED)]:
    put(f'store_start.{name}', {
        'rows': int(mask.sum()),
        'claude_enhanced': accuracy(HIT['claude_enhanced'], mask),
        'claude_base': accuracy(HIT['claude_base'], mask),
        'lead_words': int(HIT['claude_enhanced'][mask].sum() - HIT['claude_base'][mask].sum())})
    V[f'store_start.{name}.lead'] = (V[f'store_start.{name}.claude_enhanced']
                                     - V[f'store_start.{name}.claude_base'])

# The printed values, place by place, in the article and the configuration
# record; number words are mapped to numbers.
WORDS = {'none': 0, 'two': 2, 'six': 6, 'seven': 7, 'nine': 9, 'eleven': 11,
         'nineteen': 19, 'twenty-seven': 27, 'twenty-eight': 28, 'thirty-two': 32,
         'thirty-four': 34, 'thirty-eight': 38, 'forty-three': 43, 'forty-five': 45,
         'fifty-one': 51, 'fifty-three': 53,
         'fifty-five': 55, 'seventy-seven': 77, 'eighteen': 18, 'three': 3, 'four': 4}

ARTICLE = []


def cite(location, entries):
    for quantity, printed, description in entries:
        ARTICLE.append((location, description, quantity, printed))


T1_LABEL = {
    'lexical': 'Lexical resources',
    'normalization': 'Neural normalizer + EstNLTK',
    'lexicon_dialect': 'Lexicon-enhanced lemmatizer (Vabamorf + index)',
    'lexicon_vabamorf': 'Lexicon-enhanced lemmatizer (Vabamorf)',
    'glilem': 'GliLem',
    'stanza': 'Stanza',
    'pure_estnltk': 'Pure EstNLTK',
    'estnltk_dictionary': 'EstNLTK + Dictionary',
    'deepseek_base': 'Base AI (DeepSeek)',
    'deepseek_enhanced': 'Enhanced AI (DeepSeek)',
    'claude_base': 'Base AI (Claude)',
    'claude_enhanced': 'Enhanced AI (Claude)',
}
T1 = {
    'lexical': ('47.0', '39.6', '53.6', '42.1', '48.1', None, None),
    'normalization': ('47.7', '41.6', '53.0', '41.0', '49.3', None, None),
    'lexicon_dialect': ('52.9', '42.0', '62.8', '51.3', '55.2', None, None),
    'lexicon_vabamorf': ('54.7', '43.2', '65.3', '52.9', '57.0', None, None),
    'glilem': ('53.3', '42.2', '63.6', '49.8', '55.5', None, None),
    'stanza': ('54.6', '43.7', '64.6', '51.6', '57.1', None, None),
    'pure_estnltk': ('55.6', '44.0', '66.2', '54.2', '57.9', None, None),
    'estnltk_dictionary': ('60.6', '51.1', '69.0', '60.6', '63.3', None, None),
    'deepseek_base': ('75.5', '70.4', '80.1', '69.7', '79.1', '59.6', '+20.1'),
    'deepseek_enhanced': ('78.5', '73.5', '82.8', '73.7', '82.1', '59.9', '+18.9'),
    'claude_base': ('84.8', '81.4', '87.9', '81.5', '89.7', '62.7', '+9.4'),
    'claude_enhanced': ('85.4', '82.1', '88.3', '82.5', '90.4', '63.6', '+10.0'),
}

cite('Abstract', [
    ('corpus.rows', '3,216', 'evaluated word-level entries'),
    ('corpus.songs', 'forty-five', 'songs'),
    ('corpus.collection_areas', 'twenty-seven', 'collection areas'),
    ('manual.claude_enhanced', '85.4', 'Enhanced AI (Claude), agreement with the manual annotations (%)'),
    ('manual.claude_base', '84.8', 'Base AI (Claude), agreement with the manual annotations (%)'),
    ('manual.estnltk_dictionary', '60.6', 'EstNLTK + Dictionary, agreement with the manual annotations (%)'),
    ('neural.all_below_estnltk_dictionary', 'yes', 'all zero-shot neural systems below EstNLTK + Dictionary'),
    ('expert.claude_enhanced', '90.4', 'Enhanced AI (Claude), agreement with the secondary reference set (%)'),
])
cite('Section 1', [
    ('manual.estnltk_dictionary', '60.6', 'EstNLTK + Dictionary, agreement with the manual annotations (%)'),
    ('manual.claude_enhanced', '85.4', 'Enhanced AI (Claude), agreement with the manual annotations (%)'),
])
cite('Section 3', [
    ('coverage.lexical_all_rows', '2,723', 'evaluated words with a lexical-resources candidate'),
    ('corpus.rows', '3,216', 'evaluated word-level entries'),
    ('coverage.lexical_all_pct', '84.7', 'lexical-resources coverage of all evaluated words (%)'),
])
cite('Section 4.1', [
    ('corpus.songs', 'forty-five', 'songs'),
    ('corpus.collection_areas', 'twenty-seven', 'collection areas'),
    ('corpus.rows', '3,216', 'evaluated word-level entries'),
    ('corpus.excluded_rows', 'seventy-seven', 'word-level entries excluded from the secondary reference set'),
    ('corpus.expert_rows', '3,139', 'word-level entries in the secondary reference set'),
    ('corpus.region_songs.South Estonian', 'eighteen', 'South Estonian songs'),
    ('corpus.region_entries.South Estonian', '1,504', 'South Estonian entries'),
    ('corpus.south_share_pct', '46.8', 'South Estonian share of entries (%)'),
    ('corpus.seto_entries', '724', 'Seto entries'),
    ('corpus.seto_songs', 'seven', 'Seto songs'),
    ('corpus.seto_share_pct', '22.5', 'Seto share of all evaluated word-level entries (%)'),
    ('corpus.setomaa_most_represented', 'yes', 'Setomaa is the most represented collection area'),
    ('corpus.area_entries.Setumaa', '724', 'entries from Setomaa'),
    ('corpus.area_entries.Tõstamaa', '440', 'entries from Tõstamaa'),
    ('corpus.area_entries.Jämaja', '258', 'entries from Jämaja'),
    ('corpus.area_entries.Märjamaa', '175', 'entries from Märjamaa'),
])
cite('Section 4.2', [
    ('coverage.lexical_empty', '493', 'entries without a lexical-resources candidate'),
    ('corpus.rows', '3,216', 'evaluated word-level entries'),
])
cite('Section 4.3', [
    ('review.cases', '578', 'reviewed word-level entries'),
    ('review.ai_without_manual_pct', '10.9', 'share of reviewed word-level entries where Enhanced AI (Claude) or Base AI (Claude) was accepted but the manual annotation was not (%)'),
])
cite('Section 5.1', [
    ('original.manual_gain_over_pure', '29.8', 'Enhanced AI (Claude) minus Pure EstNLTK, manual annotations (points)'),
    ('original.expert_gain_over_pure', '32.6', 'Enhanced AI (Claude) minus Pure EstNLTK, secondary reference set (points)'),
    ('original.claude_gain.points', '0.59', 'Enhanced AI (Claude) minus Base AI (Claude), agreement with the manual annotations, original evaluation (points)'),
    ('original.claude_gain.tokens', 'nineteen', 'Enhanced AI (Claude) minus Base AI (Claude), original evaluation (words matching the manual annotations)'),
    ('original.claude_gain.low', '−0.47', 'Enhanced AI (Claude) minus Base AI (Claude), interval low'),
    ('original.claude_gain.high', '1.69', 'Enhanced AI (Claude) minus Base AI (Claude), interval high'),
    ('original.claude_gain.p', '0.434', 'Enhanced AI (Claude) minus Base AI (Claude), song-level test p'),
    ('corpus.rows', '3,216', 'evaluated word-level entries'),
    ('mcnemar.cf.manual.p', '0.048', 'word-level exact McNemar test, Enhanced AI (Claude) vs Base AI (Claude), case-insensitive, manual annotations, p'),
    ('mcnemar.cf.expert.p', '0.025', 'word-level exact McNemar test, Enhanced AI (Claude) vs Base AI (Claude), case-insensitive, secondary reference set, p'),
])
cite('Configuration record 7', [
    ('mcnemar.cf.manual.enhanced_only', 'fifty-one', 'Enhanced AI (Claude) vs Base AI (Claude), words correct only for Enhanced AI, case-insensitive, manual annotations'),
    ('mcnemar.cf.manual.base_only', 'thirty-two', 'Enhanced AI (Claude) vs Base AI (Claude), words correct only for Base AI, case-insensitive, manual annotations'),
])
cite('Section 5.1', [
    ('manual.lexical', '47.0', 'Lexical resources, edit-distance ranking, agreement with the manual annotations (%)'),
    ('coverage.lexical_all_pct', '84.7', 'lexical-resources coverage of all evaluated words (%)'),
    ('corpus.rows', '3,216', 'evaluated word-level entries'),
])
cite('Section 5.1', [
    ('manual.lexical_frequency_ranked', '53.2', 'Lexical resources, frequency re-ranking, agreement with the manual annotations (%)'),
    ('manual.estnltk_dictionary', '60.6', 'EstNLTK + Dictionary, agreement with the manual annotations (%)'),
    ('gap.estnltk_dictionary', '−24.8', 'EstNLTK + Dictionary minus Enhanced AI (Claude), agreement with the manual annotations (points)'),
])
cite('Section 5.5', [
    ('original.best_other_non_llm', '54.7', 'highest agreement with the manual annotations among frequency re-ranking and the zero-shot neural baselines (%)'),
])
for s, (m, lo, hi, typ, exp, sim, assoc) in T1.items():
    label = T1_LABEL[s]
    entries = [(f'manual.{s}', m, f'{label}, agreement with the manual annotations (%)'),
               (f'manual_low.{s}', lo, f'{label}, manual annotations, interval low'),
               (f'manual_high.{s}', hi, f'{label}, manual annotations, interval high')]
    if typ:
        entries.append((f'types.{s}', typ, f'{label}, type-level agreement with the manual annotations, every occurrence of the form, case-insensitive (%)'))
    entries.append((f'expert.{s}', exp, f'{label}, agreement with the secondary reference set (%)'))
    if sim:
        entries.append((f'translation.similarity.{s}', sim, f'{label}, mean similarity to the manual translation (%)'))
        entries.append((f'translation.association.{s}', assoc,
                        f'{label}, translation-lemma association: agreement with the manual annotations, high-similarity group (at least 0.7) minus low-similarity group (points)'))
    cite('Table 1', entries)
cite('Table 1', [
    ('replication.base.mean', '88.2', 'Base AI (Claude Opus 5), agreement with the manual annotations, mean of three runs (%)'),
    ('replication.base.low', '84.5', 'Base AI (Claude Opus 5), manual annotations, interval low'),
    ('replication.base.high', '91.1', 'Base AI (Claude Opus 5), manual annotations, interval high'),
    ('replication.enhanced.mean', '89.5', 'Enhanced AI (Claude Opus 5), agreement with the manual annotations, mean of three runs (%)'),
    ('replication.enhanced.low', '86.3', 'Enhanced AI (Claude Opus 5), manual annotations, descriptive spread low'),
    ('replication.enhanced.high', '92.2', 'Enhanced AI (Claude Opus 5), manual annotations, descriptive spread high'),
    ('expert.opus_base', '92.1', 'Base AI (Claude Opus 5), agreement with the secondary reference set, mean of three runs (%)'),
    ('expert.opus_enhanced', '93.5', 'Enhanced AI (Claude Opus 5), agreement with the secondary reference set, mean of three runs (%)'),
    ('types.opus_base', '86.3', 'Base AI (Claude Opus 5), type-level agreement with the manual annotations, every occurrence of the form, case-insensitive, mean of three runs (%)'),
    ('types.opus_enhanced', '88.4', 'Enhanced AI (Claude Opus 5), type-level agreement with the manual annotations, every occurrence of the form, case-insensitive, mean of three runs (%)'),
])
cite('Table 1', [
    ('corpus.rows', '3,216', 'evaluated word-level entries'),
    ('types.count', '1,750', 'distinct word forms among all evaluated words'),
    ('corpus.expert_rows', '3,139', 'word-level entries in the secondary reference set'),
])
cite('Figure 2', [(f'manual.{s}', v, f'{T1_LABEL[s]}, agreement with the manual annotations (%)')
                  for s, v in [('lexical', '47.0'), ('normalization', '47.7'),
                               ('lexicon_dialect', '52.9'), ('glilem', '53.3'),
                               ('stanza', '54.6'), ('lexicon_vabamorf', '54.7'),
                               ('pure_estnltk', '55.6'), ('estnltk_dictionary', '60.6'),
                               ('deepseek_base', '75.5'), ('deepseek_enhanced', '78.5'),
                               ('claude_base', '84.8'), ('claude_enhanced', '85.4')]])
cite('Figure 2', [
    ('replication.base.mean', '88.2', 'Base AI (Claude Opus 5), agreement with the manual annotations, mean of three runs (%)'),
    ('replication.enhanced.mean', '89.5', 'Enhanced AI (Claude Opus 5), agreement with the manual annotations, mean of three runs (%)'),
])
cite('Section 5.2', [
    ('replication.base_above_original', 'three', 'Base AI, replication with Claude Opus 5 minus original evaluation (points, lower end)'),
    ('replication.enhanced_above_original', 'four', 'Enhanced AI, replication with Claude Opus 5 minus original evaluation (points, upper end)'),
    ('replication.enhanced_vs_base.points', '+1.34', 'Enhanced AI (Claude Opus 5) minus Base AI (Claude Opus 5), agreement with the manual annotations, live store (points)'),
    ('replication.enhanced_vs_base.tokens', 'forty-three', 'Enhanced AI (Claude Opus 5) minus Base AI (Claude Opus 5), mean across runs (words matching the manual annotations)'),
    ('corpus.rows', '3,216', 'evaluated word-level entries'),
    ('replication.enhanced_vs_base.low', '0.33', 'Enhanced AI (Claude Opus 5) minus Base AI (Claude Opus 5), interval low'),
    ('replication.enhanced_vs_base.high', '2.56', 'Enhanced AI (Claude Opus 5) minus Base AI (Claude Opus 5), interval high'),
    ('replication.enhanced_vs_base.p', '0.015', 'Enhanced AI (Claude Opus 5) minus Base AI (Claude Opus 5), p'),
    ('replication.enhanced_vs_no_dictionary.points', '−0.64', 'dictionary lookup contribution (points)'),
    ('replication.enhanced_vs_no_dictionary.low', '−1.90', 'dictionary lookup contribution, interval low'),
    ('replication.enhanced_vs_no_dictionary.high', '0.33', 'dictionary lookup contribution, interval high'),
    ('replication.enhanced_vs_no_dictionary.p_adjusted', '1.000', 'dictionary lookup contribution, adjusted p'),
    ('replication.enhanced_vs_no_estnltk.points', '+0.34', 'EstNLTK contribution (points)'),
    ('replication.enhanced_vs_no_word_context.points', '+0.31', 'word-level context contribution (points)'),
    ('replication.enhanced_vs_no_corpus_store.points', '+0.17', 'corpus store contribution (points)'),
])
cite('Configuration record 7', [
    ('sensitivity.songs_left_out', 'seven', 'songs used for model selection'),
])
cite('Section 5.2', [
    ('replication.order_span', '1.03', 'Enhanced AI (Claude Opus 5), range of mean agreement with the manual annotations across four processing orders (points)'),
    ('replication.order_gap_min', '+0.31', 'smallest Enhanced AI (Claude Opus 5) order mean minus Base AI (Claude Opus 5) original-order mean, agreement with the manual annotations (points)'),
    ('replication.order_gap_max', '+1.34', 'largest Enhanced AI (Claude Opus 5) order mean minus Base AI (Claude Opus 5) original-order mean, agreement with the manual annotations (points)'),
    ('replication.frozen_vs_base.points', '+1.65', 'Enhanced AI (Claude Opus 5) with frozen store minus Base AI (Claude Opus 5), agreement with the manual annotations (points)'),
    ('replication.frozen_vs_base.tokens', 'fifty-three', 'Enhanced AI (Claude Opus 5) with frozen store minus Base AI (Claude Opus 5), mean across runs (words matching the manual annotations)'),
    ('corpus.rows', '3,216', 'evaluated word-level entries'),
    ('replication.frozen_vs_base.low', '0.66', 'Enhanced AI (Claude Opus 5) with frozen store minus Base AI (Claude Opus 5), interval low'),
    ('replication.frozen_vs_base.high', '2.78', 'Enhanced AI (Claude Opus 5) with frozen store minus Base AI (Claude Opus 5), interval high'),
    ('replication.frozen_vs_base.p', '0.003', 'Enhanced AI (Claude Opus 5) with frozen store minus Base AI (Claude Opus 5), p'),
    ('replication.frozen_vs_base.p_adjusted', '0.019', 'Enhanced AI (Claude Opus 5) with frozen store minus Base AI (Claude Opus 5), adjusted p'),
    ('replication.frozen_vs_base.seto.points', '4.14', 'Enhanced AI (Claude Opus 5) with frozen store minus Base AI (Claude Opus 5), Seto songs (points)'),
    ('corpus.seto_songs', 'seven', 'Seto songs'),
    ('replication.frozen_vs_base.without_seto.points', '0.92', 'Enhanced AI (Claude Opus 5) with frozen store minus Base AI (Claude Opus 5), songs other than the Seto songs (points)'),
    ('replication.frozen_vs_base.without_seto.low', '0.31', 'the same, songs other than the Seto songs, interval low'),
    ('replication.frozen_vs_base.without_seto.high', '1.67', 'the same, songs other than the Seto songs, interval high'),
    ('replication.enhanced_vs_frozen.points', '−0.31', 'live store minus frozen store (points; the article reports the reverse difference, frozen minus live)'),
    ('replication.enhanced_vs_frozen.low', '−1.29', 'live store minus frozen store, interval low (the article reports the reverse interval)'),
    ('replication.enhanced_vs_frozen.high', '0.37', 'live store minus frozen store, interval high (the article reports the reverse interval)'),
    ('replication.frozen_vs_no_corpus_store.points', '+0.48', 'frozen store minus no corpus store (points)'),
    ('replication.frozen_vs_no_corpus_store.low', '−0.62', 'frozen store minus no corpus store, interval low'),
    ('replication.frozen_vs_no_corpus_store.high', '1.95', 'frozen store minus no corpus store, interval high'),
    ('replication.frozen_vs_no_corpus_store.p', '0.790', 'frozen store minus no corpus store, p'),
])
T2 = [('enhanced', 'Enhanced AI (all components)', '89.49', '1.16', '86.27', '92.19'),
      ('base', 'Base AI', '88.15', '0.32', '84.54', '91.07'),
      ('frozen', 'Enhanced AI, frozen store', '89.80', '0.27', '86.81', '92.17'),
      ('no_dictionary', 'Enhanced AI without dictionary lookup', '90.13', '0.16', '87.74', '92.15'),
      ('no_estnltk', 'Enhanced AI without EstNLTK', '89.15', '0.65', '85.98', '91.85'),
      ('no_word_context', 'Enhanced AI without word-level context', '89.18', '0.14', '86.24', '91.52'),
      ('no_corpus_store', 'Enhanced AI without corpus store', '89.32', '0.53', '86.17', '92.09')]
for c, label, mean, sd, lo, hi in T2:
    cite('Table 2', [(f'replication.{c}.mean', mean, f'{label}, original order, mean agreement with the manual annotations (%)'),
                     (f'replication.{c}.sd', sd, f'{label}, original order, run-to-run SD of agreement with the manual annotations (percentage points)'),
                     (f'replication.{c}.low', lo, f'{label}, agreement with the manual annotations, lower bootstrap confidence bound (%)'),
                     (f'replication.{c}.high', hi, f'{label}, agreement with the manual annotations, upper bootstrap confidence bound (%)')])
cite('Table 2', [('corpus.rows', '3,216', 'evaluated word-level entries')])
cite('Section 5.3', [
    ('types.claude_gain_forms', 'eighteen', 'word forms counted correct at type level for Enhanced AI (Claude) minus those for Base AI (Claude), manual annotations'),
    ('types.count', '1,750', 'distinct word forms among all evaluated words'),
    ('types.claude_gain', '1.03', 'Enhanced AI (Claude) advantage over Base AI (Claude) in type-level agreement with the manual annotations, every occurrence of the form, case-insensitive (points)'),
    ('variation.claude_enhanced', 'twenty-eight', 'word forms with varying Enhanced AI lemmas'),
    ('variation.claude_base', 'thirty-four', 'word forms with varying Base AI lemmas'),
    ('variation.manual', 'thirty-eight', 'word forms with varying manual lemmas'),
    ('variation.expert', 'twenty-seven', 'word forms with varying lemmas in the secondary reference set'),
    ('corpus.rows', '3,216', 'evaluated word-level entries'),
    ('translation.high_share.claude_enhanced', '46.2', 'Enhanced AI (Claude), words with at least 70% similarity to the manual translation (%)'),
    ('translation.agreement_high.claude_enhanced', '90.8', 'Enhanced AI (Claude), lemma agreement with the manual annotations, high-similarity translations (%)'),
    ('translation.agreement_low.claude_enhanced', '80.8', 'Enhanced AI (Claude), lemma agreement with the manual annotations, low-similarity translations (%)'),
    ('translation.claude_association_smaller', 'yes', 'association between translation similarity and lemma agreement with the manual annotations is smaller for Enhanced AI (Claude) than for Enhanced AI (DeepSeek)'),
    ('translation.claude_association_smaller_all_cutoffs', 'yes', 'the association stays smaller for Enhanced AI (Claude) than for Enhanced AI (DeepSeek) at every cut-off from 0.5 to 0.9'),
])
cite('Section 5.4', [
    ('corpus.rows', '3,216', 'evaluated word-level entries'),
    ('corpus.expert_rows', '3,139', 'word-level entries in the secondary reference set'),
    ('corpus.expert_rows_pct', '97.6', 'share of the evaluated entries kept in the secondary reference set (%)'),
    ('corpus.excluded_rows', 'seventy-seven', 'word-level entries excluded from the secondary reference set'),
    ('review.cases', '578', 'reviewed word-level entries'),
    ('review.all_matched', '117', 'reviewed word-level entries where the manual annotation, Enhanced AI (Claude) and Base AI (Claude) were all accepted'),
    ('corpus.expert_rows', '3,139', 'word-level entries in the secondary reference set'),
    ('reference.single_lemma', '3,084', 'entries with a single accepted lemma'),
    ('reference.single_lemma_pct', '98.2', 'share with a single accepted lemma (%)'),
    ('reference.multi_lemma', 'fifty-five', 'entries with multiple accepted lemmas'),
    ('reference.multi_lemma_pct', '1.8', 'share with multiple accepted lemmas (%)'),
    ('review.manual_only', '270', 'reviewed word-level entries where only the manual annotation was accepted'),
])
cite('Table 3', [
    ('review.manual_only_pct', '46.7', 'share of reviewed word-level entries where only the manual annotation was accepted (%)'),
])
cite('Section 5.4', [
    ('bridge.flips_up.claude_enhanced', '112', 'Enhanced AI (Claude) word-level entries in the secondary reference set turned correct by the change of reference, case-insensitive'),
    ('bridge.flips_up.claude_base', '109', 'Base AI (Claude) word-level entries in the secondary reference set turned correct by the change of reference, case-insensitive'),
    ('bridge.flips_down.claude_enhanced', 'none', 'Enhanced AI (Claude) word-level entries in the secondary reference set turned incorrect by the change of reference, case-insensitive'),
    ('bridge.flips_down.claude_base', 'none', 'Base AI (Claude) word-level entries in the secondary reference set turned incorrect by the change of reference, case-insensitive'),
    ('bridge.outside_systems_gain_less', 'yes', 'every system outside Base AI (Claude) and Enhanced AI (Claude) gains less from the change of reference than either member of the Claude pair'),
    ('corpus.excluded_rows', 'seventy-seven', 'word-level entries excluded from the secondary reference set'),
    ('bridge.exclusion_effect', '1.43', 'Enhanced AI (Claude), effect of the exclusions (points)'),
    ('bridge.adjudication_effect', '3.57', "Enhanced AI (Claude), effect of the expert's decisions in review (points)"),
    ('bridge.replication_gains_less', 'yes', 'both Claude Opus 5 replication rows gain less from the change of reference than either member of the Claude pair'),
])
cite('Table 3', [
    ('review.ai_without_manual', '63', 'reviewed word-level entries where Enhanced AI (Claude) or Base AI (Claude) was accepted but the manual annotation was not'),
    ('review.ai_without_manual_pct', '10.9', 'share of reviewed word-level entries where Enhanced AI (Claude) or Base AI (Claude) was accepted but the manual annotation was not (%)'),
    ('review.enhanced_without_manual', '59', 'reviewed word-level entries where Enhanced AI (Claude) was accepted and the manual annotation was not'),
    ('review.enhanced_without_manual_pct', '10.2', 'share of all reviewed word-level entries where Enhanced AI (Claude) was accepted and the manual annotation was not (%)'),
    ('review.manual_only', '270', 'reviewed word-level entries where only the manual annotation was accepted'),
    ('review.manual_only_pct', '46.7', 'share of reviewed word-level entries where only the manual annotation was accepted (%)'),
    ('review.all_matched', '117', 'reviewed word-level entries where the manual annotation, Enhanced AI (Claude) and Base AI (Claude) were all accepted'),
    ('review.all_matched_pct', '20.2', 'share of reviewed word-level entries where the manual annotation, Enhanced AI (Claude) and Base AI (Claude) were all accepted (%)'),
    ('review.none_matched', '64', 'reviewed word-level entries where no candidate was accepted'),
    ('review.none_matched_pct', '11.1', 'share of reviewed word-level entries where no candidate was accepted (%)'),
    ('review.mixed', '64', 'reviewed word-level entries where the manual annotation and exactly one of Enhanced AI (Claude) and Base AI (Claude) were accepted'),
    ('review.mixed_pct', '11.1', 'share of reviewed word-level entries where the manual annotation and exactly one of Enhanced AI (Claude) and Base AI (Claude) were accepted (%)'),
    ('review.five_categories_sum', '578', 'reviewed word-level entries, total across the five categories'),
    ('review.other_codes', 'none', 'reviewed word-level entries with other decision codes'),
])
cite('Section 5.4', [
    ('review.manual_included_pct', '78.0', 'share of reviewed word-level entries where the manual annotation was accepted (%)'),
    ('review.ai_without_manual_pct', '10.9', 'share of reviewed word-level entries where Enhanced AI (Claude) or Base AI (Claude) was accepted but the manual annotation was not (%)'),
])
cite('Section 5.5', [
    ('manual.deepseek_enhanced', '78.5', 'Enhanced AI (DeepSeek), agreement with the manual annotations (%)'),
    ('original.deepseek_gap_to_claude', '6.9', 'Enhanced AI (DeepSeek) below Enhanced AI (Claude), agreement with the manual annotations (points)'),
    ('manual.deepseek_base', '75.5', 'Base AI (DeepSeek), agreement with the manual annotations (%)'),
    ('original.deepseek_gain.points', '+3.02', 'DeepSeek gain from linguistic resources (points)'),
    ('original.deepseek_gain.low', '0.45', 'DeepSeek gain, interval low'),
    ('original.deepseek_gain.high', '5.40', 'DeepSeek gain, interval high'),
    ('original.deepseek_gain.p', '0.020', 'DeepSeek gain, p'),
    ('original.deepseek_gain.p_adjusted', '0.059', 'DeepSeek gain, adjusted p'),
    ('original.claude_gain.points', '+0.59', 'Enhanced AI (Claude) minus Base AI (Claude), agreement with the manual annotations, original evaluation (points)'),
    ('original.gain_difference.points', '+2.43', 'DeepSeek minus Claude difference in Enhanced AI gains over Base AI, agreement with the manual annotations (points)'),
    ('original.gain_difference.low', '−0.26', 'DeepSeek minus Claude difference in Enhanced AI gains over Base AI, interval low'),
    ('original.gain_difference.high', '5.27', 'DeepSeek minus Claude difference in Enhanced AI gains over Base AI, interval high'),
    ('original.gain_difference.p', '0.081', 'DeepSeek minus Claude difference in Enhanced AI gains over Base AI, p'),
    ('original.gain_difference.p_adjusted', '0.163', 'DeepSeek minus Claude difference in Enhanced AI gains over Base AI, adjusted p'),
    ('manual.normalization', '47.7', 'Neural normalizer + EstNLTK, agreement with the manual annotations (%)'),
    ('neural.normalization_lowest', 'yes', 'Neural normalizer + EstNLTK has the lowest agreement with the manual annotations among the zero-shot neural baselines'),
])
cite('Configuration record 7', [
    ('neural.identity', '54.85', 'identity control, agreement with the manual annotations (%)'),
])
cite('Section 5.5', [
    ('manual.pure_estnltk', '55.6', 'Pure EstNLTK, agreement with the manual annotations (%)'),
])
cite('Configuration record 7', [
    ('neural.lowercased', '55.63', 'lowercased control, agreement with the manual annotations (%)'),
])
cite('Section 5.5', [
    ('neural.normalization_below_identity', '7.15', 'Neural normalizer + EstNLTK below the identity control (points)'),
    ('neural.normalization_vs_identity.low', '−13.36', 'normalization minus identity, interval low'),
    ('neural.normalization_vs_identity.high', '−0.75', 'normalization minus identity, interval high'),
    ('neural.normalization_vs_identity.p', '0.045', 'normalization minus identity, p'),
    ('neural.repaired', '56.4', 'normalization with the letter-restoring step, agreement with the manual annotations (%)'),
])
cite('Section 4.1', [
    ('corpus.south_share_pct', '46.8', 'South Estonian share of entries (%)'),
])
cite('Configuration record 7', [
    ('neural.repaired', '56.41', 'normalization with the letter-restoring step, agreement with the manual annotations (%)'),
    ('neural.repaired_vs_lowercased.points', '+0.78', 'letter-restoring step minus lowercased control (points)'),
    ('neural.repaired_vs_lowercased.low', '−3.37', 'letter-restoring step minus lowercased control, interval low'),
    ('neural.repaired_vs_lowercased.high', '4.99', 'letter-restoring step minus lowercased control, interval high'),
])
cite('Section 5.5', [
    ('manual.lexicon_dialect', '52.9', 'Lexicon-enhanced lemmatizer (Vabamorf + index), agreement with the manual annotations (%)'),
    ('manual.lexicon_vabamorf', '54.7', 'Lexicon-enhanced lemmatizer (Vabamorf), agreement with the manual annotations (%)'),
    ('manual.estnltk_dictionary', '60.6', 'EstNLTK + Dictionary, agreement with the manual annotations (%)'),
    ('neural.lexicon_vs_estnltk_dictionary.points', '−7.71', 'Lexicon-enhanced lemmatizer (Vabamorf + index) minus EstNLTK + Dictionary (points)'),
    ('neural.lexicon_vs_estnltk_dictionary.low', '−10.39', 'Lexicon-enhanced lemmatizer (Vabamorf + index) minus EstNLTK + Dictionary, interval low'),
    ('neural.lexicon_vs_estnltk_dictionary.high', '−4.87', 'Lexicon-enhanced lemmatizer (Vabamorf + index) minus EstNLTK + Dictionary, interval high'),
    ('neural.lexicon_vs_estnltk_dictionary.p', '<0.001', 'Lexicon-enhanced lemmatizer (Vabamorf + index) minus EstNLTK + Dictionary, p'),
    ('manual.claude_enhanced', '85.4', 'Enhanced AI (Claude), agreement with the manual annotations (%)'),
    ('neural.glilem_below_estnltk_dictionary', '7.25', 'GliLem below EstNLTK + Dictionary (points)'),
    ('neural.glilem_vs_estnltk_dictionary.low', '−10.31', 'GliLem minus EstNLTK + Dictionary, interval low'),
    ('neural.glilem_vs_estnltk_dictionary.high', '−4.11', 'GliLem minus EstNLTK + Dictionary, interval high'),
    ('neural.glilem_vs_estnltk_dictionary.p', '<0.001', 'GliLem minus EstNLTK + Dictionary, p'),
    ('neural.stanza_vs_pure.points', '−0.96', 'Stanza minus Pure EstNLTK (points)'),
    ('neural.stanza_vs_pure.low', '−2.30', 'Stanza minus Pure EstNLTK, interval low'),
    ('neural.stanza_vs_pure.high', '+0.25', 'Stanza minus Pure EstNLTK, interval high'),
    ('neural.stanza_vs_pure.p', '0.145', 'Stanza minus Pure EstNLTK, p'),
    ('neural.lowest', '47.7', 'lowest agreement with the manual annotations among the evaluated zero-shot neural baselines (%)'),
    ('neural.highest', '54.7', 'highest agreement with the manual annotations among the evaluated zero-shot neural baselines (%)'),
    ('neural.all_below_estnltk_dictionary', 'yes', 'all zero-shot neural systems below EstNLTK + Dictionary'),
])
REGION_LABEL = {'South Estonian': 'South Estonian', 'Western Estonian': 'Western Estonian',
                'North+East Estonian': 'North+East+Mid Estonian'}
cite('Section 5.6', [
    ('region.South Estonian.claude_enhanced', '80.5', 'Enhanced AI (Claude), South Estonian, agreement with the manual annotations (%)'),
    ('region.Western Estonian.claude_enhanced', '90.7', 'Enhanced AI (Claude), Western Estonian, agreement with the manual annotations (%)'),
])
F3_PRINTED = [('claude_enhanced', ('80.5', '75.1', '85.0'), ('90.7', '88.4', '93.4'), ('86.8', '79.3', '90.8')),
      ('estnltk_dictionary', ('42.8',), ('77.8',), ('71.3',)),
      ('pure_estnltk', ('34.4',), ('76.0',), ('68.6',)),
      ('lexicon_dialect', ('33.4', '23.9', '46.7'), ('71.0',), ('66.4',)),
      ('normalization', ('37.8', '31.3', '45.4'), ('56.9',), ('54.9',))]
for s, *cells in F3_PRINTED:
    for region, cell in zip(REGIONS, cells):
        label = f'{T1_LABEL[s]}, {REGION_LABEL[region]}'
        entries = [(f'region.{region}.{s}', cell[0], f'{label}, agreement with the manual annotations (%)')]
        if len(cell) == 3:
            entries += [(f'region_low.{region}.{s}', cell[1], f'{label}, agreement with the manual annotations, lower bootstrap confidence bound (%)'),
                        (f'region_high.{region}.{s}', cell[2], f'{label}, agreement with the manual annotations, upper bootstrap confidence bound (%)')]
        cite('Figure 3', entries)
cite('Section 5.6', [
    ('region.South Estonian.dictionary_gain', '8.4', 'dictionary gain for EstNLTK, South Estonian (points)'),
])
cite('Figure 3', [
    ('region.South Estonian.pure_estnltk', '34.4', 'Pure EstNLTK, South Estonian, agreement with the manual annotations (%)'),
    ('region.South Estonian.estnltk_dictionary', '42.8', 'EstNLTK + Dictionary, South Estonian, agreement with the manual annotations (%)'),
    ('region.Western Estonian.dictionary_gain', '1.8', 'dictionary gain for EstNLTK, Western Estonian: EstNLTK + Dictionary minus Pure EstNLTK, from the values in Figure 3 (points)'),
    ('region.North+East Estonian.dictionary_gain', '2.7', 'dictionary gain for EstNLTK, North+East+Mid Estonian: EstNLTK + Dictionary minus Pure EstNLTK, from the values in Figure 3 (points)'),
])
cite('Section 5.6', [
    ('subcategory.seto_voro_lexicon_one_in_five', 'yes', 'Lexicon-enhanced lemmatizer (Vabamorf + index) agrees with the manual annotations on about one word in five on Seto and on Võro'),
    ('region.margin_largest_south', 'yes', 'Enhanced AI (Claude) margin over Pure EstNLTK, EstNLTK + Dictionary, Lexicon-enhanced lemmatizer (Vabamorf + index) and Neural normalizer + EstNLTK is largest on South Estonian'),
    ('region.western_best', 'yes', 'Enhanced AI and EstNLTK + Dictionary best on Western Estonian'),
    ('region.north_east_between', 'yes', 'for Enhanced AI (Claude) and EstNLTK + Dictionary, North+East+Mid Estonian between South Estonian and Western Estonian'),
    ('subcategory.count_over_120', 'nine', 'subcategories with more than 120 entries'),
    ('subcategory.spread.claude_enhanced', '17.8', 'range of Enhanced AI (Claude) agreement with the manual annotations across the selected subcategories (points)'),
    ('subcategory.spread.estnltk_dictionary', '49.1', 'range of EstNLTK + Dictionary agreement with the manual annotations across the selected subcategories (points)'),
    ('subcategory.voro_songs', 'two', 'songs in the Võro subcategory'),
    ('subcategory.Võro.claude_enhanced', '75.3', 'Enhanced AI (Claude), Võro, agreement with the manual annotations (%)'),
    ('subcategory.Võro.claude_base', '77.6', 'Base AI (Claude), Võro, agreement with the manual annotations (%)'),
    ('subcategory.enhanced_above_rule_based', 'nine', 'subcategories where Enhanced AI (Claude) is above the rule-based and neural methods'),
])
F3 = {
    'pure_estnltk': [(34.38, 24.71, 48.15), (76.00, 72.16, 79.68), (68.63, 63.09, 77.94)],
    'estnltk_dictionary': [(42.75, 34.61, 53.37), (77.76, 74.67, 81.39), (71.32, 63.91, 77.11)],
    'normalization': [(37.77, 31.31, 45.39), (56.90, 53.60, 60.74), (54.90, 48.62, 58.35)],
    'lexicon_dialect': [(33.44, 23.88, 46.75), (71.01, 66.72, 75.23), (66.42, 62.53, 73.46)],
    'deepseek_enhanced': [(70.81, 64.13, 77.03), (87.27, 84.53, 90.39), (78.68, 72.80, 84.40)],
    'claude_base': [(79.99, 74.89, 85.04), (89.95, 87.37, 93.06), (86.27, 77.63, 90.43)],
    'claude_enhanced': [(80.45, 75.10, 85.05), (90.72, 88.38, 93.42), (86.76, 79.30, 90.80)],
}
for s, cells in F3.items():
    for region, cell in zip(REGIONS, cells):
        label = f'{T1_LABEL[s]}, {REGION_LABEL[region]}'
        entries = [(f'region.{region}.{s}', f'{cell[0]:.2f}', f'{label}, agreement with the manual annotations (%)')]
        if len(cell) == 3:
            entries += [(f'region_low.{region}.{s}', f'{cell[1]:.2f}', f'{label}, agreement with the manual annotations, lower bootstrap confidence bound (%)'),
                        (f'region_high.{region}.{s}', f'{cell[2]:.2f}', f'{label}, agreement with the manual annotations, upper bootstrap confidence bound (%)')]
        cite('Figure 3', entries)
cite('Figure 3', [(f'corpus.region_entries.{r}', n, f'{REGION_LABEL[r]} entries')
                  for r, n in zip(REGIONS, ['1,504', '1,304', '408'])])
cite('Figure 3', [(f'corpus.region_songs.{r}', n, f'{REGION_LABEL[r]} songs')
                  for r, n in zip(REGIONS, ['18', '13', '14'])])
F4 = {
    'claude_enhanced': [93.14, 93.04, 92.65, 88.46, 87.11, 86.90, 86.67, 77.49, 75.29],
    'estnltk_dictionary': [66.29, 79.90, 72.06, 79.20, 65.26, 51.72, 77.78, 30.80, 38.04],
    'lexicon_dialect': [62.86, 76.03, 61.03, 70.45, 62.63, 44.83, 73.33, 20.86, 19.22],
    'normalization': [61.71, 57.22, 59.56, 54.02, 51.05, 51.03, 54.81, 27.90, 38.43],
}
for s, values in F4.items():
    cite('Figure 4', [(f'subcategory.{c}.{s}', f'{v:.2f}', f'{T1_LABEL[s]}, {c}, agreement with the manual annotations (%)')
                      for c, v in zip(SUBCATEGORIES, values)])
cite('Figure 4', [('subcategory.count_over_120', 'nine', 'subcategories with more than 120 entries')])
cite('Section 6.1', [
    ('original.claude_gain.points', '+0.59', 'Enhanced AI (Claude) minus Base AI (Claude), agreement with the manual annotations, original evaluation (points)'),
    ('replication.enhanced_vs_base.points', '+1.34', 'Enhanced AI (Claude Opus 5) minus Base AI (Claude Opus 5), agreement with the manual annotations, live store (points)'),
    ('neural.all_below_estnltk_dictionary', 'yes', 'all zero-shot neural systems below EstNLTK + Dictionary'),
])
cite('Section 7', [
    ('divergence.manual', '15', 'Enhanced AI (Claude) word-level entries not matching the manual annotations, all evaluated entries (%, approximately)'),
    ('divergence.expert', '10', 'Enhanced AI (Claude) word-level entries not matching the secondary reference set, entries in that set (%, approximately)'),
])
cite('Section 5.2', [
    ('replication.enhanced_vs_base.points', '1.34', 'Enhanced AI (Claude Opus 5) minus Base AI (Claude Opus 5), agreement with the manual annotations, live store (points)'),
    ('replication.frozen_vs_base.points', '1.65', 'Enhanced AI (Claude Opus 5) with frozen store minus Base AI (Claude Opus 5), agreement with the manual annotations (points)'),
])
cite('Section 5.1', [
    ('original.claude_gain.points', '0.59', 'Enhanced AI (Claude) minus Base AI (Claude), agreement with the manual annotations, original evaluation (points)'),
])
cite('Section 8', [
    ('gap.claude_both_above_estnltk_dictionary_24', 'yes', 'Base AI (Claude) and Enhanced AI (Claude) both above EstNLTK + Dictionary by more than 24 points, agreement with the manual annotations'),
    ('manual.claude_enhanced', '85.4', 'Enhanced AI (Claude), agreement with the manual annotations (%)'),
    ('expert.claude_enhanced', '90.4', 'Enhanced AI (Claude), agreement with the secondary reference set (%)'),
])
cite('Appendix', [
    ('corpus.rows', '3,216', 'evaluated word-level entries'),
    ('corpus.songs', 'forty-five', 'songs'),
    ('corpus.collection_areas', 'twenty-seven', 'collection areas'),
])
cite('Configuration record 3', [
    ('prompts.max_abs_manual', '0.06', 'largest absolute prompt-version difference in agreement with the manual annotations, across Base AI (Claude Opus 5) and Enhanced AI (Claude Opus 5) (points)'),
    ('prompts.max_abs_expert', '0.23', 'largest absolute prompt-version difference in agreement with the secondary reference set, across Base AI (Claude Opus 5) and Enhanced AI (Claude Opus 5) (points)'),
    ('prompts.intervals_including_zero', 'four', 'prompt-version intervals that include zero'),
])
cite('Configuration record 6', [
    ('store_start.covered.rows', '622', 'evaluated words whose form has a corpus-store entry from the seven texts processed before the evaluated songs'),
    ('corpus.rows', '3,216', 'evaluated word-level entries'),
    ('store_start.covered.lead', '0.48', 'on those words, Enhanced AI (Claude) minus Base AI (Claude), agreement with the manual annotations (points)'),
    ('store_start.covered.claude_enhanced', '94.21', 'on those words, Enhanced AI (Claude), agreement with the manual annotations (%)'),
    ('store_start.covered.claude_base', '93.73', 'on those words, Base AI (Claude), agreement with the manual annotations (%)'),
    ('store_start.covered.lead_words', '3', 'on those words, Enhanced AI (Claude) minus Base AI (Claude), words matching the manual annotations'),
    ('store_start.other.lead', '0.62', 'on the other words, Enhanced AI (Claude) minus Base AI (Claude), agreement with the manual annotations (points)'),
    ('store_start.other.rows', '2,594', 'evaluated words without such an entry'),
    ('store_start.other.claude_enhanced', '83.31', 'on the other words, Enhanced AI (Claude), agreement with the manual annotations (%)'),
    ('store_start.other.claude_base', '82.69', 'on the other words, Base AI (Claude), agreement with the manual annotations (%)'),
    ('store_start.other.lead_words', '16', 'on the other words, Enhanced AI (Claude) minus Base AI (Claude), words matching the manual annotations'),
])
cite('Configuration record 7', [
    ('mcnemar.cs.manual.p', '0.029', 'Enhanced AI (Claude) vs Base AI (Claude), case-sensitive exact McNemar test, manual annotations, p'),
    ('mcnemar.cs.expert.p', '0.015', 'Enhanced AI (Claude) vs Base AI (Claude), case-sensitive exact McNemar test, secondary reference set, p'),
    ('mcnemar.cs.manual.enhanced_only', '53', 'word-level entries matching the manual annotations for Enhanced AI (Claude) but not Base AI (Claude), case-sensitive'),
    ('mcnemar.cs.manual.base_only', '32', 'word-level entries matching the manual annotations for Base AI (Claude) but not Enhanced AI (Claude), case-sensitive'),
])
for o, label, mean, sd, gap in [('enhanced', 'Original', '89.49', '1.16', '+1.34'),
                                ('order2', 'Alternative 1', '88.46', '0.41', '+0.31'),
                                ('order3', 'Alternative 2', '88.82', '0.09', '+0.66'),
                                ('order4', 'Alternative 3', '89.24', '0.54', '+1.09')]:
    cite('Configuration record 7', [
        (f'replication.{o}.mean', mean, f'{label} order, Enhanced AI (all components), mean agreement with the manual annotations (%)'),
        (f'replication.{o}.sd', sd, f'{label} order, Enhanced AI (all components), run-to-run SD of agreement with the manual annotations (percentage points)'),
        (f'replication.order_gap.{o}', gap, f'{label} order, Enhanced AI (Claude Opus 5) mean minus Base AI (Claude Opus 5) original-order mean, agreement with the manual annotations (points)')])
cite('Configuration record 7', [
    ('corpus.rows', '3,216', 'evaluated word-level entries'),
    ('replication.base.mean', '88.15', 'Base AI (Claude Opus 5), original order, mean agreement with the manual annotations (%)'),
    ('replication.order_span', '1.03', 'Enhanced AI (Claude Opus 5), range of mean agreement with the manual annotations across four processing orders (points)'),
])
cite('Configuration record 7', [
    ('sensitivity.enhanced_vs_base.points', '+1.60', 'without the model-selection songs: Enhanced AI (Claude Opus 5) minus Base AI (Claude Opus 5) (points)'),
    ('sensitivity.enhanced_vs_base.p', '0.007', 'without the model-selection songs: Enhanced AI (Claude Opus 5) minus Base AI (Claude Opus 5), p'),
    ('sensitivity.enhanced_vs_no_dictionary.points', '−0.37', 'without the model-selection songs: dictionary contribution (points)'),
    ('sensitivity.enhanced_vs_no_dictionary.p', '0.876', 'without the model-selection songs: dictionary contribution, p'),
])
cite('Configuration record 7', [
    ('neural.identity', '54.85', 'identity control, agreement with the manual annotations (%)'),
    ('neural.lowercased', '55.63', 'lowercased control, agreement with the manual annotations (%)'),
])
cite('Configuration record 7', [
    ('neural.lexicon_vs_estnltk_dictionary.songs_b', 'twenty-eight', 'songs where agreement with the manual annotations is lower for Lexicon-enhanced lemmatizer (Vabamorf + index) than for EstNLTK + Dictionary'),
    ('neural.lexicon_vs_estnltk_dictionary.songs_a', 'eleven', 'songs where agreement with the manual annotations is higher for Lexicon-enhanced lemmatizer (Vabamorf + index) than for EstNLTK + Dictionary'),
    ('neural.lexicon_vs_estnltk_dictionary.songs_tied', 'six', 'songs where agreement with the manual annotations is equal for Lexicon-enhanced lemmatizer (Vabamorf + index) and EstNLTK + Dictionary'),
])


# Compare each recomputed value with the printed value at the printed
# precision, and write the results.
def agrees(value, printed):
    if printed == 'yes':
        return value is True
    s = str(WORDS.get(printed, printed)).replace(',', '').replace('−', '-').replace('+', '')
    if s.startswith('<'):
        return value < float(s[1:])
    decimals = len(s.split('.')[1]) if '.' in s else 0
    return f'{value + 0.0:.{decimals}f}' == s


def shown(value, printed):
    if isinstance(value, bool):
        return str(value)
    s = str(WORDS.get(printed, printed)).replace(',', '').lstrip('<+−-')
    decimals = len(s.split('.')[1]) if '.' in s else 0
    return f'{value + 0.0:.{max(decimals, 4) if printed.startswith("<") else decimals}f}'


def main():
    unknown = sorted({q for _, _, q, _ in ARTICLE} - set(V))
    if unknown:
        sys.exit(f'no computed value for: {", ".join(unknown)}')
    differ = 0
    with open('recomputed_results.csv', 'w', newline='', encoding='utf-8') as f:
        out = csv.writer(f)
        out.writerow(['key', 'value', 'article', 'agrees', 'location', 'description'])
        for location, description, quantity, printed in ARTICLE:
            value = V[quantity]
            if isinstance(value, np.bool_):
                value = bool(value)
            ok = agrees(value, printed)
            differ += not ok
            out.writerow([quantity, repr(value if isinstance(value, bool) else float(value)),
                          printed, ok, location, description])
            print(f'{location} | {description}: {shown(value, printed)} '
                  f'(article {printed}) {"agrees" if ok else "DIFFERS"}')
    print(f'{len(ARTICLE)} figures compared, {differ} differ')
    return 1 if differ else 0


if __name__ == '__main__':
    sys.exit(main())
