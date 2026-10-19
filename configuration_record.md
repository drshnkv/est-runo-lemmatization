# Configuration record

This record accompanies the article "AI-assisted lemmatization of Estonian dialectal runosongs: Language models and linguistic resources for cultural heritage texts". It gives the settings, runs and run dates of the article's language-model experiments, the comparisons specified before the runs, implementation details, further results, song lists with the songs' archive references, and a guide to the columns of the two data files.

## 1. Original evaluation (July and September 2025)

The original evaluation used Claude 3.5 Sonnet (model ID `claude-3-5-sonnet-20241022`) through the Anthropic API. Every model call the pipeline makes sets `temperature=0` and `max_tokens=4000` and leaves the other parameters at the API defaults. The model uses no tools or retrieval beyond the pipeline components described in the article's Section 3. EstNLTK was used in version 1.7.4 throughout.

The DeepSeek-V3.1 runs of Base AI and Enhanced AI processed the forty-five songs on 16 September 2025. They used DeepSeek's Anthropic-compatible API (`https://api.deepseek.com/anthropic`) with the model name `deepseek-chat`, which from 21 August to 22 September 2025 stood for DeepSeek-V3.1 in non-thinking mode (DeepSeek API change log, https://api-docs.deepseek.com/updates). The runs used the July 2025 pipeline with only the model name and the service address changed, and the July 2025 analysis prompts word for word.

## 2. Replication and ablation experiments (August 2026)

Claude 3.5 Sonnet was retired from the API on 28 October 2025, so the replication used `claude-opus-5` through Anthropic's Claude Code command-line interface (versions 2.1.247 to 2.1.251, run non-interactively) on 27–28 August 2026. The July 2025 program was run by a separate script, not released, which imported it unchanged, sent its model calls through this interface instead of the Anthropic API and set up each condition. The model received the program's prompts exactly as the program sends them to the API. The interface ran with its tools and settings switched off and in an empty folder; in a test call at the start of each run, the model reported no file names or folder paths in its input. The model identifier returned with each response confirmed the serving model.

There were twenty-eight runs: twenty-seven evaluation runs and one that built a corpus store from forty-eight songs outside the evaluation set, which the frozen-store condition then used unchanged. The three runs in the original processing order (section 7) produced Base AI and Enhanced AI outputs; the other evaluation runs produced Enhanced AI outputs only, as none of their conditions changes what Base AI receives. The model does not accept settings for temperature, top-p or top-k, so each condition was run three times over the same forty-five songs, and the spread between the three runs shows how much the results vary from run to run. The English translations given to the model were produced once and shared unchanged by all conditions. The predictions of the evaluation runs are in `data/per_word_predictions.csv` (section 9); the songs that built the store are listed in section 8.

## 3. Prompt-version comparison (September 2026)

The lemma-selection guideline was revised in late September 2025, after the original evaluation, to bring it closer to the annotation guidelines of Pivovarova et al. (2025), which the manual annotations also follow. The revised guideline always asks for a form of the same word, never for a different word of similar meaning, and, where Standard Estonian has no counterpart, for an Estonian dialect form rather than a Finnish or other Finnic one; it also adds a rule for ambiguous pronoun forms (the long form, such as mina). The guideline is part of three prompts, which were revised with it: the Base AI and Enhanced AI analysis prompts, which include all of it, and the prompt of the candidate-filtering step (step 7 in the article's Figure 1), which includes it up to the end of its section 9. Version 1.0.0 of this repository (December 2025) holds the revised versions; this comparison tests whether the revised prompts change agreement. Three runs on 17–18 September 2026 used `claude-opus-5` through the replication's command-line interface (version 2.1.247, run non-interactively), each producing Base AI outputs and Enhanced AI outputs, the latter with the frozen corpus store. As the September 2025 program (`code/enhanced_comb_3_v8_september_2025.py`) differs from the July 2025 program only in these three prompts, the runs used the July 2025 program, as in August 2026, with the three prompts taken from the September 2025 program. They were compared with the August 2026 runs that used the July 2025 prompts: the three original-order runs for Base AI and the three frozen-store runs for Enhanced AI.

The revised prompts did not measurably change agreement with either reference set: the differences were at most 0.06 points against the manual annotations (the primary reference set) and at most 0.23 points against the secondary reference set, and the 95 per cent intervals of all four differences (Base AI and Enhanced AI, against each reference set) included zero.

## 4. Check on fourteen unpublished songs (October 2026)

This check (the article's Section 6.2) runs the replication's setups on fourteen unpublished songs held in the Estonian Folklore Archives and compares agreement there with agreement on the forty-five public songs.

The second author (Taive Särg) found the fourteen songs, from several dialect areas, among the archive's transcriptions of sound recordings and lemmatized every word following the annotation guidelines of Pivovarova et al. (2025), which the manual annotations of the forty-five songs also follow. Of the 1,221 words, 1,209 are scored; she marked the other twelve as uncertain or to be left out. One of the songs is a Seto song, with 294 of the scored words. Her lemmas were finalized on 12 October 2026, before any song was sent to a language model.

Six runs on 12 October 2026 used `claude-opus-5` through the replication's command-line interface (version 2.1.247), with the replication's pipeline and July 2025 prompts, the August 2026 index build (section 6) and the checks described in section 2. The songs were given to the model in the same form as the forty-five and with the same metadata fields (section 6), except that they had no ERAB identifier for the collection area and carried the check's own song identifiers. Their English translations were made once and shared by all six runs. Three runs produced Base AI and Enhanced AI outputs, giving Enhanced AI an empty corpus store that is never updated, so that it is shown no earlier analyses; the other three produced Enhanced AI outputs with the replication's frozen store, unchanged. Each setup's three runs take the songs in three different orders.

Each run is scored with the matching rule used for every agreement figure in the article (`lemmas_match` in `analysis/recompute_results.py`), an empty answer counting as wrong. Each setup is compared with the replication condition that gives the model the same context on the forty-five songs. The table gives three-run means in per cent agreement, and differences in percentage points computed from unrounded values:

| Setup (matching condition on the forty-five songs) | 14 songs (1,209 words) | 45 songs (3,216 words) | Difference |
|---|---|---|---|
| Base AI (original-order runs) | 84.20 | 88.15 | −3.95 |
| Enhanced AI, frozen store (frozen-store condition) | 85.99 | 89.80 | −3.81 |
| Enhanced AI, empty store (no-corpus-store condition) | 86.57 | 89.32 | −2.75 |

The run-to-run standard deviations on the fourteen songs are 0.74, 0.55 and 0.70 points, in the order of the table's rows. A regional adjustment allows for the different regional make-up of the two sets: each dialect subcategory's agreement on the forty-five songs is weighted by its share of the fourteen songs' scored words. With it, the differences are −3.81, −3.64 and −2.38 points.

To check whether the fourteen songs are harder to lemmatize than the forty-five, EstNLTK + Dictionary, which uses no language model, was run with the August 2026 index build on the words in the form the models received. It agrees with the annotations on 61.70 per cent of the fourteen songs' scored words, against 59.51 per cent of the forty-five songs' words: a difference of +2.19 points, or −0.53 after the regional adjustment (descriptive 95 per cent interval −6.73 to 4.33). On the Seto song it agrees on 32.65 per cent of the words, against 30.25 per cent on the seven Seto songs among the forty-five. The 60.6 per cent for EstNLTK + Dictionary in the article's Table 1 is from the July 2025 run, which used the earlier index build.

## 5. Comparisons specified before the runs

Replication experiments: the main comparison tests Enhanced AI with all components against Base AI, on agreement with the manual annotations over all 3,216 words. Seven further comparisons are adjusted together with the Holm method: frozen store against Base AI; frozen store against no corpus store; all components against the frozen store; and all components against each of the four conditions that remove one component (no dictionary; no EstNLTK; no word-level context, in which nothing from EstNLTK or dictionary lookup reaches the model; no corpus store). These eight comparisons were specified on 27 August 2026, before the first evaluation run. The processing-order comparisons are descriptive.

Model-selection sensitivity analysis: the main comparisons repeated without the seven model-selection songs (section 8), specified on 27 August 2026, before the first evaluation run.

Lemmatizer comparison: the lexicon-enhanced neural lemmatizer of Milintsevich and Sirts (2020), with the lemma–variant index, against EstNLTK + Dictionary over all 3,216 words, with a song-level sign-flip test and a bootstrap interval for the difference, specified on 30 August 2026, before the lemmatizer was trained.

Fourteen-song check: Base AI, Enhanced AI with an empty corpus store, and Enhanced AI with the frozen store, each compared with its matching condition on the forty-five songs; specified on 3 September 2026, before the songs were annotated or sent to any model. The regional adjustment and the EstNLTK + Dictionary comparison were added to the check on 6 October 2026, before the runs.

## 6. Implementation details

### Index builds

The original evaluation used the July 2025 build of the lemma–variant index: 29,328 lemmas and 166,360 distinct variant spellings (sha256 `a743eeece6611ba03c4e32faa0a60807e8b5644a64af79fe24516027b1cbf6fd`). The replication experiments and the fourteen-song check used the August 2026 build: 33,584 lemmas and 175,493 distinct variant spellings (sha256 `b7194060d57c603a950c0ac14d6655f0a332486a79be97965ecae494318d10bd`). The growth comes from a Seto dictionary integrated in the first half of 2026, which added new lemmas and enlarged 3,776 existing entries with Seto variant forms. Neither build is released.

### Corpus store at the start

In the original evaluation's July 2025 runs with Claude 3.5 Sonnet, the corpus store was not empty when the first evaluated song was processed. It already held about 600 word forms from seven earlier texts: a test sentence and six songs that are not among the forty-five (entries 1 to 7 of the list `processed_texts` in `data/corpus_database.json`). Two of the six songs were processed again on 13 and 14 July, which replaced their earlier entries; the seven texts still account for about 600 word forms in the released file. These entries show no sign of having favoured Enhanced AI, the only system that reads the store: 622 of the 3,216 evaluated words have a word form with an entry from the seven earlier texts (forms compared in lower case), and on these words Enhanced AI's agreement with the manual annotations exceeds Base AI's by 0.48 percentage points (94.21 against 93.73 per cent, a difference of 3 words), against 0.62 points on the other 2,594 words (83.31 against 82.69, a difference of 16 words). In the DeepSeek-V3.1 runs and in the replication's runs with a growing store, the store started empty.

### Coverage

The coverage figures in the article's Section 3 rest on the 451,371 distinct word forms of the runosong corpus: direct lookup covers 85,863 of them (19.02 per cent); h-variation matching and suffix stripping add 1,946 and 6,209, so 8,155 forms (1.81 percentage points) are covered only through these transformations, and 94,018 in total (20.83 per cent). The figures leave out two further steps that the lookup tries during lemmatization: matching spellings up to two letter changes away (Levenshtein distance) and splitting compound words.

### Resampling and tests

Percentile cluster-bootstrap intervals use 400,000 replicates for the main comparisons and 10,000 for the regional analyses of the original evaluation, with random seed 20261019. The song-level sign-flip test uses 100,000 permutations. For the replication and the model-selection sensitivity analysis, the test is computed on each song's difference in matching words summed over the three runs, so that ties with the observed statistic are counted exactly. This way of computing the test was adopted on 24 September 2026, after the replication's runs were complete; the comparisons, their Holm adjustment and all other settings are as specified before the first run. The original evaluation's three comparisons are Holm-adjusted together: the gain of Enhanced AI over Base AI with Claude 3.5 Sonnet, the same gain with DeepSeek-V3.1, and the difference between the two gains.

### Metadata

The fields supplied to the model from ERAB are the collection area's name and ERAB identifier, the year, and the record's display name.

## 7. Further results

### Word-level tests (the article's Section 5.1)

In the article's exact McNemar test, which compares Enhanced AI and Base AI with Claude 3.5 Sonnet against the primary reference set, fifty-one words match the manual annotations in Enhanced AI's output but not in Base AI's, and thirty-two the other way round. With case-sensitive matching, the test gives p = 0.029 against the primary reference set (53 and 32 such words) and p = 0.015 against the secondary one.

### Processing orders (the article's Section 5.2)

Enhanced AI was run three times under each of four processing orders with the growing corpus store, scored against the manual annotations for 3,216 words. The original order is the order of the forty-five songs in the July 2025 corpus store (entries 8 to 52 of the list `processed_texts` in `data/corpus_database.json`); the other three shuffle it with Python's `random.Random`, seeds 101, 102 and 103. The Original row summarizes the three runs of the all-components condition. Base AI ran under the original order only; each gap is the order's mean minus Base AI's 88.15, from unrounded values. These comparisons are descriptive.

| Processing order | Mean agreement % (3 runs) | Run-to-run SD | Gap to Base AI (points) |
|---|---|---|---|
| Original | 89.49 | 1.16 | +1.34 |
| Alternative 1 | 88.46 | 0.41 | +0.31 |
| Alternative 2 | 88.82 | 0.09 | +0.66 |
| Alternative 3 | 89.24 | 0.54 | +1.09 |

Mean agreement spans 1.03 percentage points across the four orders, comparable to the variation between repeated runs.

### Model-selection sensitivity

Leaving out the seven evaluated songs that had been used to choose the model for the replication (section 8) raises the lead of Enhanced AI with all components over Base AI from +1.34 to +1.60 points (p = 0.007, descriptive) and brings the dictionary's estimated contribution closer to zero, from −0.64 to −0.37 points (p = 0.876, descriptive).

### Normalization controls (the article's Section 5.5)

The identity control (EstNLTK applied to the verse lines without the rewriting, capitals kept) reaches 54.85 per cent agreement with the manual annotations, and the lowercased control 55.63 per cent. Restoring the missing letters õ, ä, ö and ü in the normalization models' output gives 56.41 per cent, 0.78 points above the lowercased control (95 per cent interval −3.37 to 4.99).

### Retraining check (the article's Section 5.5)

Retrained from its authors' released implementation in their configuration, the lexicon-enhanced lemmatizer reached 97.39 per cent accuracy on material from its training domain, within one percentage point of the 96.64 per cent documented in its authors' repository.

### Song-level tally (the article's Section 5.5)

With the lemma–variant index, the lexicon-enhanced lemmatizer scores below EstNLTK + Dictionary in twenty-eight of the forty-five songs, above it in eleven, and ties in six.

## 8. Song lists

Evaluated songs: the forty-five songs, in the order of `data/evaluation_results.csv`, with their archive references in the Estonian Folklore Archives as ERAB gives them.

| `poem_id` | Archive reference |
|---|---|
| ars0104990010 | ARS 1, 499/500 (10) |
| ekms40100930139 | EKmS 4° 1, 93 (139) |
| eks0500110007 | EKS 5, 11/2 (7) |
| era200703520005 | ERA II 7, 352/3 (5) |
| era200901010004 | ERA II 9, 101 (4) |
| era201003030001 | ERA II 10, 303 (1) |
| era201702520173 | ERA II 17, 252/3 (173) |
| era203000970003 | ERA II 30, 97 (3) |
| era203203650010 | ERA II 32, 365/7 (10) |
| era203601950002 | ERA II 36, 195/6 (2) |
| era207003330001 | ERA II 70, 333 (1) |
| era207206830022 | ERA II 72, 683 (22) |
| era207207010009 | ERA II 72, 701 (9) |
| era207208550005 | ERA II 72, 855 (5) |
| era207209190010 | ERA II 72, 919 (10) |
| era207801520003 | ERA II 78, 152 (3) |
| era208003410240 | ERA II 80, 341/2 (240) |
| era210102140008 | ERA II 101, 214/5 (8) |
| era212300820083 | ERA II 123, 82/3 (83) |
| era212601970012 | ERA II 126, 197/8 (12) |
| era214700180008 | ERA II 147, 18/20 (8) |
| era215104990004 | ERA II 151, 499/501 (4) |
| era216102270022 | ERA II 161, 227/9 (22) |
| era216600650068 | ERA II 166, 65 (68) |
| era217201600004 | ERA II 172, 160 (4) |
| era219001990005 | ERA II 190, 199/201 (5) |
| era21900204008a | ERA II 190, 204 (8a) |
| era22480349000b | ERA II 248, 349 (b) |
| era225300200009 | ERA II 253, 20 (9) |
| estk0200970115 | E, StK 2, 97 (115) |
| estk0300530005 | E, StK 3, 53 (5) |
| estk0500110002 | E, STK 5, 11 (2) |
| estk0501860031 | E, StK 25, 186/9 (31) |
| estk1501170082 | E, StK 15, 117 (82) |
| estk2301160001 | E, StK 23, 116 (1) |
| estk4000730109 | E, StK 40, 73/4 (109) |
| e0000000X0078 | E X 17 (78) |
| e000066940001 | E 6694/6 (1) |
| e000130710023 | E 13071/3 (23) |
| e000156560142 | E 15656 (142) |
| e000203010003 | E 20301/2 (3) |
| e000203150009 | E 20315 (9) |
| e000249000026 | E 24900 (26) |
| e000252410001 | E 25241/45 (1) |
| e000401360005 | E 40136 (5) |

Earlier songs in the corpus store: the six songs among the seven earlier texts of the July 2025 corpus store (section 6), entries 2 to 7 of the list `processed_texts` in `data/corpus_database.json`, whose verse lines appear in the store's word contexts.

| `poem_id` | Archive reference |
|---|---|
| h20603730038 | H II 6, 373/8 (38) |
| h23200320022 | H II 32, 32/3 (22) |
| h22710080003 | H II 27, 1008 (3) |
| h20400530015 | H II 4, 53/4 (15) |
| era207904830005 | ERA II 79, 483/7 (5) |
| h23702970009 | H II 37, 297/8 (9) |

Store-building songs: forty-eight further Estonian songs from the same Finnic test corpus as the evaluated songs, none of them among the forty-five, from which one run built the corpus store of the frozen-store condition. Their texts are in the benchmark repository of Pivovarova et al. (2025; GitHub `hsci-r/filter-llm-lemmatization`, identifiers in `annotations/ids.txt`). Identifiers in processing order:

```
e000429160000 e000513390039 e000522050000 e000594650000 e000685380005
e000711700006 e000854260002 eys0413630084 eys0415420081 h20604900130
h20708050014 h21003870114 h21706930081 h22403540121 h22500950062
h22511320026 h23000660003 h23000680012 h23105460002 h23200870077
h23601580203 h23602660412 h23702730007 h24006680016 h24006910037
h24007000016 h24304830023 h24406010004 h24406170058 h24504220003
h24603060011 h25805820004 h25807780009 h26601600063 h26804900006
h30201200004 h30701130078 h30803540002 h31104200015 h31307630018
h31502160003 h31605540003 h32204160006 h32505810010 h33003270000
h40100970040 h40401390001 ve0604130059
```

Model-selection songs: seven of the forty-five evaluated songs, listed below by their `poem_id`, used to choose the model for the replication. On 26 August 2026, four models (`claude-opus-5`, `claude-fable-5`, `claude-sonnet-5` and `claude-haiku-4-5-20251001`) lemmatized their 618 words with a plain prompt and no pipeline context, at default sampling settings through the same command-line interface. `claude-opus-5` and `claude-fable-5` showed indistinguishable agreement with the manual annotations, slightly above `claude-sonnet-5` and far above `claude-haiku-4-5-20251001`, and `claude-opus-5` was chosen as the less expensive.

```
ars0104990010
era203203650010
era203601950002
era210102140008
era216102270022
era219001990005
estk1501170082
```

## 9. Column guide

Both data files are UTF-8 comma-separated files with one row per evaluated word (3,216 rows), in the same order.

`data/evaluation_results.csv` holds the words, both reference sets and the outputs of the original evaluation:

- `poem_id`: the song's ERAB identifier (section 8 gives each song's archive reference); `place_name`: the collection area (`Setumaa` is spelled Setomaa in the article); `verse`: the verse line in which the word occurs; `original_word`: the word form, in lower case and without punctuation.
- `manual_lemma_standard`: the manual annotations of Pivovarova et al. (2025), the primary reference set. `proper_gold_standard`: the secondary reference set; where it accepts several lemmas, they are separated by commas, and it is empty for the seventy-seven words left out.
- `claude_standard_lemma_modern`, `claude_enhanced_lemma_modern`, `deepseek_standard_lemma_modern` and `deepseek_enhanced_lemma_modern`: the lemmas of Base AI (`standard`) and Enhanced AI (`enhanced`) with Claude 3.5 Sonnet and with DeepSeek-V3.1; `estnltk_lemma`: EstNLTK + Dictionary; `lexical_levenshtein_lemma`: Lexical resources, empty where it found no lemma.
- `manual_english_translation`: the annotators' English translation of the word; `claude_standard_english_translation`, `claude_translation_enhanced`, `deepseek_standard_english_translation` and `deepseek_enhanced_english_translation`: the translations of Base AI and Enhanced AI with each model.
- `dialect_primary_merged`: the dialect group (South Estonian, Western Estonian or North+East Estonian, which the article calls North+East+Mid Estonian); `dialect_subcategory_merged`: one of fifteen dialect subcategories, such as Seto.
- `expert_comparison`: the expert's decision codes, as in the article's Table A1. `M`, `LE` and `LS` mean that the expert preferred the manual annotation, the lemma of Enhanced AI or the lemma of Base AI (both with Claude 3.5 Sonnet); a combined entry such as `M, LE` means that the expert accepted the lemmas of each method named, and the one entry written `M, EL` stands for `M, LE`. `0` and `-` mark the seventy-seven words left out of the secondary reference set, `0` being Table 3's "no method matched" cases. An empty code marks an entry accepted automatically, outside the expert review: a Claude lemma matched the manual lemma, which became the reference lemma. The one entry marked `?` also keeps the manual lemma.

Scoring compares the six lemma columns of this file with the two reference sets. The systems added later are in `data/per_word_predictions.csv`:

- `row` (0-based row number in the evaluation file), `poem_id`, `original_word`: the keys that join the two files.
- `pure_estnltk`: Pure EstNLTK; `lexical_frequency_ranked`: the frequency-ranked lexical variant reported in the article's Section 5.1.
- `enhanced_<condition>_run1` to `_run3`: the twenty-seven evaluation runs of the replication, three for each of nine conditions: `original_order`, `order2`, `order3`, `order4` (the original and three alternative processing orders), `frozen_store`, `no_dictionary`, `no_estnltk`, `no_word_context` and `no_corpus_store`.
- `base_original_order_run1` to `_run3`: Base AI, from the three original-order runs.
- `september_prompts_enhanced_run1` to `_run3` and `september_prompts_base_run1` to `_run3`: the prompt-version comparison (section 3), compared with `enhanced_frozen_store_run1` to `_run3` and `base_original_order_run1` to `_run3`.
- `normalized_estnltk`, `unnormalized_estnltk`, `lowercased_estnltk`, `normalized_repaired_estnltk`: the normalization test in the article's Section 5.5 (normalized forms, the identity control, the lowercased control, and the normalized forms with õ, ä, ö and ü restored, each lemmatized by EstNLTK).
- `lexicon_lemmatizer_dialect` and `lexicon_lemmatizer_vabamorf`: the lexicon-enhanced lemmatizer with candidates from Vabamorf and the lemma–variant index, and from Vabamorf only; `glilem`: GliLem; `stanza`: Stanza. These four columns are stored as the systems output them; the recompute script removes the compound-boundary marks `_` and `=` before scoring.
