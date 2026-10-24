# Estonian Runosong Lemmatization

Code, prompts and data for the article "AI-assisted lemmatization of Estonian dialectal runosongs: Language models and linguistic resources for cultural heritage texts" by Kaarel Veskis and Taive Särg. The article's citation and DOI will be added on publication.

The article evaluates large language models for lemmatizing Estonian dialectal runosongs, with and without linguistic resources, and compares them with rule-based and neural tools. Its pipeline has two versions: in Base AI the language model works from the song text, its own English translation and the archival metadata, while Enhanced AI also draws on morphological analysis from EstNLTK, lookups in a lemma–variant index compiled from seven Estonian dictionaries and corpora, and context accumulated from previously analysed songs. The original evaluation ran both versions with Claude 3.5 Sonnet and with DeepSeek-V3.1, and a replication with Claude Opus 5 repeated them and removed Enhanced AI's components one at a time. All systems are scored on 3,216 words from forty-five songs against their original manual annotations, and on 3,139 of these words against a secondary reference set compiled by a runosong expert.

## Contents

- `configuration_record.md`: the settings, runs and run dates of the language-model experiments, the comparisons specified before the runs, implementation details, further results, song lists with archive references, and a guide to the columns of the two data files.
- `code/`: the pipeline as run in July 2025 (`enhanced_comb_3_v8.py` and its three helper files), the September 2025 version of the main program (`enhanced_comb_3_v8_september_2025.py`), and the pipeline's requirements (`code/requirements.txt`: EstNLTK 1.7.4, and the Anthropic library below version 1.0, as version 1.0 removed the temperature setting the pipeline uses).
- `prompts/`: the July 2025 analysis prompts of Base AI and Enhanced AI, and the September 2025 versions of both prompts and of the lemma-selection guideline (configuration record, section 3).
- `data/evaluation_results.csv`: the 3,216 evaluated words with their verse line, song, collection area, dialect group and subcategory; both reference sets; the lemmas of Base AI and Enhanced AI from the original evaluation, with Claude 3.5 Sonnet and with DeepSeek-V3.1, and of two rule-based and lexical baselines, EstNLTK + Dictionary and Lexical resources; the annotators' and the four AI systems' English translations; and the expert's decision codes (column guide in the configuration record, section 9).
- `data/per_word_predictions.csv`: the lemmas from the later experiments: the replication's twenty-seven evaluation runs with Claude Opus 5, three runs with the September 2025 prompts, the four zero-shot neural systems and the controls of the normalization test, and two further baselines, Pure EstNLTK and a frequency-ranked version of Lexical resources (column guide in the configuration record, section 9).
- `data/corpus_database.json`: the corpus store of the original evaluation's July 2025 runs with Claude 3.5 Sonnet (configuration record, section 6).
- `analysis/recompute_results.py`: recomputes the reported results; its function `lemmas_match` is the matching rule behind every agreement figure.

## Recompute the reported results

```
pip install -r requirements.txt
python analysis/recompute_results.py
```

The script reads the files in `data/`, needs only NumPy and SciPy, and runs in under a minute. It was tested with Python 3.10 at the versions pinned in `requirements.txt`, and with NumPy 2.5.4 and SciPy 1.18.1 on Python 3.12. For each figure it prints where the figure appears, the recomputed value, the value given there, and whether the two agree, and it writes the same list to `recomputed_results.csv`.

## Two program versions

`code/enhanced_comb_3_v8.py` is the main program as run in July 2025. `code/enhanced_comb_3_v8_september_2025.py` is the main program of version 1.0.0 of this repository (DOI 10.5281/zenodo.18089185); it differs from the July program only in three prompts, revised in late September 2025 together with the lemma-selection guideline (configuration record, section 3). Both use the same three helper files, here in their July 2025 versions; in version 1.0.0, one of them, `estonian_comb_3.py`, also differs from the July one in one short passage of each of its two translation prompts.

Both programs call Claude 3.5 Sonnet, which has since been retired, so running them again needs another model in its place; Enhanced AI as evaluated also needs the lemma–variant index, which is not included.

## Sources

The songs' verse lines and metadata come from the Database of Estonian Runosongs (ERAB) of the Estonian Folklore Archives, cited below in the form the archive asks for. The configuration record (section 8) gives the archive reference of every song whose verse lines this repository contains. The manual lemmas, which form the primary reference set, and the annotators' English translations come from the benchmark of Pivovarova et al. (2025), whose data are at https://github.com/hsci-r/filter-llm-lemmatization.

- Eesti regilaulude andmebaas. Eesti Kirjandusmuuseumi Eesti Rahvaluule Arhiiv. Tartu, 2003–2026. https://www.folklore.ee/regilaul/andmebaas, https://doi.org/10.15155/9-00-0000-0000-0000-0008FL
- Pivovarova, L., Kallio, K., Kanner, A., Lindström, J., Mäkelä, E., Saarlo, L., Veskis, K. and Väina, M. (2025). Benchmarking Large Language Models for Lemmatization and Translation of Finnic Runosongs. In Proceedings of the 10th International Workshop on Computational Linguistics for Uralic Languages (IWCLUL 2025), pp. 87–105. Joensuu, Finland: Association for Computational Linguistics. https://aclanthology.org/2025.iwclul-1.12

## Licence and citation

CC BY 4.0 (see `LICENSE`), except for the material from the two sources above: the songs' verse lines and metadata, and the manual lemmas and the annotators' English translations. That material remains under its sources' terms; cite the sources above when reusing it. Please cite this repository as described in `CITATION.cff`.
