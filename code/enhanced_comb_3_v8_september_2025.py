#!/usr/bin/env python3
"""
Enhanced comb_3.py v8 -  Estonian Runosong Analysis
See README
Pipeline: EstNLTK 1.7.4 Context → Cached Pre-Analysis (Claude API) → Contextual Corpus Guidance → Enhanced Analysis
"""

import anthropic
import pandas as pd
from tqdm import tqdm
import time
from pathlib import Path
import logging
import sys
import csv
import os
from datetime import datetime
import re
from io import StringIO
import glob
import json
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Import the original components
sys.path.append(str(Path(__file__).parent))
from runosong_lemmatizer_with_estnltk_174_v8 import RunosongLemmatizerWithEstNLTK174
from config_manager import ConfigManager
from estonian_comb_3 import EstonianPromptTester

class EnhancedEstonianPromptTesterV8(EstonianPromptTester):
    """
    Enhanced version v8 of EstonianPromptTester with morphology-first approach.
    Fixes v7's compound over-detection while maintaining EstNLTK 1.7.4 benefits and v6's prompt caching.
    """
    
    def __init__(self, api_key=None, annotation_dir=None, corpus_file=None, force_corpus_update=False):
        """
        Initialize the enhanced prompt tester v8.
        
        Args:
            api_key: Anthropic API key
            annotation_dir: Directory containing EstNLTK annotation files
            corpus_file: Path to corpus database JSON file
            force_corpus_update: Force corpus updates even for already processed texts
        """
        # Initialize the base class
        super().__init__(api_key)
        
        # Initialize EstNLTK 1.7.4 analyzer for context generation
        try:
            self.estnltk_analyzer = RunosongLemmatizerWithEstNLTK174(annotation_dir or ".")
            self.context_enabled = True
            self.logger.info("EstNLTK 1.7.4 analyzer loaded for context generation")
        except Exception as e:
            self.estnltk_analyzer = None
            self.context_enabled = False
            self.logger.warning(f"EstNLTK 1.7.4 analyzer not available: {e}")
        
        # Analysis prompts (standard)
        self.analysis_prompts = {
            "tabular_analysis": self._get_tabular_analysis_prompt()
        }
        
        # Enhanced analysis prompts with context
        self.enhanced_analysis_prompts = {
            "tabular_analysis_with_context": self._get_enhanced_tabular_analysis_prompt()
        }
        
        # Pre-analysis prompt for EstNLTK context evaluation
        self.pre_analysis_prompt = self._get_pre_analysis_prompt()
        
        # Initialize corpus learning system
        self.corpus_file = corpus_file or "corpus_database.json"
        self.force_corpus_update = force_corpus_update
        self.corpus_database = self._load_corpus_database()
        
        # Ensure corpus database has proper structure
        if not isinstance(self.corpus_database, dict):
            self.corpus_database = {}
        if "words" not in self.corpus_database:
            self.corpus_database["words"] = {}
        if "processed_texts" not in self.corpus_database:
            self.corpus_database["processed_texts"] = []
        if "metadata" not in self.corpus_database:
            self.corpus_database["metadata"] = {
                "created": datetime.now().strftime("%Y-%m-%d"),
                "version": "v8"
            }
        
        # Processing statistics
        self.enhancement_stats = {
            "texts_with_context": 0,
            "texts_without_context": 0,
            "context_generation_time": 0,
            "pre_analysis_time": 0,
            "enhanced_processing_time": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "pre_analysis_calls": 0,
            "pre_analysis_cache_hits": 0,
            "pre_analysis_cache_misses": 0
        }
        
        # Thread lock for statistics
        self._stats_lock = threading.Lock()
        
        # Cache for pre-analysis results
        self._pre_analysis_cache = {}
    
    def _get_pre_analysis_prompt(self):
        """Return the pre-analysis prompt for EstNLTK context evaluation."""
        return """
You are an expert Estonian linguist evaluating EstNLTK+dictionary analysis results for Estonian dialectal runosong texts.

Your task is to analyze the provided EstNLTK suggestions and filter them for quality and relevance to improve the context provided to the main analysis.

# INPUT FORMAT

You will receive EstNLTK analysis results in this format:
- Original word
- Suggested lemma
- Confidence score
- Analysis method (estnltk, estnltk+dict, dict, levenshtein)
- Morphological information (if available)
- Alternative suggestions

# EVALUATION CRITERIA

## Relevance Assessment
- **Dialectal compatibility**: Does the suggestion make sense for Estonian dialects (especially South Estonian)?
- **Morphological consistency**: Are the POS tags and morphological forms correct?
- **Lemma quality**: Is the suggested lemma a valid Estonian dictionary form?
- **Alternative value**: Are the alternative suggestions helpful or misleading?

## Quality Indicators
- **High confidence validation**: Are 1.00 confidence scores actually reliable?
- **Method reliability**: How trustworthy is the analysis method for this word?
- **Contextual appropriateness**: Does the suggestion fit the runosong context?
- **Estonian standards**: Does the lemma follow Estonian linguistic conventions (EKSS)?

## Dialectal Considerations
- **Võro/Seto features**: Recognition of dialectal phonological patterns
- **Dialectal lemmatization**: Proper mapping to standard Estonian forms
- **Archaic forms**: Handling of historical Estonian vocabulary
- **Morphological variants**: South Estonian case endings and verb forms

# OUTPUT FORMAT

For each word, provide:

```json
{
  "original_word": "word",
  "evaluation": {
    "keep_suggestion": true/false,
    "confidence_adjustment": 0.0-1.0,
    "quality_score": 0.0-1.0,
    "relevance_score": 0.0-1.0,
    "reasoning": "brief explanation"
  },
  "filtered_alternatives": ["alt1", "alt2", ...],
  "morphological_notes": "additional morphological insights",
  "recommended_lemma": "suggested Estonian lemma if different"
}
```

# FILTERING GUIDELINES

## Keep if:
- High confidence (≥0.8) with method "estnltk+dict"
- Morphologically consistent analysis
- Valid Estonian lemma form
- Relevant to dialectal context
- Alternatives provide useful options

## Adjust confidence if:
- Method is "estnltk" only (reduce by 0.1-0.2)
- Lemma seems questionable (reduce by 0.2-0.3)
- Alternatives are poor quality (reduce by 0.1)
- South Estonian features not recognized (reduce by 0.1)

## Filter out if:
- Confidence < 0.4 and method is "levenshtein"
- Lemma is clearly wrong or non-Estonian
- Alternatives are mostly irrelevant
- Morphological analysis is inconsistent

## Alternative filtering:
- Keep only alternatives with edit distance ≤ 3
- Prioritize Estonian dictionary forms
- Remove obviously incorrect suggestions
- Limit to top 5-8 most relevant alternatives

# SPECIAL CONSIDERATIONS

- **South Estonian dialectal forms**: Be generous with Võro/Seto variants
- **Archaic vocabulary**: Preserve historical Estonian forms
- **Compound analysis**: Consider compound word structure
- **Morphological ambiguity**: Note when multiple analyses are valid
- **Finnish cognates**: Flag suggestions that might be Finnish rather than Estonian

Analyze each word systematically and provide filtered, high-quality suggestions that will improve the main lemmatization analysis.
"""
    
    def _get_enhanced_tabular_analysis_prompt(self):
        """Return the enhanced tabular analysis prompt with context support."""
        return """
You are an expert in Estonian linguistics with deep knowledge of Estonian dialectal variations, including standard Estonian, South Estonian (Võro, Seto), island dialects, and historical Estonian forms.

Your task is to analyze the provided Estonian runosong text tokens and create a comprehensive word-by-word analysis table.

IMPORTANT: All input texts are Estonian runosongs (regilaulud) in Estonian or Estonian dialects. Provide Estonian lemmas and analysis throughout.

You may receive additional reference context from Estonian linguistic analysis systems that has been pre-evaluated for quality and relevance. Use this filtered reference material where helpful but continue to apply expert judgment.

# OUTPUT FORMAT

Create a single word-by-word analysis table with this format:
| original form | comment | English translation | normalized orthography | lemma (original) | lemma (modern) | etymological root |
|--------------|---------|---------------------|------------------------|------------------|----------------|-------------------|
| [Word as in text] | [Translation notes] | [English equivalent] | [Modern spelling] | [Basic form in original] | [Modern language lemma] | [Etymological root(s)] |

# ANALYSIS GUIDELINES

## Core Principles
- Analyze each word in the order it appears in the original text
- For dialectal variants, provide both the original dialect lemma and modern standard equivalent 
- Include morphological analysis in comments when relevant
- Consider the cultural and linguistic context of the runosong tradition

##  Dialect Features
Pay special attention to the dialect and runosong-specific characteristics of the Estonian dialect detected

### Phonological Features
- Front vowel harmony: ä/ö vowels in suffixes (issäke, emäke, vennäkesed)
- Distinctive vowel patterns: mino (instead of minu), nähkä (instead of nahka)
- h-variation and consonant gradation patterns 
- etc

### Morphological Features

- Distinctive case endings and diminutive formations
- personal pronouns and possessive patterns

### Lexical Features
- dialect-specific vocabulary items and semantic shifts
- Archaic Estonian forms preserved in dialects
- Germanic and other loanword adaptations specific to the dialect

## Column Specifications

### original form
- Exact word as it appears in the text
- Preserve all diacritics, archaic spellings, and dialect features

### comment
- Brief morphological analysis with attention to dialect features
- Dialect identification notes (especially South Estonian markers when present)
- Cultural/contextual information
- Translation challenges or ambiguities

### English translation
- Direct English equivalent of the word
- Contextually appropriate meaning
- Cultural concepts explained when necessary

### normalized orthography
- Modern standard spelling of the word
- Apply current orthographic conventions
- Resolve archaic or dialect spellings

### lemma (original)
- Dictionary headword form in the original Estonian dialect
- Preserve historical or dialectal Estonian lemma forms
- Show the word as it would appear in an Estonian historical/dialectal dictionary

### lemma (modern)
- Contemporary standard Estonian lemma following EKSS dictionary standards
- Apply the systematic lemma selection guidelines (see Modern Lemma Selection Guide below)
- Standard Estonian reference form for linguistic analysis
- For diminutives: use the -ke form (not -kene) as the modern lemma (for example, use lemma "kanake" and not "kana" for the wordform "kanakõnõ")
- Do not change the case of the components of the determined lemma (e.g. correct is "leivapaik", incorrect is "leib+paik" or "leibpaik"; correct is "ümberringi", not "ümber+ring")


### etymological root
- Proto-Finnic or other reconstructed forms when known
- Related forms in sister languages
- Borrowing source if applicable

# MODERN LEMMA SELECTION GUIDE

Apply these systematic guidelines when determining the modern Estonian lemma, following morphological/etymological principles rather than semantic equivalence.

## CORE PRINCIPLE: ETYMOLOGICAL SIMILARITY
- **When choosing between alternatives, prefer stems that are etymologically related**
- **Give the standard basic form corresponding to the word in Estonian**
- **Preserve derivative word forms** - do not reduce to root forms
- **No semantic translation** - maintain etymological relationships only

## 1. Declinable Words (nouns, adjectives, pronouns, numerals)
- **Lemma = singular nominative** (sg Nom)
- Examples: *puu*, *puust*, *puudele* → **puu**; *oluline*, *olulisse* → **oluline**
- Same principle for declinable numerals (*kolm*, *kolmega* → **kolm**) and pronouns (*see*, *sellesse* → **see**)

## 1.1 Pronouns - Short vs. Long Forms
- **Preserve the form type (short/long) that appears in the text**
- Do NOT standardize all pronouns to one canonical form
- Short forms receive short lemmas; long forms receive long lemmas
- Example personal pronouns:
  - *minä*, "minno" → **mina** (long form lemma)
  - *mu* → **ma** (short form lemma)
  
## 1.2 For homonymous pronouns and dialectal pronouns where short/long form distinction is ambiguous:
  - Examples: "mind" (could be from "mina" or "ma"), "teda" (could be from "ta" or "tema"), "nea" (could be "nad" or "nemad")
  - Always lemmatize to the long form: mina, sina, tema, nemad, etc. 

## 2. Adjectives and Adverbs with Comparative Forms
- **Lemma = positive singular nominative**
- Examples: *ilus*, *ilusamaid* → **ilus**; *kõvasti*, *kõvemini* → **kõvasti**
- Comparative and superlative forms always receive the base lemma

## 3. Verbs
- **Lemma: ma-infinitive** (supine) – *lugema*, *kirjutama*
- All participles (*loetud*, *loetav*) and converbs (*lugedes*) receive the same verb lemma
- Based on EKI guidelines: ma-infinitive is the standard dictionary form

## 4. Invariable Words (adverbs, adpositions, conjunctions, interjections)
- **Lemma = same written form** (no morphological variants)
- Examples: *ja*, *ehk*, *oh* remain unchanged in lemmatization
- For -kuhu/-kus/-kust variants (*alla*, *all*, *alt*), each form is lemmatized separately
- Example: *peal*, *peale*, *pealt* remain unchanged
- Common runosong refrain words such as *kaske*, *kaasike*, *kaanike* etc remain unchanged in lemmatization

### 4.1 Derived Adverbs
- **Adverbs derived from adjectives maintain their adverbial lemma form**
- Do NOT reduce to the root adjective
- Examples:
  - *nobõhe* (quickly) → **nobedalt** (not *nobe*)
  - *palavasti* → **palavasti** (not *palav*)
  - *kõvasti*, *kõvalt* → **kõvasti**, **kõvalt** (not *kõva*)
- Rationale: Adverbs function as invariable words with distinct grammatical roles from their source adjectives

## 5. Compound Words and Derivatives
- **Lemma = complete compound in sg Nom/ma-infinitive form**
- **CRITICAL: Keep derivative word forms!**
- Examples: *raudteejaam*, *laulja*, *lugemine* maintain full forms
- Do NOT reduce derivatives to root forms
- Hyphenated compounds retain hyphens: *Ameerika-Eesti*, *ida-lääne*

## 6. Derivational Morphology
- **Preserve all derivational suffixes and infixes** - create different lemmas for derivationally distinct words
- Examples: *naeratama*/*naeratelema* → separate lemmas **naeratama** and **naeratelema**
- Maintain morphological distinctions that create semantic differences

## 7. Diminutives
- **Estonian diminutives**: Use -ke form in lemma (not -kene)
- Example: *kanakõnõ* → **kanake** (not *kana*)
- Preserve diminutive character in the lemma

## 8. Fallback Hierarchy for Non-Standard Forms (Estonian Texts Only)
When standard Estonian forms are impossible or nonexistent:
1. **Primary**: Standard Estonian dictionary form (EKSS) - etymologically related only
2. **Secondary**: Official Estonian dialectal standard form
   - EMK/VMS standardized forms
   - Seto dictionary forms
3. **Final**: Text-derived Estonian dialectal main form
   - Derive basic form from the text itself
   - Apply consistent Estonian dialectal orthographic principles
   - Preserve the dialectal character of the original

**CRITICAL**: Only use etymologically related forms within Estonian varieties. If *üsäh* is etymologically related to *üsk*, use *üsk*. Do NOT use semantically equivalent but etymologically unrelated forms like *süli*.

## 9. Orthographic Standards for Estonian Dialectal Forms
- Use standard-like orthography when possible based on specialized Estonian dictionaries:
- Eesti murrete sõnaraamat: https://arhiiv.eki.ee/dict/ems/
- Väike murdesõnastik: https://arhiiv.eki.ee/dict/vms/
- Avoid non-standard Estonian letters: Do not use letters like 'q' from modern Võru
  standardization (synaq.org)  
- Apply consistent orthographic principles for Estonian dialectal lemmas

## EXPLICITLY PROHIBITED PRACTICES
- **No semantic translation**: Do not provide semantically equivalent but etymologically unrelated lemmas
- **No cross-linguistic semantic matching**: Stay within Estonian varieties only
- **No Finnish fallbacks**: Do not use Finnish forms for Estonian texts
- **No root reduction**: Preserve derivative morphology
- **No standardization across form types**: Maintain pronoun length distinctions

## REQUIRED PRACTICES
- **Etymology over semantics**: Always prioritize etymological relationships
- **Preserve derivative forms**: Keep derivational suffixes and morphology
- **Estonian-only solutions**: Use Estonian standard or dialectal forms only
- **Text-based dialectal forms**: When no standard exists, derive from the text itself

## REFERENCED SUGGESTIONS
- When you receive pre-filtered EstNLTK reference suggestions:
- The context has been evaluated for relevance but trust your linguistic expertise over the suggestions
- Apply these conservative lemmatization guidelines even when suggestions differ
- Use EstNLTK suggestions as morphological reference, not lemmatization authority

## MORPHOLOGICAL COMPLEXITY
- Analyze compound words appropriately using conservative principles
- Identify diminutive and augmentative forms (especially South Estonian -ke, -kese patterns)
- Recognize archaic morphological patterns within Estonian varieties
- Consider Estonian internal dialectal influences (not cross-linguistic)
- Apply etymological similarity principles to complex forms

## SPECIAL CONSIDERATIONS FOR ESTONIAN RUNOSONGS

### Dialectal Features Recognition
- Recognize Estonian vowel harmony variations and dialectal innovations
- Identify Estonian consonant gradation patterns
- Note archaic Estonian case endings and verb forms
- Consider regional Estonian phonological changes

### Cultural Context
- Understand Estonian runosong-specific vocabulary
- Recognize Estonian formulaic expressions
- Consider Estonian ritual and ceremonial language
- Account for poetic and metaphorical usage in Estonian tradition

### South Estonian Varieties
- When encountering Võru or Seto forms without Standard Estonian equivalents:
- Check for official Võru/Seto dictionary standards first
- If no official standard exists, create text-derived dialectal lemma
- Maintain Estonian dialectal authenticity


CRITICAL REQUIREMENT: You MUST analyze EVERY SINGLE TOKEN provided in the input. Create a complete table with ALL provided tokens included - do not skip any tokens or abbreviate the analysis. The table must include every token from the provided list. DO NOT ask if you should continue - just provide the complete analysis for ALL provided tokens. Never truncate or summarize - analyze every single provided token.

CRITICAL REQUIREMENT 2: Focus on providing the correct modern lemma. 
Double check that the chosen lemma is actually a modern standard Estonian lemma.
Only provide a dialect or hypothetical made-up lemma in the modern lemma field when it is impossible to determine the likely modern equivalent.
Double check that the modern lemma chosen corresponds to the POS of the corresponding text wordform.


# 7. USING ENGLISH TRANSLATION
Use the provided English translation to mentally:
- Resolve ambiguous word meanings
- Confirm correct lemmatization
- Verify grammatical relationships
- Identify idiomatic expressions
- Understand contextual meaning
- Distinguish between homonyms
- Verify correct case assignments
- Identify metaphorical usage

# QUALITY CONTROL VALIDATION

Before finalizing each lemma, perform these validation checks:

## Semantic Validation
- Does the lemma appropriately represent the word's meaning in context?
- Is the lemma semantically plausible for an Estonian runosong?
- Does the lemma maintain narrative coherence?

## Morphological Validation  
- Is the proposed lemma morphologically sound in Estonian?
- Does the lemma follow Estonian word formation patterns?
- Are dialectal features appropriately standardized?

## Consistency Validation
- Is the lemmatization consistent with similar forms in the same text?
- Do related words (compounds, derivatives) show appropriate lemma relationships?
- Are parallel structures lemmatized consistently?

Remember: The English translation provided may contain errors. When you detect clear translation issues, prioritize linguistic context over the translation.

Return ONLY the markdown table with no additional explanations, comments, or text. The table must be complete and comprehensive for all provided tokens.

Provide the analysis table for the given tokens, ensuring each token receives thorough linguistic attention with special focus on South Estonian dialectal features.
"""
    
    def _get_tabular_analysis_prompt(self):
        """Override base class method to add South Estonian guidance to standard prompt."""
        return """
You are an expert in Estonian linguistics with deep knowledge of Estonian dialectal variations, including standard Estonian, South Estonian (Võro, Seto), island dialects, and historical Estonian forms.

Your task is to analyze the provided Estonian runosong text tokens and create a comprehensive word-by-word analysis table.

IMPORTANT: All input texts are Estonian runosongs (regilaulud) in Estonian or Estonian dialects. Provide Estonian lemmas and analysis throughout.

# OUTPUT FORMAT

Create a single word-by-word analysis table with this format:
| original form | comment | English translation | normalized orthography | lemma (original) | lemma (modern) | etymological root |
|--------------|---------|---------------------|------------------------|------------------|----------------|-------------------|
| [Word as in text] | [Translation notes] | [English equivalent] | [Modern spelling] | [Basic form in original] | [Modern language lemma] | [Etymological root(s)] |

# ANALYSIS GUIDELINES

## Core Principles
- Analyze each word in the order it appears in the original text
- For dialectal variants, provide both the original dialect lemma and modern standard equivalent 
- Include morphological analysis in comments when relevant
- Consider the cultural and linguistic context of the runosong tradition

##  Dialect Features
Pay special attention to the dialect and runosong-specific characteristics of the Estonian dialect detected

### Phonological Features
- Front vowel harmony: ä/ö vowels in suffixes (issäke, emäke, vennäkesed)
- Distinctive vowel patterns: mino (instead of minu), nähkä (instead of nahka)
- h-variation and consonant gradation patterns 
- etc

### Morphological Features

- Distinctive case endings and diminutive formations
- personal pronouns and possessive patterns

### Lexical Features
- dialect-specific vocabulary items and semantic shifts
- Archaic Estonian forms preserved in dialects
- Germanic and other loanword adaptations specific to the dialect

## Column Specifications

### original form
- Exact word as it appears in the text
- Preserve all diacritics, archaic spellings, and dialect features

### comment
- Brief morphological analysis with attention to dialect features
- Dialect identification notes (especially South Estonian markers when present)
- Cultural/contextual information
- Translation challenges or ambiguities

### English translation
- Direct English equivalent of the word
- Contextually appropriate meaning
- Cultural concepts explained when necessary

### normalized orthography
- Modern standard spelling of the word
- Apply current orthographic conventions
- Resolve archaic or dialect spellings

### lemma (original)
- Dictionary headword form in the original Estonian dialect
- Preserve historical or dialectal Estonian lemma forms
- Show the word as it would appear in an Estonian historical/dialectal dictionary

### lemma (modern)
- Contemporary standard Estonian lemma following EKSS dictionary standards
- Apply the systematic lemma selection guidelines (see Modern Lemma Selection Guide below)
- Standard Estonian reference form for linguistic analysis
- For diminutives: use the -ke form (not -kene) as the modern lemma (for example, use lemma "kanake" and not "kana" for the wordform "kanakõnõ")
- If the given word is a compound then indicate this with a + sign (e.g. hõbe+raha)
- If there are multiple standard corresponding lemmas available, prefer the one closer to the word occurring in the text, prioritizing the root connection over semantics (e.g. "üsäh" corresponds to lemmas "üsk" and "süli", choose "üsk")
- Do not change the case of the components of the determined lemma (e.g. correct is "leiva+paik", incorrect is "leib+paik"; correct is "ümber+ringi", not "ümber+ring")


### etymological root
- Proto-Finnic or other reconstructed forms when known
- Related forms in sister languages
- Borrowing source if applicable

# MODERN LEMMA SELECTION GUIDE

Apply these systematic guidelines when determining the modern Estonian lemma, following morphological/etymological principles rather than semantic equivalence.

## CORE PRINCIPLE: ETYMOLOGICAL SIMILARITY
- **When choosing between alternatives, prefer stems that are etymologically related**
- **Give the standard basic form corresponding to the word in Estonian**
- **Preserve derivative word forms** - do not reduce to root forms
- **No semantic translation** - maintain etymological relationships only

## 1. Declinable Words (nouns, adjectives, pronouns, numerals)
- **Lemma = singular nominative** (sg Nom)
- Examples: *puu*, *puust*, *puudele* → **puu**; *oluline*, *olulisse* → **oluline**
- Same principle for declinable numerals (*kolm*, *kolmega* → **kolm**) and pronouns (*see*, *sellesse* → **see**)

## 1.1 Pronouns - Short vs. Long Forms
- **Preserve the form type (short/long) that appears in the text**
- Do NOT standardize all pronouns to one canonical form
- Short forms receive short lemmas; long forms receive long lemmas
- Example personal pronouns:
  - *minä*, "minno" → **mina** (long form lemma)
  - *mu* → **ma** (short form lemma)
  
## 1.2 For homonymous pronouns and dialectal pronouns where short/long form distinction is ambiguous:
  - Examples: "mind" (could be from "mina" or "ma"), "teda" (could be from "ta" or "tema"), "nea" (could be "nad" or "nemad")
  - Always lemmatize to the long form: mina, sina, tema, nemad, etc. 

## 2. Adjectives and Adverbs with Comparative Forms
- **Lemma = positive singular nominative**
- Examples: *ilus*, *ilusamaid* → **ilus**; *kõvasti*, *kõvemini* → **kõvasti**
- Comparative and superlative forms always receive the base lemma

## 3. Verbs
- **Lemma: ma-infinitive** (supine) – *lugema*, *kirjutama*
- All participles (*loetud*, *loetav*) and converbs (*lugedes*) receive the same verb lemma
- Based on EKI guidelines: ma-infinitive is the standard dictionary form

## 4. Invariable Words (adverbs, adpositions, conjunctions, interjections)
- **Lemma = same written form** (no morphological variants)
- Examples: *ja*, *ehk*, *oh* remain unchanged in lemmatization
- For -kuhu/-kus/-kust variants (*alla*, *all*, *alt*), each form is lemmatized separately
- Example: *peal*, *peale*, *pealt* remain unchanged
- Common runosong refrain words such as *kaske*, *kaasike*, *kaanike* etc remain unchanged in lemmatization

### 4.1 Derived Adverbs
- **Adverbs derived from adjectives maintain their adverbial lemma form**
- Do NOT reduce to the root adjective
- Examples:
  - *nobõhe* (quickly) → **nobedalt** (not *nobe*)
  - *palavasti* → **palavasti** (not *palav*)
  - *kõvasti*, *kõvalt* → **kõvasti**, **kõvalt** (not *kõva*)
- Rationale: Adverbs function as invariable words with distinct grammatical roles from their source adjectives

## 5. Compound Words and Derivatives
- **Lemma = complete compound in sg Nom/ma-infinitive form**
- **CRITICAL: Keep derivative word forms!**
- Examples: *raudteejaam*, *laulja*, *lugemine* maintain full forms
- Do NOT reduce derivatives to root forms
- Hyphenated compounds retain hyphens: *Ameerika-Eesti*, *ida-lääne*

## 6. Derivational Morphology
- **Preserve all derivational suffixes and infixes** - create different lemmas for derivationally distinct words
- Examples: *naeratama*/*naeratelema* → separate lemmas **naeratama** and **naeratelema**
- Maintain morphological distinctions that create semantic differences

## 7. Diminutives
- **Estonian diminutives**: Use -ke form in lemma (not -kene)
- Example: *kanakõnõ* → **kanake** (not *kana*)
- Preserve diminutive character in the lemma

## 8. Fallback Hierarchy for Non-Standard Forms (Estonian Texts Only)
When standard Estonian forms are impossible or nonexistent:
1. **Primary**: Standard Estonian dictionary form (EKSS) - etymologically related only
2. **Secondary**: Official Estonian dialectal standard form
   - EMK/VMS standardized forms
   - Seto dictionary forms
3. **Final**: Text-derived Estonian dialectal main form
   - Derive basic form from the text itself
   - Apply consistent Estonian dialectal orthographic principles
   - Preserve the dialectal character of the original

**CRITICAL**: Only use etymologically related forms within Estonian varieties. If *üsäh* is etymologically related to *üsk*, use *üsk*. Do NOT use semantically equivalent but etymologically unrelated forms like *süli*.

## 9. Orthographic Standards for Estonian Dialectal Forms
- Use standard-like orthography when possible based on specialized Estonian dictionaries:
- Eesti murrete sõnaraamat: https://arhiiv.eki.ee/dict/ems/
- Väike murdesõnastik: https://arhiiv.eki.ee/dict/vms/
- Avoid non-standard Estonian letters: Do not use letters like 'q' from modern Võru
  standardization (synaq.org)  
- Apply consistent orthographic principles for Estonian dialectal lemmas

## EXPLICITLY PROHIBITED PRACTICES
- **No semantic translation**: Do not provide semantically equivalent but etymologically unrelated lemmas
- **No cross-linguistic semantic matching**: Stay within Estonian varieties only
- **No Finnish fallbacks**: Do not use Finnish forms for Estonian texts
- **No root reduction**: Preserve derivative morphology
- **No standardization across form types**: Maintain pronoun length distinctions

## REQUIRED PRACTICES
- **Etymology over semantics**: Always prioritize etymological relationships
- **Preserve derivative forms**: Keep derivational suffixes and morphology
- **Estonian-only solutions**: Use Estonian standard or dialectal forms only
- **Text-based dialectal forms**: When no standard exists, derive from the text itself

## REFERENCED SUGGESTIONS
- When you receive pre-filtered EstNLTK reference suggestions:
- The context has been evaluated for relevance but trust your linguistic expertise over the suggestions
- Apply these conservative lemmatization guidelines even when suggestions differ
- Use EstNLTK suggestions as morphological reference, not lemmatization authority

## MORPHOLOGICAL COMPLEXITY
- Analyze compound words appropriately using conservative principles
- Identify diminutive and augmentative forms (especially South Estonian -ke, -kese patterns)
- Recognize archaic morphological patterns within Estonian varieties
- Consider Estonian internal dialectal influences (not cross-linguistic)
- Apply etymological similarity principles to complex forms

## SPECIAL CONSIDERATIONS FOR ESTONIAN RUNOSONGS

### Dialectal Features Recognition
- Recognize Estonian vowel harmony variations and dialectal innovations
- Identify Estonian consonant gradation patterns
- Note archaic Estonian case endings and verb forms
- Consider regional Estonian phonological changes

### Cultural Context
- Understand Estonian runosong-specific vocabulary
- Recognize Estonian formulaic expressions
- Consider Estonian ritual and ceremonial language
- Account for poetic and metaphorical usage in Estonian tradition

### South Estonian Varieties
- When encountering Võru or Seto forms without Standard Estonian equivalents:
- Check for official Võru/Seto dictionary standards first
- If no official standard exists, create text-derived dialectal lemma
- Maintain Estonian dialectal authenticity


CRITICAL REQUIREMENT: You MUST analyze EVERY SINGLE TOKEN provided in the input. Create a complete table with ALL provided tokens included - do not skip any tokens or abbreviate the analysis. The table must include every token from the provided list. DO NOT ask if you should continue - just provide the complete analysis for ALL provided tokens. Never truncate or summarize - analyze every single provided token.

CRITICAL REQUIREMENT 2: Focus on providing the correct modern lemma. 
Double check that the chosen lemma is actually a modern standard Estonian lemma.
Only provide a dialect or hypothetical made-up lemma in the modern lemma field when it is impossible to determine the likely modern equivalent.
Double check that the modern lemma chosen corresponds to the POS of the corresponding text wordform.


# 7. USING ENGLISH TRANSLATION
Use the provided English translation to mentally:
- Resolve ambiguous word meanings
- Confirm correct lemmatization
- Verify grammatical relationships
- Identify idiomatic expressions
- Understand contextual meaning
- Distinguish between homonyms
- Verify correct case assignments
- Identify metaphorical usage

# QUALITY CONTROL VALIDATION

Before finalizing each lemma, perform these validation checks:

## Semantic Validation
- Does the lemma appropriately represent the word's meaning in context?
- Is the lemma semantically plausible for an Estonian runosong?
- Does the lemma maintain narrative coherence?

## Morphological Validation  
- Is the proposed lemma morphologically sound in Estonian?
- Does the lemma follow Estonian word formation patterns?
- Are dialectal features appropriately standardized?

## Consistency Validation
- Is the lemmatization consistent with similar forms in the same text?
- Do related words (compounds, derivatives) show appropriate lemma relationships?
- Are parallel structures lemmatized consistently?

Remember: The English translation provided may contain errors. When you detect clear translation issues, prioritize linguistic context over the translation.

Return ONLY the markdown table with no additional explanations, comments, or text. The table must be complete and comprehensive for all provided tokens.

Provide the analysis table for the given tokens, ensuring each token receives thorough linguistic attention with special focus on South Estonian dialectal features.
"""
    
    @lru_cache(maxsize=128)
    def _generate_raw_estnltk_analysis(self, text: str) -> dict:
        """
        Generate raw EstNLTK analysis without dictionary lookup or method selection.
        
        Args:
            text: The runosong text to analyze
            
        Returns:
            Dictionary containing raw EstNLTK morphological analysis
        """
        if not self.context_enabled:
            return {"raw_analysis_available": False}
        
        try:
            from estnltk import Text
            from estnltk.taggers import VabamorfTagger
            
            # Clean punctuation like in the backend
            punctuation_chars = '´`\'''""„"‚'
            cleaned_text = text
            for char in punctuation_chars:
                cleaned_text = cleaned_text.replace(char, '')
            
            # Create EstNLTK text object and analyze
            text_obj = Text(cleaned_text)
            text_obj.tag_layer(['words', 'sentences', 'morph_analysis'])
            
            # Standard disambiguation
            tagger = VabamorfTagger(postdisambiguate=True, output_layer='morph_analysis_disamb')
            tagger.tag(text_obj)
            
            # Extract raw EstNLTK results using EstNLTK's own tokenization
            raw_results = []
            original_words = self._extract_words_from_text(text)
            
            # Build mapping from original words to EstNLTK words
            original_word_index = 0
            
            for i, word_span in enumerate(text_obj.words):
                estnltk_word = word_span.text
                
                # Skip pure punctuation in EstNLTK tokenization
                if re.fullmatch(r'[^\w\s]+', estnltk_word):
                    continue
                
                # Get corresponding original word
                original_word = original_words[original_word_index] if original_word_index < len(original_words) else estnltk_word
                
                if i < len(text_obj.morph_analysis_disamb):
                    morph_layer = text_obj.morph_analysis_disamb[i]
                    if morph_layer.annotations:
                        annotation = morph_layer.annotations[0]  # Take first (best) analysis
                        raw_results.append({
                            "original_word": original_word,
                            "raw_estnltk_lemma": annotation.get('lemma', ''),
                            "raw_estnltk_pos": annotation.get('partofspeech', ''),
                            "raw_estnltk_form": annotation.get('form', ''),
                            "raw_estnltk_ending": annotation.get('ending', ''),
                            "raw_estnltk_clitic": annotation.get('clitic', ''),
                            "raw_estnltk_root": annotation.get('root', ''),
                            "raw_estnltk_text": morph_layer.text
                        })
                    else:
                        # No annotation available
                        raw_results.append({
                            "original_word": original_word,
                            "raw_estnltk_lemma": '',
                            "raw_estnltk_pos": '',
                            "raw_estnltk_form": '',
                            "raw_estnltk_ending": '',
                            "raw_estnltk_clitic": '',
                            "raw_estnltk_root": '',
                            "raw_estnltk_text": estnltk_word
                        })
                else:
                    # Word not in EstNLTK analysis
                    raw_results.append({
                        "original_word": original_word,
                        "raw_estnltk_lemma": '',
                        "raw_estnltk_pos": '',
                        "raw_estnltk_form": '',
                        "raw_estnltk_ending": '',
                        "raw_estnltk_clitic": '',
                        "raw_estnltk_root": '',
                        "raw_estnltk_text": estnltk_word
                    })
                
                original_word_index += 1
            
            return {
                "raw_analysis_available": True,
                "raw_estnltk_analyses": raw_results,
                "total_words": len(raw_results)
            }
            
        except Exception as e:
            self.logger.warning(f"Raw EstNLTK analysis failed: {e}")
            return {"raw_analysis_available": False, "error": str(e)}

    def _generate_estnltk_context_cached(self, text: str, metadata: dict = None) -> dict:
        """
        Generate EstNLTK+dictionary context for the given text with caching.
        
        Args:
            text: The runosong text to analyze
            metadata: Additional metadata (place_name, year, etc.) for enhanced context
            
        Returns:
            Dictionary containing context information
        """
        with self._stats_lock:
            if text in getattr(self, '_context_cache', {}):
                self.enhancement_stats["cache_hits"] += 1
            else:
                self.enhancement_stats["cache_misses"] += 1
        
        if not self.context_enabled:
            return {"context_available": False}
        
        try:
            start_time = time.time()
            
            # Use fixed word extraction to prevent word merging
            words = self._extract_words_from_text(text)
            estnltk_results = []
            
            # Process each word individually to prevent EstNLTK Text() merging
            for word in words:
                try:
                    # Process individual word
                    individual_results = self.estnltk_analyzer.lemmatize_text(word)
                    
                    if individual_results:
                        # Take the first result and ensure original word matches
                        result = individual_results[0]
                        result["original_word"] = word
                        estnltk_results.append(result)
                    else:
                        # Create fallback for unprocessed words
                        estnltk_results.append({
                            "original_word": word,
                            "best_lemma": word.lower(), 
                            "method": "fallback",
                            "confidence_score": 0.3,
                            "alternatives": [],
                            "is_unknown": True,
                            "pos": "",
                            "form": ""
                        })
                        
                except Exception as e:
                    self.logger.warning(f"EstNLTK processing failed for word '{word}': {e}")
                    # Create error fallback
                    estnltk_results.append({
                        "original_word": word,
                        "best_lemma": word.lower(),
                        "method": "error_fallback",
                        "confidence_score": 0.1, 
                        "alternatives": [],
                        "is_unknown": True,
                        "pos": "",
                        "form": ""
                    })
            
            # Process results into context format
            context = {
                "context_available": True,
                "word_analyses": [],
                "processing_summary": {
                    "total_words": len(estnltk_results),
                    "methods_used": [],
                    "confidence_distribution": []
                }
            }
            
            for result in estnltk_results:
                word_analysis = {
                    "original_word": result["original_word"],
                    "estnltk_lemma": result["best_lemma"],
                    "confidence": result["confidence_score"],
                    "method": result["method"],
                    "pos": result.get("pos", ""),
                    "form": result.get("form", ""),
                    "alternatives": self._filter_alternatives(result.get("original_word", ""), result.get("alternatives", [])),
                    "is_unknown": result["is_unknown"]
                }
                
                context["word_analyses"].append(word_analysis)
                context["processing_summary"]["methods_used"].append(result["method"])
                context["processing_summary"]["confidence_distribution"].append(result["confidence_score"])
            
            # Add summary statistics
            methods_count = {}
            for method in context["processing_summary"]["methods_used"]:
                methods_count[method] = methods_count.get(method, 0) + 1
            context["processing_summary"]["methods_used"] = methods_count
            
            avg_confidence = sum(context["processing_summary"]["confidence_distribution"]) / len(context["processing_summary"]["confidence_distribution"]) if context["processing_summary"]["confidence_distribution"] else 0
            context["processing_summary"]["average_confidence"] = avg_confidence
            
            with self._stats_lock:
                self.enhancement_stats["context_generation_time"] += time.time() - start_time
            
            self.logger.info(f"Generated EstNLTK context for {len(estnltk_results)} words")
            
            return context
            
        except Exception as e:
            self.logger.error(f"Failed to generate EstNLTK context: {e}")
            return {"context_available": False, "error": str(e)}
    
    def _optimize_word_analyses_for_tokens(self, word_analyses: list) -> list:
        """
        Optimize word analyses for token usage by:
        1. Removing redundant "is_unknown": false entries
        2. Deduplicating repeated words with occurrence counts
        3. Using shorter field names while preserving all essential data
        
        Args:
            word_analyses: List of word analysis dictionaries
            
        Returns:
            Optimized list with reduced token usage
        """
        if not word_analyses:
            return word_analyses
        
        # Track unique words and their analyses
        unique_words = {}
        
        for analysis in word_analyses:
            word = analysis.get("original_word", "")
            
            # Create optimized analysis entry
            optimized_analysis = {
                "word": word,  # Shorter field name
                "lemma": analysis.get("estnltk_lemma", ""),  # Shorter field name
                "confidence": analysis.get("confidence", 0),
                "method": analysis.get("method", ""),
                "pos": analysis.get("pos", ""),
                "form": analysis.get("form", ""),
                "alternatives": analysis.get("alternatives", [])
                # Note: Removed "is_unknown": false as it's redundant when false
            }
            
            # Only include is_unknown if it's True (saves tokens for false cases)
            if analysis.get("is_unknown", False):
                optimized_analysis["is_unknown"] = True
            
            # Track word occurrences
            if word in unique_words:
                unique_words[word]["count"] += 1
            else:
                unique_words[word] = {
                    **optimized_analysis,
                    "count": 1
                }
        
        # Convert back to list format with occurrence counts
        optimized_list = []
        for word_data in unique_words.values():
            # Only include count if > 1 (saves tokens for single occurrences)
            if word_data["count"] > 1:
                optimized_list.append(word_data)
            else:
                # Remove count field for single occurrences
                analysis_without_count = {k: v for k, v in word_data.items() if k != "count"}
                optimized_list.append(analysis_without_count)
        
        return optimized_list
    
    def _build_context_formatting_content(self, estnltk_data: dict, corpus_data: dict, original_text: str, metadata: dict = None) -> str:
        """
        Build the content string for context formatting API call including metadata.
        
        Args:
            estnltk_data: EstNLTK analysis data
            corpus_data: Corpus knowledge data
            original_text: Original Estonian runosong text
            metadata: Additional metadata (place_name, year, etc.)
            
        Returns:
            Formatted content string for API call
        """
        content_parts = [
            f"Original Estonian Runosong Text:\n{original_text}",
            f"\nEstNLTK+Dictionary Analysis Data:\n{json.dumps(estnltk_data, ensure_ascii=False, indent=2)}",
            f"\nCorpus Knowledge from Previous Analyses:\n{json.dumps(corpus_data, ensure_ascii=False, indent=2)}"
        ]
        
        if metadata:
            metadata_str = "\n\nAdditional Context for Enhanced Analysis:"
            for key, value in metadata.items():
                if value:
                    clean_value = value.replace('"', '') if isinstance(value, str) else value
                    metadata_str += f"\n{key}: {clean_value}"
            content_parts.append(metadata_str)
        
        content_parts.append("\n\nPlease analyze and provide structured guidance for Estonian lemmatization decisions. Use the geographic and temporal context information to enhance the accuracy of your Estonian linguistic analysis.")
        
        return "\n".join(content_parts)
    
    def _build_pre_analysis_content(self, pre_analysis_input: dict, metadata: dict = None) -> str:
        """
        Build the content string for pre-analysis API call including metadata.
        
        Args:
            pre_analysis_input: Pre-analysis input data
            metadata: Additional metadata (place_name, year, etc.)
            
        Returns:
            Formatted content string for API call
        """
        content_parts = [
            f"EstNLTK Analysis Data:\n{json.dumps(pre_analysis_input, ensure_ascii=False, indent=2)}"
        ]
        
        if metadata:
            metadata_str = "\n\nAdditional Context for Enhanced Analysis:"
            for key, value in metadata.items():
                if value:
                    clean_value = value.replace('"', '') if isinstance(value, str) else value
                    metadata_str += f"\n{key}: {clean_value}"
            content_parts.append(metadata_str)
        
        content_parts.append("\n\nPlease evaluate the following EstNLTK suggestions for Estonian runosong analysis. Use the geographic and temporal context information to enhance the accuracy of your evaluation.")
        
        return "\n".join(content_parts)
    
    def _simple_context_formatting(self, context: dict, original_text: str, metadata: dict = None) -> dict:
        """
        Simple context formatting approach - single API call to format EstNLTK context.
        This replaces the complex pre-analysis with the user's original simpler idea.
        Now includes the original runosong text for better contextual analysis.
        
        Args:
            context: Context dictionary from _generate_estnltk_context_cached
            original_text: The original Estonian runosong text that was analyzed
            metadata: Additional metadata (place_name, year, etc.) for enhanced context
            
        Returns:
            Context with simply formatted EstNLTK data
        """
        if not context.get("context_available", False):
            return context
        
        # No word limit - process any size text
        word_count = len(context.get("word_analyses", []))
        self.logger.info(f"🔄 Preparing Estonian linguistic context for AI analysis ({word_count} words)")
        
        # Create cache key including original text
        cache_data = {"context": context, "original_text": original_text}
        cache_key = f"simple_format_{json.dumps(cache_data, sort_keys=True)}"
        
        # Check cache first
        with self._stats_lock:
            if cache_key in self._pre_analysis_cache:
                self.enhancement_stats["pre_analysis_cache_hits"] += 1
                self.logger.debug(f"⚡ Using cached Estonian context (cache hit)")
                return self._pre_analysis_cache[cache_key]
            else:
                self.enhancement_stats["pre_analysis_cache_misses"] += 1
        
        start_time = time.time()
        
        try:
            # Get corpus context for current text words
            current_words = self._extract_words_from_text(original_text)
            raw_corpus_context = self._get_corpus_context(current_words)
            corpus_context = self._preprocess_corpus_context(raw_corpus_context)
            
            # Prepare optimized data for single API call including original text and corpus
            # Token optimization: deduplicate words and remove redundant fields
            raw_word_analyses = context.get("word_analyses", [])
            optimized_word_analyses = self._optimize_word_analyses_for_tokens(raw_word_analyses)
            
            estnltk_data = {
                "original_runosong_text": original_text,
                "word_count": word_count,
                "word_analyses": optimized_word_analyses
            }
            
            # Add corpus context if available
            corpus_data = {
                "corpus_available": len(corpus_context) > 0,
                "relevant_corpus_entries": corpus_context
            }
            
            # Improved formatting prompt with balanced weight (50% corpus, 50% linguistic analysis) and no meta-references
            formatting_prompt = """You are an expert Estonian linguist preparing guidance for lemmatizing Estonian runosong texts. Your analysis will be provided to another linguistic expert LLM who will perform the final lemmatization without access to the raw technical data.

You have access to:
1. **Original Estonian runosong text** - Primary context for understanding meanings and dialectal patterns
2. **Corpus knowledge from previous analyses** - Lemmatization patterns from similar texts, may or may not be accurate
3. **Estonian linguistic analysis data** - Automated morphological suggestions of EstNLTK and dictionary related words to evaluate

Your task is to synthesize this information into clear, actionable guidance that balances corpus insights (50%) with linguistic analysis (50%) for optimal lemmatization decisions.
Focus on error prevention in the final lemmatization phase. 
Try to provide only or mostly general recommendations that you yourself would need when you would have only the input runosong text and if you would not have the additional corpus, EstNLTK and dictionaries context.
Keep in mind that your answer will be forwarded to another linguistic expert LLM that, unlike you, has also the correct translation of the text.
Thus, the other LLM actually has more useful context but does not have the corpus and EstNLTK context that you have.

## PRIMARY GUIDANCE SOURCE: INSPIRED BY YOUR REVIEW OF THE INPUT TEXT AND THE CORPUS CONTEXT (50% weight)

**When corpus data is available, prioritize these insights:**
**However, the MODERN LEMMA SELECTION GUIDE should be considered more important than existing lemmas in the corpus.
For example: for pronouns the correct lemmatization approach is: *mu* → **ma** (short form lemma).
If the corpus has *mu* → **mina**, then it should not be brought out as a correct example.

### Proven Lemmatization Patterns
- Identify words where corpus data shows consistent, reliable lemma assignments across multiple texts
- Highlight successful disambiguation strategies for ambiguous word forms
- Note semantic patterns that guide lemma selection in different contexts
- For diminutives: if the corpus data shows lemmas missing the -ke form then do not suggest using the lemma missing the suffix, we want lemmas with the -ke suffix for diminutives (e.g. when the lemma of 'õdekene' is 'õde' in the corpus then the correct lemma to suggest would be the standard language 'õeke'')
- Do not include in your answer consistent lemma assignments that go against the lemma selection guide below (e.g. do not include that "ta" shows consistent lemmatization pattern to "tema")

### Contextual Disambiguation
- For words with multiple possible lemmas in corpus, analyze which interpretation fits current text context
- Consider semantic role, grammatical function, and poetic meaning in this specific runosong
- Recognize when current context may require different lemma than previous instances
- Do not include in your answer  lemma assignments that go against the lemma selection guide below 
(e.g. do not suggest "sul" should be lemmatized to "sina" as it clearly should be lemmatized as "sa" according to the guide below)

### Cross-textual Validation
- Use corpus frequency patterns to validate morphologically ambiguous forms
- Leverage proven dialectal form → modern lemma mappings from previous successful analyses
- For diminutives: use the -ke form (not -kene or a form without diminutive suffix) as the modern lemma (for example, use lemma "kanake" and not "kana" for the wordform "kanakõnõ")
-

## SECONDARY GUIDANCE SOURCE: LINGUISTIC ANALYSIS INSPIRED BY YOUR REVIEW OF THE INPUT TEXT AND THE ESTNLTK+DICT (NOT VERY RELIABLE) CONTEXT (50% weight)

### Morphological Insights
- Identify reliable morphological patterns that align with known Estonian dialectal features
- Recognize South Estonian, Võro/Seto, or other specific dialectal markers
- Validate case endings, verb forms, and compound structures

### Dialectal Pattern Recognition
- Note consistent vowel harmony, consonant alternations, or other phonological patterns
- Identify archaic Estonian forms preserved in runosong tradition
- Distinguish Estonian dialectal forms from potential Finnish cognate confusion

### Lemmatization Recommendations
- Provide shortly some general recommendations based on your analysis of the text but do not present any specific lemma recommendations in addition to what the corpus and EstNLTK context suggests.
- Provide the EstNLTK lemmas that in your opinion are very likely the correct ones considering the context but only for dialect word forms that could be hard for an expert to lemmatize otherwise.
- Consider also the list of alternative lemmas that can also contain the right lemma.

### About the sources
- Shortly indicate whether you used or did not use the provided EstNLTK+dict sources when compiling this part of your answer

## OUTPUT FORMAT

Provide balanced guidance in this structure:

### **CORPUS-BASED LEMMATIZATION GUIDANCE:**
[List words where corpus data provides strong directional guidance, focusing on disambiguation strategies and proven patterns. Present as direct linguistic facts, not technical references.]

### **MORPHOLOGICAL LEMMATIZATION GUIDANCE:**  
[List words where morphological analysis provides reliable insights, focusing on dialectal patterns and Estonian-specific features. Present as linguistic expertise, not system outputs.]

### **DIALECTAL PATTERN RECOGNITION:**
[Identify 3-5 key dialectal features that will guide lemmatization decisions - vowel patterns, morphological endings, lexical characteristics, etc.]

### **CONTEXTUAL DISAMBIGUATION STRATEGIES:**
[Provide specific guidance for handling ambiguous forms in this text, balancing corpus evidence with morphological analysis.]

### **LEMMATIZATION RECOMMENDATIONS:**
[Targeted advice for challenging words where expert judgment is needed, integrating both corpus patterns and morphological insights.]

## CRITICAL PRINCIPLES

- **Balance corpus and morphological evidence equally (50/50)**
- **Present insights as direct linguistic facts, not technical system references**
- **Focus on actionable guidance for lemmatization decisions**
- **Prioritize Estonian dialectal expertise and runosong context**
- **Provide disambiguation strategies, not rigid rules**
- **Consider semantic context alongside morphological evidence**

Your goal is to provide expert linguistic guidance that enables accurate, contextually appropriate lemmatization of Estonian runosongs by integrating corpus wisdom with morphological analysis.

[END OF ANALYSIS GUIDE, THE FOLLOWING IS ADDITIONAL CONTEXT FOR CORRECT LEMMATIZATION:]

## Lemmatization Guide (Also accessible to the LLM in the final step, no need to repeat in your answer)

- Contemporary standard Estonian lemma following EKSS dictionary standards
- Apply the systematic lemma selection guidelines (see Modern Lemma Selection Guide below)
- For diminutives: use the -ke form (not -kene) as the modern lemma (for example, use lemma "kanake" and not "kana" for the wordform "kanakõnõ")
- If the given word is a compound then indicate this with a + sign (e.g. hõbe+raha)
- If there are multiple standard corresponding lemmas available, prefer the one closer to the word occurring in the text, prioritizing the root connection over semantics (e.g. "üsäh" corresponds to lemmas "üsk" and "süli", choose "üsk")
- If the first component of the lemma is not in nominative case then do not change it into nominative (e.g. correct is "leiva+paik", incorrect is "leib+paik"; correct is "ümber+ringi", not "ümber+ring" )

# MODERN LEMMA SELECTION GUIDE

Apply these systematic guidelines when determining the modern Estonian lemma, following morphological/etymological principles rather than semantic equivalence.

## CORE PRINCIPLE: ETYMOLOGICAL SIMILARITY
- **When choosing between alternatives, prefer stems that are etymologically related**
- **Give the standard basic form corresponding to the word in Estonian**
- **Preserve derivative word forms** - do not reduce to root forms
- **No semantic translation** - maintain etymological relationships only

## 1. Declinable Words (nouns, adjectives, pronouns, numerals)
- **Lemma = singular nominative** (sg Nom)
- Examples: *puu*, *puust*, *puudele* → **puu**; *oluline*, *olulisse* → **oluline**
- Same principle for declinable numerals (*kolm*, *kolmega* → **kolm**) and pronouns (*see*, *sellesse* → **see**)

## 1.1 Pronouns - Short vs. Long Forms
- **Preserve the form type (short/long) that appears in the text**
- Do NOT standardize all pronouns to one canonical form
- Short forms receive short lemmas; long forms receive long lemmas
- Example personal pronouns:
  - *minä*, "minno" → **mina** (long form lemma)
  - *mu* → **ma** (short form lemma)
  
## 1.2 For homonymous pronouns and dialectal pronouns where short/long form distinction is ambiguous:
  - Examples: "mind" (could be from "mina" or "ma"), "teda" (could be from "ta" or "tema"), "nea" (could be "nad" or "nemad")
  - Always lemmatize to the long form: mina, sina, tema, nemad, etc. 

## 2. Adjectives and Adverbs with Comparative Forms
- **Lemma = positive singular nominative**
- Examples: *ilus*, *ilusamaid* → **ilus**; *kõvasti*, *kõvemini* → **kõvasti**
- Comparative and superlative forms always receive the base lemma

## 3. Verbs
- **Lemma: ma-infinitive** (supine) – *lugema*, *kirjutama*
- All participles (*loetud*, *loetav*) and converbs (*lugedes*) receive the same verb lemma
- Based on EKI guidelines: ma-infinitive is the standard dictionary form

## 4. Invariable Words (adverbs, adpositions, conjunctions, interjections)
- **Lemma = same written form** (no morphological variants)
- Examples: *ja*, *ehk*, *oh* remain unchanged in lemmatization
- For -kuhu/-kus/-kust variants (*alla*, *all*, *alt*), each form is lemmatized separately
- Example: *peal*, *peale*, *pealt* remain unchanged
- Common runosong refrain words such as *kaske*, *kaasike*, *kaanike* etc remain unchanged in lemmatization

### 4.1 Derived Adverbs
- **Adverbs derived from adjectives maintain their adverbial lemma form**
- Do NOT reduce to the root adjective
- Examples:
  - *nobõhe* (quickly) → **nobedalt** (not *nobe*)
  - *palavasti* → **palavasti** (not *palav*)
  - *kõvasti*, *kõvalt* → **kõvasti**, **kõvalt** (not *kõva*)
- Rationale: Adverbs function as invariable words with distinct grammatical roles from their source adjectives

## 5. Compound Words and Derivatives
- **Lemma = complete compound in sg Nom/ma-infinitive form**
- **CRITICAL: Keep derivative word forms!**
- Examples: *raudteejaam*, *laulja*, *lugemine* maintain full forms
- Do NOT reduce derivatives to root forms
- Hyphenated compounds retain hyphens: *Ameerika-Eesti*, *ida-lääne*

## 6. Derivational Morphology
- **Preserve all derivational suffixes and infixes** - create different lemmas for derivationally distinct words
- Examples: *naeratama*/*naeratelema* → separate lemmas **naeratama** and **naeratelema**
- Maintain morphological distinctions that create semantic differences

## 7. Diminutives
- **Estonian diminutives**: Use -ke form in lemma (not -kene)
- Example: *kanakõnõ* → **kanake** (not *kana*)
- Preserve diminutive character in the lemma

## 8. Fallback Hierarchy for Non-Standard Forms (Estonian Texts Only)
When standard Estonian forms are impossible or nonexistent:
1. **Primary**: Standard Estonian dictionary form (EKSS) - etymologically related only
2. **Secondary**: Official Estonian dialectal standard form
   - EMK/VMS standardized forms
   - Seto dictionary forms
3. **Final**: Text-derived Estonian dialectal main form
   - Derive basic form from the text itself
   - Apply consistent Estonian dialectal orthographic principles
   - Preserve the dialectal character of the original

**CRITICAL**: Only use etymologically related forms within Estonian varieties. If *üsäh* is etymologically related to *üsk*, use *üsk*. Do NOT use semantically equivalent but etymologically unrelated forms like *süli*.

## 9. Orthographic Standards for Estonian Dialectal Forms
- Use standard-like orthography when possible based on specialized Estonian dictionaries:
- Eesti murrete sõnaraamat: https://arhiiv.eki.ee/dict/ems/
- Väike murdesõnastik: https://arhiiv.eki.ee/dict/vms/
- Avoid non-standard Estonian letters: Do not use letters like 'q' from modern Võru
  standardization (synaq.org)  
- Apply consistent orthographic principles for Estonian dialectal lemmas
"""

            # Single API call with retry logic
            def make_context_formatting_call():
                return self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=4000,
                    temperature=0,
                    system=[{
                        "type": "text", 
                        "text": formatting_prompt,
                        "cache_control": {"type": "ephemeral"}
                    }],
                    messages=[{
                        "role": "user",
                        "content": self._build_context_formatting_content(estnltk_data, corpus_data, original_text, metadata)
                    }]
                )
            
            response = self._api_call_with_retry("Context formatting", make_context_formatting_call)
            
            if response.content:
                formatted_context = response.content[0].text.strip()
                
                # Log token usage metrics with cache information
                if hasattr(response, 'usage') and response.usage:
                    usage_info = {
                        'input_tokens': response.usage.input_tokens,
                        'output_tokens': response.usage.output_tokens,
                        'cache_creation_input_tokens': getattr(response.usage, 'cache_creation_input_tokens', 0),
                        'cache_read_input_tokens': getattr(response.usage, 'cache_read_input_tokens', 0)
                    }
                    self.logger.info(f"🔄 Estonian context preparation token usage: {usage_info}")
                
                # Create simple formatted context with optimization data
                simple_context = {
                    **context,
                    "simple_formatting_available": True,
                    "formatted_context": formatted_context,
                    "formatting_method": "simple_single_call",
                    "corpus_entries_used": len(corpus_context),
                    "optimization_data": {
                        "original_word_analyses": raw_word_analyses,
                        "optimized_word_analyses": optimized_word_analyses,
                        "optimization_stats": {
                            "original_entries": len(raw_word_analyses),
                            "optimized_entries": len(optimized_word_analyses),
                            "entries_reduction": len(raw_word_analyses) - len(optimized_word_analyses),
                            "entries_reduction_percent": round(((len(raw_word_analyses) - len(optimized_word_analyses)) / len(raw_word_analyses) * 100), 1) if raw_word_analyses else 0,
                            "original_json_size": len(json.dumps(raw_word_analyses, ensure_ascii=False)),
                            "optimized_json_size": len(json.dumps(optimized_word_analyses, ensure_ascii=False)),
                            "json_size_reduction": len(json.dumps(raw_word_analyses, ensure_ascii=False)) - len(json.dumps(optimized_word_analyses, ensure_ascii=False)),
                            "json_size_reduction_percent": round(((len(json.dumps(raw_word_analyses, ensure_ascii=False)) - len(json.dumps(optimized_word_analyses, ensure_ascii=False))) / len(json.dumps(raw_word_analyses, ensure_ascii=False)) * 100), 1) if raw_word_analyses else 0
                        }
                    }
                }
                
                # Cache the result
                with self._stats_lock:
                    self._pre_analysis_cache[cache_key] = simple_context
                    self.enhancement_stats["pre_analysis_time"] += time.time() - start_time
                    self.enhancement_stats["pre_analysis_calls"] += 1
                
                self.logger.info(f"✅ Estonian context preparation completed ({word_count} words, {time.time() - start_time:.2f}s, cached for reuse)")
                return simple_context
            else:
                self.logger.warning("No response from simple formatting API call")
                return context
                
        except Exception as e:
            self.logger.error(f"Simple formatting failed: {e}")
            return context

    def _pre_analyze_context(self, context: dict) -> dict:
        """
        DEPRECATED: Complex pre-analysis with batch processing and JSON parsing.
        This method has been replaced by _simple_context_formatting() which is 8-40x faster
        and more reliable. Kept for reference and comparison purposes.
        
        Args:
            context: Context dictionary from _generate_estnltk_context_cached
            
        Returns:
            Pre-analyzed context with quality scores and filtering
        """
        if not context.get("context_available", False):
            return context
        
        # Performance optimization: Skip pre-analysis for texts with >100 words
        word_count = len(context.get("word_analyses", []))
        if word_count > 100:
            self.logger.info(f"Skipping pre-analysis for large text ({word_count} words) - using standard EstNLTK context")
            return {
                **context,
                "pre_analysis_available": False,
                "pre_analysis_summary": f"Pre-analysis skipped for performance (>{word_count} words)"
            }
        
        # Create cache key for pre-analysis
        cache_key = json.dumps(context, sort_keys=True)
        
        # Check cache first
        with self._stats_lock:
            if cache_key in self._pre_analysis_cache:
                self.enhancement_stats["pre_analysis_cache_hits"] += 1
                return self._pre_analysis_cache[cache_key]
            else:
                self.enhancement_stats["pre_analysis_cache_misses"] += 1
        
        start_time = time.time()
        
        try:
            # Prepare input for pre-analysis
            word_analyses = context.get("word_analyses", [])
            
            # Batch process words for pre-analysis (optimized batch size)
            pre_analyzed_words = []
            batch_size = min(15, max(5, word_count // 5))  # Adaptive batch size: 5-15 words
            
            for i in range(0, len(word_analyses), batch_size):
                batch = word_analyses[i:i + batch_size]
                
                # Create pre-analysis input
                pre_analysis_input = {
                    "words": []
                }
                
                for word_analysis in batch:
                    word_input = {
                        "original_word": word_analysis["original_word"],
                        "suggested_lemma": word_analysis["estnltk_lemma"],
                        "confidence_score": word_analysis["confidence"],
                        "method": word_analysis["method"],
                        "pos": word_analysis.get("pos", ""),
                        "form": word_analysis.get("form", ""),
                        "alternatives": self._filter_alternatives(word_analysis.get("original_word", ""), word_analysis.get("alternatives", []))
                    }
                    pre_analysis_input["words"].append(word_input)
                
                # Make API call for pre-analysis with retry logic
                def make_pre_analysis_call():
                    return self.client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=2000,
                        temperature=0,
                        system=[{
                            "type": "text",
                            "text": self.pre_analysis_prompt
                        }],
                        messages=[{
                            "role": "user",
                            "content": self._build_pre_analysis_content(pre_analysis_input, metadata)
                        }]
                    )
                
                response = self._api_call_with_retry("Pre-analysis", make_pre_analysis_call)
                
                if response.content:
                    pre_analysis_result = response.content[0].text.strip()
                    self.logger.debug(f"Pre-analysis batch result: {pre_analysis_result[:200]}...")
                    
                    # Parse pre-analysis result
                    try:
                        # Extract JSON from the response
                        import re
                        json_matches = re.findall(r'```json\s*(\{.*?\})\s*```', pre_analysis_result, re.DOTALL)
                        if not json_matches:
                            # Try to find JSON without code blocks
                            json_matches = re.findall(r'(\{[^}]*"original_word"[^}]*\})', pre_analysis_result, re.DOTALL)
                        
                        self.logger.debug(f"Found {len(json_matches)} JSON matches for batch")
                        
                        batch_parsed = 0
                        for json_match in json_matches:
                            try:
                                evaluation = json.loads(json_match)
                                if "original_word" in evaluation:
                                    pre_analyzed_words.append(evaluation)
                                    batch_parsed += 1
                            except json.JSONDecodeError as e:
                                self.logger.debug(f"JSON parse error: {e}")
                                continue
                        
                        self.logger.debug(f"Successfully parsed {batch_parsed} words from batch")
                        
                        # If no valid JSON found, fall back to original analysis
                        if batch_parsed == 0:
                            self.logger.warning("No valid JSON found in pre-analysis response, using fallback")
                            for word_analysis in batch:
                                pre_analyzed_words.append({
                                    "original_word": word_analysis["original_word"],
                                    "evaluation": {
                                        "keep_suggestion": True,
                                        "confidence_adjustment": word_analysis["confidence"],
                                        "quality_score": 0.7,
                                        "relevance_score": 0.7,
                                        "reasoning": "Pre-analysis parsing failed, using original"
                                    },
                                    "filtered_alternatives": self._filter_alternatives(word_analysis.get("original_word", ""), word_analysis.get("alternatives", [])),
                                    "morphological_notes": "",
                                    "recommended_lemma": word_analysis["estnltk_lemma"]
                                })
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to parse pre-analysis result: {e}")
                        # Fall back to original analysis
                        for word_analysis in batch:
                            pre_analyzed_words.append({
                                "original_word": word_analysis["original_word"],
                                "evaluation": {
                                    "keep_suggestion": True,
                                    "confidence_adjustment": word_analysis["confidence"],
                                    "quality_score": 0.7,
                                    "relevance_score": 0.7,
                                    "reasoning": "Pre-analysis failed, using original"
                                },
                                "filtered_alternatives": self._filter_alternatives(word_analysis.get("original_word", ""), word_analysis.get("alternatives", [])),
                                "morphological_notes": "",
                                "recommended_lemma": word_analysis["estnltk_lemma"]
                            })
                
                # Add shorter delay between batches for better performance
                if i + batch_size < len(word_analyses):
                    time.sleep(0.5)  # Reduced from 1 second to 0.5 seconds
            
            # Update context with pre-analysis results
            pre_analyzed_context = {
                **context,
                "pre_analysis_available": True,
                "pre_analyzed_words": pre_analyzed_words
            }
            
            # Cache the result
            with self._stats_lock:
                self._pre_analysis_cache[cache_key] = pre_analyzed_context
                self.enhancement_stats["pre_analysis_time"] += time.time() - start_time
                self.enhancement_stats["pre_analysis_calls"] += 1
            
            self.logger.info(f"Pre-analyzed {len(pre_analyzed_words)} words from EstNLTK context")
            
            return pre_analyzed_context
            
        except Exception as e:
            self.logger.error(f"Pre-analysis failed: {e}")
            # Return original context if pre-analysis fails
            return context
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]
    
    def _format_simple_context_for_prompt(self, context: dict) -> str:
        """
        Format simply formatted EstNLTK context for inclusion in Claude prompt.
        
        Args:
            context: Simple formatted context dictionary
            
        Returns:
            Formatted context string for prompt
        """
        if not context.get("context_available", False):
            return "\n# REFERENCE CONTEXT\nNo additional morphological reference context available.\n"
        
        if not context.get("simple_formatting_available", False):
            # Fall back to standard formatting
            return self._format_context_for_prompt(context)
        
        formatted_context = context.get("formatted_context", "")
        
        context_header = [
            "\n# ESTONIAN MORPHOLOGICAL REFERENCE CONTEXT (FORMATTED)",
            "",
            "The following Estonian linguistic analysis has been formatted by an Estonian linguistics expert.",
            "Use this context to inform your lemmatization decisions where helpful.",
            "Trust your Estonian linguistic expertise when making final choices.",
            "",
            formatted_context,
            "",
            "Remember: This context is provided as reference material to assist with your analysis.",
            "Apply your Estonian linguistic expertise when making final lemmatization decisions.",
            ""
        ]
        
        return "\n".join(context_header)

    def _format_pre_analyzed_context_for_prompt(self, context: dict) -> str:
        """
        Format pre-analyzed EstNLTK context for inclusion in Claude prompt.
        
        Args:
            context: Pre-analyzed context dictionary
            
        Returns:
            Formatted context string for prompt
        """
        if not context.get("context_available", False):
            return "\n# REFERENCE CONTEXT\nNo additional morphological reference context available.\n"
        
        if not context.get("pre_analysis_available", False):
            # Fall back to standard formatting
            return self._format_context_for_prompt(context)
        
        context_lines = [
            "\n# ESTONIAN MORPHOLOGICAL REFERENCE CONTEXT (PRE-ANALYZED)",
            "",
            "The following Estonian linguistic analysis has been pre-evaluated for quality and relevance.",
            "High-quality suggestions are provided with assessment scores. Use this refined context where helpful.",
            "Trust your Estonian linguistic expertise when disagreeing with suggestions.",
            ""
        ]
        
        # Group words by pre-analysis quality
        high_quality_words = []
        moderate_quality_words = []
        low_quality_words = []
        
        pre_analyzed_words = context.get("pre_analyzed_words", [])
        
        for word_eval in pre_analyzed_words:
            if not word_eval.get("evaluation", {}).get("keep_suggestion", False):
                continue
            
            word = word_eval["original_word"]
            evaluation = word_eval["evaluation"]
            quality_score = evaluation.get("quality_score", 0.5)
            relevance_score = evaluation.get("relevance_score", 0.5)
            confidence = evaluation.get("confidence_adjustment", 0.5)
            reasoning = evaluation.get("reasoning", "")
            
            recommended_lemma = word_eval.get("recommended_lemma", word)
            filtered_alternatives = word_eval.get("filtered_alternatives", [])
            morphological_notes = word_eval.get("morphological_notes", "")
            
            # Create entry with quality indicators
            entry = f"• {word} → {recommended_lemma} (quality: {quality_score:.2f}, relevance: {relevance_score:.2f}, confidence: {confidence:.2f})"
            
            if morphological_notes:
                entry += f"\n  Morphology: {morphological_notes}"
            
            if filtered_alternatives:
                entry += f"\n  Alternatives: {', '.join(filtered_alternatives)}"
            
            if reasoning:
                entry += f"\n  Assessment: {reasoning}"
            
            # Group by overall quality
            overall_quality = (quality_score + relevance_score) / 2
            if overall_quality >= 0.8:
                high_quality_words.append(entry)
            elif overall_quality >= 0.6:
                moderate_quality_words.append(entry)
            else:
                low_quality_words.append(entry)
        
        # Add high quality suggestions
        if high_quality_words:
            context_lines.extend([
                "## HIGH QUALITY SUGGESTIONS:",
                "These suggestions have been evaluated as high-quality and relevant:",
                ""
            ])
            context_lines.extend(high_quality_words)
            context_lines.append("")
        
        # Add moderate quality suggestions
        if moderate_quality_words:
            context_lines.extend([
                "## MODERATE QUALITY SUGGESTIONS:",
                "These suggestions are useful but require expert judgment:",
                ""
            ])
            context_lines.extend(moderate_quality_words)
            context_lines.append("")
        
        # Add low quality suggestions
        if low_quality_words:
            context_lines.extend([
                "## LOWER QUALITY SUGGESTIONS:",
                "These suggestions may be less reliable - use with caution:",
                ""
            ])
            context_lines.extend(low_quality_words)
            context_lines.append("")
        
        # Add final note
        context_lines.extend([
            "Remember: This context has been pre-evaluated for quality and relevance.",
            "Quality and relevance scores indicate the assessment of each suggestion.",
            "Apply your Estonian linguistic expertise when making final decisions.",
            ""
        ])
        
        return "\n".join(context_lines)
    
    def _extract_words_from_text(self, text: str) -> list:
        """Extract words from text, preserving word boundaries for dialectal forms."""
        import re
        # Split by whitespace first to preserve word boundaries
        # This prevents EstNLTK from merging words like "Koolõ' õks" → "Koolõ'õks"
        raw_words = text.split()
        
        # Clean each word but preserve the core word
        cleaned_words = []
        for word in raw_words:
            # Remove leading/trailing punctuation but keep internal apostrophes/hyphens
            cleaned = re.sub(r'^[^\w\'´`-]+|[^\w\'´`-]+$', '', word)
            if cleaned and not re.match(r'^[^\w]+$', cleaned):  # Skip pure punctuation
                cleaned_words.append(cleaned)
        
        return cleaned_words
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]
    
    def _filter_alternatives(self, original_word: str, alternatives: list, max_alternatives: int = 10) -> list:
        """
        Filter alternatives to most relevant entries.
        
        Args:
            original_word: The original word being analyzed
            alternatives: List of alternative lemmas/forms
            max_alternatives: Maximum number of alternatives to return
            
        Returns:
            Filtered list of most relevant alternatives
        """
        if not alternatives:
            return []
        
        # Filter out single-character strings and very short words
        filtered = [alt for alt in alternatives if len(alt) > 1]
        
        # If we have few alternatives, return them all
        if len(filtered) <= max_alternatives:
            return filtered
        
        # Calculate Levenshtein distance for each alternative
        scored_alternatives = []
        original_lower = original_word.lower().strip("'´`")  # Remove dialectal marks for comparison
        
        for alt in filtered:
            distance = self._levenshtein_distance(original_lower, alt.lower())
            # Prefer shorter distances (more similar words)
            scored_alternatives.append((alt, distance))
        
        # Sort by distance (closest first) and take top N
        scored_alternatives.sort(key=lambda x: x[1])
        
        return [alt for alt, _ in scored_alternatives[:max_alternatives]]
    
    def _get_completed_tokens_from_analysis(self, analysis: str) -> list:
        """Extract completed tokens from a markdown table analysis."""
        try:
            if not analysis or "|" not in analysis:
                return []
            
            # Split into lines and find table rows
            lines = analysis.strip().split('\n')
            table_lines = [line for line in lines if line.strip().startswith('|') and 'original form' not in line and '---' not in line]
            
            completed_tokens = []
            for line in table_lines:
                if not line.strip():
                    continue
                    
                # Split by | and clean up
                parts = [part.strip() for part in line.split('|')[1:-1]]  # Remove empty first/last parts
                
                if len(parts) >= 1:  # At least original form
                    original_form = parts[0]
                    if original_form:  # Non-empty
                        completed_tokens.append(original_form)
            
            return completed_tokens
            
        except Exception as e:
            self.logger.error(f"Error extracting tokens from analysis: {e}")
            return []
    
    def _get_unannotated_slice(self, full_text: str, completed_tokens: list, max_tokens: int = 80) -> str:
        """Get the next slice of unannotated tokens, up to max_tokens."""
        words = self._extract_words_from_text(full_text)
        
        # Log basic statistics for debugging
        total_tokens = len(words)
        completed_count = len(completed_tokens)
        self.logger.debug(f"Total tokens: {total_tokens}, Completed: {completed_count}")
        
        # Track occurrences in original text
        original_counts = {}
        for word in words:
            original_counts[word] = original_counts.get(word, 0) + 1
            
        # Track occurrences in completed tokens
        annotated_counts = {}
        for token in completed_tokens:
            annotated_counts[token] = annotated_counts.get(token, 0) + 1
        
        # Check if all tokens are fully annotated
        all_annotated = True
        for word, count in original_counts.items():
            if annotated_counts.get(word, 0) < count:
                all_annotated = False
                break
        
        if all_annotated:
            self.logger.info("All tokens appear to be fully annotated")
            return ""
            
        # Get unannotated tokens
        unannotated_slice = []
        token_positions = []  # Track which positions we're taking from
        
        for idx, word in enumerate(words):
            current_annotated = annotated_counts.get(word, 0)
            if current_annotated < original_counts[word]:
                unannotated_slice.append(word)
                token_positions.append(idx)
                annotated_counts[word] = current_annotated + 1
                if len(unannotated_slice) >= max_tokens:
                    break
        
        # Log detailed info about what we're returning
        if unannotated_slice:
            self.logger.debug(f"Returning slice with {len(unannotated_slice)} tokens from positions: {token_positions}")
            self.logger.debug(f"Slice content: {' '.join(unannotated_slice)}")
        else:
            self.logger.warning("No unannotated tokens found but all_annotated=False - possible inconsistency")
        
        return ' '.join(unannotated_slice)
    
    def _merge_analysis_tables(self, partial_analyses: list, original_text: str) -> str:
        """Merge partial analysis tables while preserving original text order."""
        try:
            original_tokens = self._extract_words_from_text(original_text)
            token_analyses = {}
            
            # Parse all partial analyses
            for analysis in partial_analyses:
                if not analysis or "|" not in analysis:
                    continue
                    
                # Extract table rows
                lines = analysis.strip().split('\n')
                table_lines = [line for line in lines if line.strip().startswith('|') and 'original form' not in line and '---' not in line]
                
                for line in table_lines:
                    if not line.strip():
                        continue
                        
                    # Split by | and clean up
                    parts = [part.strip() for part in line.split('|')[1:-1]]  # Remove empty first/last parts
                    
                    if len(parts) >= 7:  # Ensure we have all columns
                        original_form = parts[0]
                        if original_form:
                            # Store the complete row
                            if original_form not in token_analyses:
                                token_analyses[original_form] = []
                            token_analyses[original_form].append(parts)
            
            # Build the merged table
            merged_lines = [
                "| original form | comment | English translation | normalized orthography | lemma (original) | lemma (modern) | etymological root |",
                "|--------------|---------|---------------------|------------------------|------------------|----------------|-------------------|"
            ]
            
            used_counts = {}
            for token in original_tokens:
                if token not in used_counts:
                    used_counts[token] = 0
                    
                if token in token_analyses and used_counts[token] < len(token_analyses[token]):
                    row_parts = token_analyses[token][used_counts[token]]
                    merged_lines.append(f"| {' | '.join(row_parts)} |")
                    used_counts[token] += 1
                else:
                    self.logger.warning(f"Missing analysis for token: {token} (occurrence {used_counts[token] + 1})")
                    # Add placeholder row
                    merged_lines.append(f"| {token} | Missing analysis | {token} | {token} | {token} | {token} | Unknown |")
            
            return "\n".join(merged_lines)
            
        except Exception as e:
            self.logger.error(f"Error merging analysis tables: {e}")
            return None
    
    def _verify_analysis_completeness(self, original_text: str, analysis: str) -> bool:
        """Verify that all token occurrences from original text are present in the analysis."""
        try:
            original_tokens = self._extract_words_from_text(original_text)
            analyzed_tokens = self._get_completed_tokens_from_analysis(analysis)
            
            # Count occurrences
            original_counts = {}
            for token in original_tokens:
                original_counts[token] = original_counts.get(token, 0) + 1
            
            analyzed_counts = {}
            for token in analyzed_tokens:
                analyzed_counts[token] = analyzed_counts.get(token, 0) + 1
            
            # Check for missing or underrepresented tokens
            missing_tokens = []
            underrepresented_tokens = []
            
            for token, count in original_counts.items():
                if token not in analyzed_counts:
                    missing_tokens.append(token)
                elif analyzed_counts[token] < count:
                    underrepresented_tokens.append(f"{token} ({analyzed_counts[token]}/{count})")
            
            if missing_tokens:
                self.logger.warning(f"Tokens missing from analysis: {', '.join(missing_tokens[:10])}")
                if len(missing_tokens) > 10:
                    self.logger.warning(f"... and {len(missing_tokens) - 10} more missing tokens")
                return False
                
            if underrepresented_tokens:
                self.logger.warning(f"Underrepresented tokens: {', '.join(underrepresented_tokens[:10])}")
                if len(underrepresented_tokens) > 10:
                    self.logger.warning(f"... and {len(underrepresented_tokens) - 10} more underrepresented tokens")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error verifying analysis completeness: {e}")
            return False
    
    def _format_context_for_prompt(self, context: dict) -> str:
        """
        Format EstNLTK context for inclusion in Claude prompt with improved filtering.
        
        Args:
            context: Context dictionary from _generate_estnltk_context_cached
            
        Returns:
            Formatted context string for prompt
        """
        if not context.get("context_available", False):
            return "\n# REFERENCE CONTEXT\nNo additional morphological reference context available.\n"
        
        context_lines = [
            "\n# ESTONIAN MORPHOLOGICAL REFERENCE CONTEXT",
            "",
            "The following Estonian linguistic analysis is provided by Estonian lemmatization tools (EstNLTK + Estonian dictionaries).",
            "Use this Estonian reference material with caution for dialectal forms. Ensure all lemmas are Estonian forms, not Finnish cognates.",
            "Trust your Estonian linguistic expertise over these suggestions. When you disagree with a suggestion, briefly explain your reasoning.",
            ""
        ]
        
        # Group words by analysis characteristics for better presentation
        high_confidence_words = []
        uncertain_words = []
        challenging_words = []
        
        # Track seen word→lemma combinations to avoid duplicates
        seen_combinations = set()
        
        for analysis in context["word_analyses"]:
            word = analysis["original_word"]
            lemma = analysis["estnltk_lemma"]
            conf = analysis["confidence"]
            method = analysis["method"]
            alternatives = analysis["alternatives"]
            pos = analysis.get("pos", "")
            form = analysis.get("form", "")
            
            # Skip entries with method "unknown" or confidence < 0.4
            if method == "unknown" or conf < 0.4:
                continue
            
            # Create a unique key for this word→lemma combination
            combination_key = f"{word}→{lemma}"
            
            # Skip if we've already seen this exact combination
            if combination_key in seen_combinations:
                continue
            seen_combinations.add(combination_key)
            
            # Filter alternatives using the improved filtering method
            filtered_alternatives = self._filter_alternatives(word, alternatives)
            
            # Build entry with method-based confidence indicators
            method_indicator = ""
            if method == "estnltk":
                method_indicator = " [EstNLTK analysis]"
            elif method == "estnltk+dict":
                method_indicator = " [EstNLTK+dictionary]"
            elif method == "dict":
                method_indicator = " [Dictionary lookup]"
            
            entry = f"• {word} → {lemma} (confidence: {conf:.2f}){method_indicator}"
            
            # Add morphological information ONLY for EstNLTK-analyzed words
            if method == "estnltk" and (pos or form):
                morpho_info = []
                if pos:
                    morpho_info.append(f"POS: {pos}")
                if form:
                    morpho_info.append(f"Form: {form}")
                if morpho_info:
                    entry += f"\n  Morphology: {', '.join(morpho_info)}"
            
            if filtered_alternatives:
                entry += f"\n  Alternatives: {', '.join(filtered_alternatives)}"
            
            # Group by confidence, treating EstNLTK-only results with more caution
            if method == "estnltk+dict" and conf >= 0.9:
                # Only EstNLTK+dictionary combinations with high confidence go to high confidence
                high_confidence_words.append(entry)
            elif method == "estnltk" and conf >= 0.8:
                # EstNLTK-only results go to moderate confidence even with high scores
                uncertain_words.append(entry)
            elif conf >= 0.6:
                uncertain_words.append(entry)
            else:
                challenging_words.append(entry)
        
        # Add high confidence suggestions
        if high_confidence_words:
            context_lines.extend([
                "## HIGH CONFIDENCE SUGGESTIONS:",
                "These EstNLTK+dictionary combinations are likely reliable for validation:",
                ""
            ])
            context_lines.extend(high_confidence_words)
            context_lines.append("")
        
        # Add uncertain cases
        if uncertain_words:
            context_lines.extend([
                "## MODERATE CONFIDENCE SUGGESTIONS:",
                "Consider these suggestions but apply expert judgment (includes EstNLTK-only results):",
                ""
            ])
            context_lines.extend(uncertain_words)
            context_lines.append("")
        
        # Add challenging words
        if challenging_words:
            context_lines.extend([
                "## CHALLENGING WORDS:",
                "Lower confidence - dialectal expertise especially valuable:",
                ""
            ])
            context_lines.extend(challenging_words)
            context_lines.append("")
        
        # Add final note
        context_lines.extend([
            "Remember: Use this context selectively. Your linguistic expertise is authoritative.",
            "When choosing different lemmas than suggested, briefly note your reasoning in the comment field.",
            ""
        ])
        
        return "\n".join(context_lines)
    
    def _analyze_text_with_context_chunked(self, text: str, title: str, improved_translation: str, 
                                          dialect_info: str = None, metadata: dict = None, 
                                          prompt_name: str = "tabular_analysis_with_context", 
                                          estnltk_context: dict = None, 
                                          estnltk_formatted_context: str = None, 
                                          max_tokens_per_pass: int = 80, 
                                          max_iterations: int = 50) -> str:
        """
        Analyze text with pre-analyzed EstNLTK context enhancement using chunking.
        
        Args:
            text: Original text to analyze
            title: Title of the text
            improved_translation: Improved English translation
            dialect_info: Dialect information from previous stage
            metadata: Additional metadata
            prompt_name: Name of the analysis prompt to use
            estnltk_context: Pre-generated EstNLTK context (optional)
            estnltk_formatted_context: Pre-formatted EstNLTK context string (optional)
            max_tokens_per_pass: Maximum tokens to process per API call
            max_iterations: Maximum iterations to prevent infinite loops
            
        Returns:
            Enhanced analysis result
        """
        start_time = time.time()
        
        try:
            # Generate or use provided EstNLTK context
            if estnltk_context is None:
                context = self._generate_estnltk_context_cached(text, metadata)
            else:
                context = estnltk_context
            
            # Use pre-formatted context if provided, otherwise format it
            if estnltk_formatted_context is not None:
                context_text = estnltk_formatted_context
                self.logger.debug(f"✅ Using pre-formatted context for {title} ({len(context_text)} chars)")
            else:
                # Only call formatting if not pre-provided
                self.logger.debug(f"🔄 Formatting context for {title} (not pre-provided)")
                simple_context = self._simple_context_formatting(context, text, metadata)
                if simple_context.get("simple_formatting_available", False):
                    context_text = self._format_simple_context_for_prompt(simple_context)
                else:
                    # Fallback to standard context formatting if simple formatting failed
                    self.logger.warning(f"Simple formatting failed for {title}, using standard EstNLTK context formatting")
                    context_text = self._format_context_for_prompt(context)
            
            # Get the enhanced prompt
            if prompt_name not in self.enhanced_analysis_prompts:
                self.logger.warning(f"Enhanced prompt '{prompt_name}' not found, using standard")
                return self.analyze_text(text, title, improved_translation, dialect_info, metadata, "tabular_analysis")
            
            analysis_prompt = self.enhanced_analysis_prompts[prompt_name]
            
            # Iterative chunked processing
            completed_tokens = []
            partial_analyses = []
            previously_requested_slices = set()
            
            iteration = 0
            while iteration < max_iterations:
                iteration += 1
                
                remaining_slice = self._get_unannotated_slice(text, completed_tokens, max_tokens_per_pass)
                
                # Check if we're done
                if not remaining_slice:
                    self.logger.info("All tokens analyzed for enhanced analysis")
                    break
                    
                self.logger.info(f"🔍 AI analysis (enhanced) - processing batch {iteration} of text: {title}")
                    
                # Check if we're in a loop
                if remaining_slice in previously_requested_slices:
                    self.logger.warning(f"Duplicate enhanced analysis request detected for: {remaining_slice}")
                    self.logger.warning("Breaking potential infinite loop")
                    break
                
                previously_requested_slices.add(remaining_slice)
                
                # Create system messages with caching
                system_messages = [
                    {
                        "type": "text", 
                        "text": analysis_prompt + context_text,
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
                
                # Prepare text content
                text_content = f"Original text:\n{text}\n\nImproved translation:\n{improved_translation}"
                
                if dialect_info:
                    text_content += f"\n\nDialect information:\n{dialect_info}"
                
                if metadata:
                    metadata_str = "\n\nAdditional context:"
                    for key, value in metadata.items():
                        if value:
                            clean_value = value.replace('"', '') if isinstance(value, str) else value
                            metadata_str += f"\n{key}: {clean_value}"
                    text_content += metadata_str
                
                system_messages.append({
                    "type": "text",
                    "text": text_content,
                    "cache_control": {"type": "ephemeral"}
                })
                
                # Make API call for this slice with retry logic
                def make_analysis_call():
                    return self.client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=4000,
                        temperature=0,
                        system=system_messages,
                        messages=[{
                            "role": "user",
                            "content": f"Please analyze the following tokens from the text titled '{title}':\n{remaining_slice}\n\nProvide complete analysis for these tokens, maintaining their exact order as they appear in the text. Return ONLY the markdown table with no explanations, comments, or additional text."
                        }]
                    )
                
                response = self._api_call_with_retry(f"Analysis chunk {iteration + 1} for {title}", make_analysis_call)
                
                if not response.content:
                    self.logger.error("No content in enhanced analysis response")
                    break
                
                partial_result = response.content[0].text.strip()
                
                if partial_result:
                    partial_analyses.append(partial_result)
                    
                    # Track how many tokens we had before
                    previous_token_count = len(completed_tokens)
                    
                    newly_completed = self._get_completed_tokens_from_analysis(partial_result)
                    completed_tokens.extend(newly_completed)
                    
                    # Check if we actually added any new tokens
                    actual_new_tokens = len(completed_tokens) - previous_token_count
                    self.logger.info(f"Enhanced analysis completed {len(newly_completed)} new tokens, added {actual_new_tokens} unique")
                    
                    # Break if we're not making progress anymore
                    if actual_new_tokens == 0 and iteration > 3:
                        self.logger.warning("No new unique tokens added in enhanced analysis, breaking loop")
                        break
                else:
                    self.logger.error("Failed to produce enhanced analysis for current slice")
                    break
                
                # Add delay between iterations
                time.sleep(2)
            
            # Check if we hit the iteration limit
            if iteration >= max_iterations:
                self.logger.warning(f"Reached maximum iterations ({max_iterations}) for enhanced analysis, forcing completion")
            
            # Merge partial analyses
            if partial_analyses:
                self.logger.info(f"Merging {len(partial_analyses)} partial enhanced analyses for {title}")
                final_analysis = self._merge_analysis_tables(partial_analyses, text)
                
                if final_analysis:
                    is_complete = self._verify_analysis_completeness(text, final_analysis)
                    self.logger.info(f"Enhanced analysis complete: {is_complete}")
                    
                    # Update statistics
                    with self._stats_lock:
                        self.enhancement_stats["enhanced_processing_time"] += time.time() - start_time
                        self.enhancement_stats["texts_with_context"] += 1
                    
                    return final_analysis
            
            self.logger.error(f"Failed to complete enhanced analysis for {title}")
            return None
            
        except Exception as e:
            self.logger.error(f"Enhanced analysis failed for {title}: {e}")
            return None
    
    def analyze_text_with_context(self, text: str, title: str, improved_translation: str, 
                                 dialect_info: str = None, metadata: dict = None, 
                                 prompt_name: str = "tabular_analysis_with_context", 
                                 estnltk_context: dict = None, 
                                 estnltk_formatted_context: str = None) -> str:
        """
        Analyze text with pre-analyzed EstNLTK context enhancement (chunked version).
        
        Args:
            text: Original text to analyze
            title: Title of the text
            improved_translation: Improved English translation
            dialect_info: Dialect information from previous stage
            metadata: Additional metadata
            prompt_name: Name of the analysis prompt to use
            estnltk_context: Pre-generated EstNLTK context (optional)
            estnltk_formatted_context: Pre-formatted EstNLTK context string (optional)
            
        Returns:
            Enhanced analysis result
        """
        return self._analyze_text_with_context_chunked(
            text, title, improved_translation, dialect_info, metadata, 
            prompt_name, estnltk_context, estnltk_formatted_context
        )
    
    def _analyze_text_chunked(self, text: str, title: str, improved_translation: str, 
                             dialect_info: str = None, metadata: dict = None, 
                             prompt_name: str = "tabular_analysis", 
                             max_tokens_per_pass: int = 80, 
                             max_iterations: int = 50) -> str:
        """
        Analyze text using chunked processing to prevent truncation.
        
        Args:
            text: Original text to analyze
            title: Title of the text
            improved_translation: Improved English translation
            dialect_info: Dialect information from previous stage
            metadata: Additional metadata
            prompt_name: Name of the analysis prompt to use
            max_tokens_per_pass: Maximum tokens to process per API call
            max_iterations: Maximum iterations to prevent infinite loops
            
        Returns:
            Analysis result
        """
        start_time = time.time()
        
        try:
            # Get the analysis prompt
            if prompt_name not in self.analysis_prompts:
                self.logger.warning(f"Analysis prompt '{prompt_name}' not found, using default")
                prompt_name = "tabular_analysis"
            
            analysis_prompt = self.analysis_prompts[prompt_name]
            
            # Iterative chunked processing
            completed_tokens = []
            partial_analyses = []
            previously_requested_slices = set()
            
            iteration = 0
            while iteration < max_iterations:
                iteration += 1
                
                remaining_slice = self._get_unannotated_slice(text, completed_tokens, max_tokens_per_pass)
                
                # Check if we're done
                if not remaining_slice:
                    self.logger.info("All tokens analyzed for standard analysis")
                    break
                    
                self.logger.info(f"🔍 AI analysis (standard) - processing batch {iteration} of text: {title}")
                    
                # Check if we're in a loop
                if remaining_slice in previously_requested_slices:
                    self.logger.warning(f"Duplicate standard analysis request detected for: {remaining_slice}")
                    self.logger.warning("Breaking potential infinite loop")
                    break
                
                previously_requested_slices.add(remaining_slice)
                
                # Create system messages with caching
                system_messages = [
                    {
                        "type": "text", 
                        "text": analysis_prompt,
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
                
                # Prepare text content
                text_content = f"Original text:\n{text}\n\nImproved translation:\n{improved_translation}"
                
                if dialect_info:
                    text_content += f"\n\nDialect information:\n{dialect_info}"
                
                if metadata:
                    metadata_str = "\n\nAdditional context:"
                    for key, value in metadata.items():
                        if value:
                            clean_value = value.replace('"', '') if isinstance(value, str) else value
                            metadata_str += f"\n{key}: {clean_value}"
                    text_content += metadata_str
                
                system_messages.append({
                    "type": "text",
                    "text": text_content,
                    "cache_control": {"type": "ephemeral"}
                })
                
                # Make API call for this slice with retry logic
                def make_analysis_call():
                    return self.client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=4000,
                        temperature=0,
                        system=system_messages,
                        messages=[{
                            "role": "user",
                            "content": f"Please analyze the following tokens from the text titled '{title}':\n{remaining_slice}\n\nProvide complete analysis for these tokens, maintaining their exact order as they appear in the text. Return ONLY the markdown table with no explanations, comments, or additional text."
                        }]
                    )
                
                response = self._api_call_with_retry(f"Analysis chunk {iteration + 1} for {title}", make_analysis_call)
                
                if not response.content:
                    self.logger.error("No content in standard analysis response")
                    break
                
                partial_result = response.content[0].text.strip()
                
                if partial_result:
                    partial_analyses.append(partial_result)
                    
                    # Track how many tokens we had before
                    previous_token_count = len(completed_tokens)
                    
                    newly_completed = self._get_completed_tokens_from_analysis(partial_result)
                    completed_tokens.extend(newly_completed)
                    
                    # Check if we actually added any new tokens
                    actual_new_tokens = len(completed_tokens) - previous_token_count
                    self.logger.info(f"Standard analysis completed {len(newly_completed)} new tokens, added {actual_new_tokens} unique")
                    
                    # Break if we're not making progress anymore
                    if actual_new_tokens == 0 and iteration > 3:
                        self.logger.warning("No new unique tokens added in standard analysis, breaking loop")
                        break
                else:
                    self.logger.error("Failed to produce standard analysis for current slice")
                    break
                
                # Add delay between iterations
                time.sleep(2)
            
            # Check if we hit the iteration limit
            if iteration >= max_iterations:
                self.logger.warning(f"Reached maximum iterations ({max_iterations}) for standard analysis, forcing completion")
            
            # Merge partial analyses
            if partial_analyses:
                self.logger.info(f"Merging {len(partial_analyses)} partial standard analyses for {title}")
                final_analysis = self._merge_analysis_tables(partial_analyses, text)
                
                if final_analysis:
                    is_complete = self._verify_analysis_completeness(text, final_analysis)
                    self.logger.info(f"Standard analysis complete: {is_complete}")
                    
                    return final_analysis
            
            self.logger.error(f"Failed to complete standard analysis for {title}")
            return None
            
        except Exception as e:
            self.logger.error(f"Standard analysis failed for {title}: {e}")
            return None
    
    def _process_text_stage(self, stage_func, *args, **kwargs):
        """Helper method to process a single stage with error handling."""
        try:
            return stage_func(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Stage processing failed: {e}")
            return None
    
    def _parse_row_ranges(self, rows_spec: str = None, start_row: int = 1, end_row: int = None, total_rows: int = None):
        """
        Parse row range specifications into a list of 0-based indices.
        
        Args:
            rows_spec: String like "1,3,5-8" or "10-20" (1-based)
            start_row: Start row (1-based, default: 1)
            end_row: End row (1-based, default: None for all rows)
            total_rows: Total number of rows available
        
        Returns:
            List of 0-based indices to process
        """
        if total_rows is None:
            raise ValueError("total_rows must be specified")
        
        # If specific rows format is provided, use that
        if rows_spec:
            indices = []
            parts = rows_spec.split(',')
            
            for part in parts:
                part = part.strip()
                if '-' in part:
                    # Handle range like "5-10"
                    try:
                        start, end = map(int, part.split('-'))
                        if start < 1 or end < 1:
                            raise ValueError(f"Row numbers must be >= 1, got: {part}")
                        if start > total_rows or end > total_rows:
                            raise ValueError(f"Row numbers must be <= {total_rows}, got: {part}")
                        if start > end:
                            raise ValueError(f"Invalid range {part}: start > end")
                        
                        # Convert to 0-based and add range (end is inclusive in user spec)
                        indices.extend(range(start - 1, end))  # end is inclusive, so we need start-1 to end since range is exclusive at end
                    except ValueError as e:
                        if "invalid literal" in str(e):
                            raise ValueError(f"Invalid range format: {part}")
                        else:
                            raise
                else:
                    # Handle single number like "3"
                    try:
                        row_num = int(part)
                        if row_num < 1:
                            raise ValueError(f"Row numbers must be >= 1, got: {row_num}")
                        if row_num > total_rows:
                            raise ValueError(f"Row number must be <= {total_rows}, got: {row_num}")
                        
                        # Convert to 0-based
                        indices.append(row_num - 1)
                    except ValueError as e:
                        if "invalid literal" in str(e):
                            raise ValueError(f"Invalid row number: {part}")
                        else:
                            raise
            
            # Remove duplicates and sort
            indices = sorted(list(set(indices)))
            return indices
        
        # Use start_row and end_row
        if start_row < 1:
            raise ValueError(f"start_row must be >= 1, got: {start_row}")
        if start_row > total_rows:
            raise ValueError(f"start_row must be <= {total_rows}, got: {start_row}")
        
        actual_end_row = end_row if end_row is not None else total_rows
        if actual_end_row < start_row:
            raise ValueError(f"end_row ({actual_end_row}) must be >= start_row ({start_row})")
        if actual_end_row > total_rows:
            raise ValueError(f"end_row must be <= {total_rows}, got: {actual_end_row}")
        
        # Convert to 0-based indices
        return list(range(start_row - 1, actual_end_row))
    
    def process_with_comparison(self, input_file: str, output_dir: str, 
                              prompt_names: list = None, metadata_columns: list = None,
                              start_row: int = 1, end_row: int = None, rows_spec: str = None):
        """
        Process texts with both standard and enhanced analysis for comparison using concurrent processing.
        
        Args:
            input_file: CSV file with texts to process
            output_dir: Directory to save results
            prompt_names: List of prompts to test (defaults to tabular_analysis)
            metadata_columns: Additional metadata columns to include
            start_row: First row to process (1-based, default: 1)
            end_row: Last row to process (1-based, default: None for all rows)
            rows_spec: Row range specification like "1,3,5-8" (overrides start_row/end_row)
        """
        if prompt_names is None:
            prompt_names = ["tabular_analysis"]
        
        self.logger.info(f"Starting comparison processing with contextual corpus usage: {input_file}")
        
        try:
            # Read input data
            data = []
            with open(input_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                for row in reader:
                    data.append(row)
            
            if not data:
                self.logger.error("No data found in input file")
                return
            
            # Parse row ranges and filter data
            total_rows = len(data)
            try:
                selected_indices = self._parse_row_ranges(
                    rows_spec=rows_spec,
                    start_row=start_row,
                    end_row=end_row,
                    total_rows=total_rows
                )
            except ValueError as e:
                self.logger.error(f"Invalid row range specification: {e}")
                return
            
            # Filter data based on selected indices
            original_data = data
            data = [original_data[i] for i in selected_indices]
            
            # Log row range information
            if rows_spec:
                self.logger.info(f"Processing {len(selected_indices)} selected rows from {total_rows} total rows (rows: {rows_spec})")
            elif end_row:
                self.logger.info(f"Processing rows {start_row}-{min(end_row, total_rows)} from {total_rows} total rows")
            else:
                self.logger.info(f"Processing rows {start_row}-{total_rows} from {total_rows} total rows")
            
            if not data:
                self.logger.error("No rows selected for processing")
                return
            
            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Process each text through the pipeline with updated progress tracking
            for processed_idx, row in enumerate(tqdm(data, desc="Processing texts")):
                # Calculate original index for proper title generation (convert to 1-based for filenames)
                original_idx = selected_indices[processed_idx] + 1
                # Extract title and text
                if 'poemTitle' in row and row['poemTitle'].strip():
                    title = row['poemTitle'].strip().strip('"')
                else:
                    title = f"Text_{original_idx}"
                
                if 'text' not in row or not row['text'].strip():
                    continue
                
                text = row['text'].strip()
                if text.startswith('"') and text.endswith('"'):
                    text = text[1:-1]
                
                # Create metadata - automatically include standard dialectal fields
                metadata = {}
                
                # Automatic inclusion of standard dialectal/geographic fields for better dialect detection
                standard_metadata_fields = ['place_name', 'place_id', 'year', 'display_name']
                for field in standard_metadata_fields:
                    if field in row and row[field]:
                        metadata[field] = row[field]
                
                # Include user-specified metadata columns
                if metadata_columns:
                    for col in metadata_columns:
                        if col in row and row[col]:
                            metadata[col] = row[col]
                
                # STAGE 1: Translation with dialect detection
                self.logger.info(f"🌍 STAGE 1 (Row {original_idx + 1}/{total_rows}): Translating Estonian runosong text: {title}")
                translation, dialect_info = self.translate_text(
                    text=text,
                    title=title,
                    metadata=metadata
                )
                
                if not translation:
                    self.logger.warning(f"Translation failed for: {title}")
                    continue
                
                # STAGE 2: Translation improvement
                self.logger.info(f"✨ STAGE 2: Improving translation quality for: {title}")
                improved_translation = self.improve_translation(
                    text=text,
                    translation=translation,
                    title=title,
                    dialect_info=dialect_info,
                    metadata=metadata
                )
                
                if not improved_translation:
                    self.logger.warning(f"Translation improvement failed for: {title}")
                    improved_translation = translation
                
                # STAGE 3: Analysis with both methods (concurrent processing)
                for prompt_name in prompt_names:
                    self.logger.info(f"🧠 STAGE 3: Running AI analysis (standard + enhanced with Estonian context) for: {title}")
                    
                    # Pre-generate and pre-analyze EstNLTK context for enhanced analysis
                    estnltk_context = None
                    estnltk_formatted_context = None
                    raw_corpus_context = None
                    raw_estnltk_analysis = None
                    optimization_data = None
                    if self.context_enabled:
                        # CRITICAL FIX: Clean corpus BEFORE getting context when force_corpus_update is True
                        normalized_title = self._normalize_source_title(title)
                        title_variants = self._find_title_variants_in_processed_texts(normalized_title)
                        
                        if title_variants and self.force_corpus_update:
                            self.logger.info(f"Text variants found in processed_texts: {title_variants}")
                            self.logger.info(f"Cleaning old corpus entries before analysis (force_corpus_update=True)")
                            
                            # Clean all variants of this title from corpus
                            for variant_title in title_variants:
                                self.logger.info(f"Cleaning corpus entries from source: '{variant_title}'")
                                self._clean_corpus_entries_from_source(variant_title)
                            
                            # Clean processed_texts list by removing all variants
                            processed_texts = self.corpus_database.get("processed_texts", [])
                            original_count = len(processed_texts)
                            cleaned_processed_texts = [t for t in processed_texts if t not in title_variants]
                            self.corpus_database["processed_texts"] = cleaned_processed_texts
                            removed_count = original_count - len(cleaned_processed_texts)
                            if removed_count > 0:
                                self.logger.info(f"Removed {removed_count} title variants from processed_texts list")
                            
                            # Validation: Check that cleaning actually worked
                            self._validate_corpus_cleaning(normalized_title)
                        
                        estnltk_context = self._generate_estnltk_context_cached(text, metadata)
                        # Generate raw EstNLTK analysis for comparison
                        raw_estnltk_analysis = self._generate_raw_estnltk_analysis(text)
                        # Extract raw corpus context for current text words (now clean if force_update was used)
                        current_words = self._extract_words_from_text(text)
                        raw_corpus_context = self._get_corpus_context(current_words)
                        simple_context = self._simple_context_formatting(estnltk_context, text, metadata)
                        estnltk_formatted_context = self._format_simple_context_for_prompt(simple_context)
                        # Extract optimization data for JSON output
                        optimization_data = simple_context.get("optimization_data")
                    
                    # Run standard and enhanced analyses sequentially to avoid API overload
                    # Process standard analysis first
                    standard_analysis = self._process_text_stage(
                        self._analyze_text_chunked,
                        text, title, improved_translation, dialect_info, metadata, prompt_name
                    )
                    
                    # Process enhanced analysis second (if enabled)
                    enhanced_analysis = None
                    if self.context_enabled:
                        enhanced_analysis = self._process_text_stage(
                            self.analyze_text_with_context,
                            text, title, improved_translation, dialect_info, metadata,
                            f"{prompt_name}_with_context", estnltk_context, estnltk_formatted_context
                        )
                        
                        # Update corpus with enhanced analysis results using normalized title
                        if enhanced_analysis:
                            normalized_title = self._normalize_source_title(title)
                            self._update_corpus_from_analysis(enhanced_analysis, normalized_title, text, self.force_corpus_update)
                        
                        if not self.context_enabled:
                            with self._stats_lock:
                                self.enhancement_stats["texts_without_context"] += 1
                    
                    # Save results
                    saved_formatted_context = ""
                    if estnltk_formatted_context:
                        saved_formatted_context = estnltk_formatted_context
                        self.logger.debug(f"💾 Formatted context saved for {title}: {len(saved_formatted_context)} chars")
                    else:
                        self.logger.debug(f"⚠️ No formatted context to save for {title}")
                    
                    result_data = {
                        'title': title,
                        'original_text': text,
                        'initial_translation': translation,
                        'improved_translation': improved_translation,
                        'dialect_info': dialect_info,
                        'standard_analysis': standard_analysis,
                        'enhanced_analysis': enhanced_analysis,
                        'context_available': self.context_enabled,
                        'estnltk_context': estnltk_context,
                        'formatted_context_for_report': saved_formatted_context,
                        'raw_corpus_context': raw_corpus_context,
                        'raw_estnltk_analysis': raw_estnltk_analysis,
                        'token_optimization': optimization_data,
                        'processing_metadata': {
                            'timestamp': datetime.now().isoformat(),
                            'prompt_name': prompt_name,
                            'row_index': original_idx,
                            'version': 'v8_morphology_first_raw_estnltk_optimized'
                        }
                    }
                    
                    # Add metadata fields
                    for col in headers:
                        if col not in ['poemTitle', 'text'] and col in row and row[col]:
                            result_data[col] = row[col]
                    
                    # Save individual result
                    output_file = f"{output_dir}/enhanced_analysis_{prompt_name}_{original_idx:04d}.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(result_data, f, ensure_ascii=False, indent=2)
                    
                    # Generate and save human-readable comparison report
                    comparison_report_file = f"{output_dir}/comparison_report_{prompt_name}_{original_idx:04d}.md"
                    self.save_comparison_report(result_data, comparison_report_file)
                
                # Add adaptive delay between texts based on API performance
                time.sleep(1)
            
            # Save processing statistics
            stats_file = f"{output_dir}/enhancement_statistics.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.enhancement_stats, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Comparison processing with contextual corpus usage completed. Results saved to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error in comparison processing: {str(e)}")
            raise
    
    def get_enhancement_statistics(self) -> dict:
        """Get statistics about the enhancement processing including pre-analysis."""
        return {
            **self.enhancement_stats,
            "context_enabled": self.context_enabled,
            "contextual_corpus_enabled": True,
            "enhancement_rate": (
                self.enhancement_stats["texts_with_context"] / 
                max(1, self.enhancement_stats["texts_with_context"] + self.enhancement_stats["texts_without_context"])
            ) * 100
        }
    
    def _parse_analysis_table(self, analysis_text: str) -> dict:
        """
        Parse a tabular analysis result into structured data.
        
        Args:
            analysis_text: The markdown table from Claude's analysis
            
        Returns:
            Dictionary mapping words to their analysis data
        """
        words_data = {}
        
        if not analysis_text or "|" not in analysis_text:
            return words_data
        
        # Split into lines and find table rows
        lines = analysis_text.strip().split('\n')
        table_lines = [line for line in lines if line.strip().startswith('|') and 'original form' not in line and '---' not in line]
        
        for line in table_lines:
            if not line.strip():
                continue
                
            # Split by | and clean up
            parts = [part.strip() for part in line.split('|')[1:-1]]  # Remove empty first/last parts
            
            if len(parts) >= 6:  # Ensure we have enough columns
                original_form = parts[0]
                comment = parts[1]
                english_translation = parts[2]
                normalized_orthography = parts[3]
                lemma_original = parts[4]
                lemma_modern = parts[5]
                etymological_root = parts[6] if len(parts) > 6 else ""
                
                words_data[original_form] = {
                    "original_form": original_form,
                    "comment": comment,
                    "english_translation": english_translation,
                    "normalized_orthography": normalized_orthography,
                    "lemma_original": lemma_original,
                    "lemma_modern": lemma_modern,
                    "etymological_root": etymological_root
                }
        
        return words_data
    
    def _determine_improvement_type(self, standard_data: dict, enhanced_data: dict, word: str) -> str:
        """
        Determine the type of improvement between standard and enhanced analysis.
        
        Args:
            standard_data: Parsed standard analysis data
            enhanced_data: Parsed enhanced analysis data
            word: The word being compared
            
        Returns:
            String indicating improvement type
        """
        if word not in standard_data or word not in enhanced_data:
            return "missing_data"
        
        std = standard_data[word]
        enh = enhanced_data[word]
        
        # Compare key fields
        lemma_modern_same = std["lemma_modern"].lower().strip() == enh["lemma_modern"].lower().strip()
        lemma_original_same = std["lemma_original"].lower().strip() == enh["lemma_original"].lower().strip()
        etymology_same = std["etymological_root"].lower().strip() == enh["etymological_root"].lower().strip()
        
        # Check for specific improvements
        if lemma_modern_same and lemma_original_same and etymology_same:
            # Check if enhanced has more detail
            if len(enh["comment"]) > len(std["comment"]) * 1.2:
                return "enhanced_detail"
            return "same"
        
        # Check for lemma improvements (Estonian lemmas vs Finnish cognates)
        std_lemma = std["lemma_modern"].lower()
        enh_lemma = enh["lemma_modern"].lower()
        
        # Common cases where Estonian lemmas replace Finnish cognates
        finnish_cognates = ["työ", "kyntö", "sitoa", "tehdä", "varjo", "kammari", "murkaa", "valvoa"]
        estonian_lemmas = ["töö", "künd", "siduma", "tegema", "vari", "kamber", "murdma", "vaatama"]
        
        if any(pattern in std_lemma for pattern in finnish_cognates) and any(pattern in enh_lemma for pattern in estonian_lemmas):
            return "corrected_estonian_lemma"
        
        if not lemma_modern_same:
            # Check if it's a correction or refinement
            if "Estonian" in enh["comment"] and "Finnish" not in enh["comment"]:
                return "corrected"
            return "refined"
        
        if not etymology_same and "Proto-Finnic" in enh["etymological_root"]:
            return "enhanced_etymology"
        
        return "enhanced"
    
    def generate_human_readable_comparison(self, result_data: dict) -> str:
        """
        Generate a human-readable comparison report from analysis results with improved visualization.
        
        Args:
            result_data: The result data from process_with_comparison
            
        Returns:
            Human-readable comparison report as markdown string
        """
        title = result_data.get("title", "Unknown")
        original_text = result_data.get("original_text", "")
        standard_analysis = result_data.get("standard_analysis", "")
        enhanced_analysis = result_data.get("enhanced_analysis", "")
        estnltk_context = result_data.get("estnltk_context", {})
        
        # Use saved formatted context or regenerate from raw context if available
        estnltk_formatted_context = result_data.get("formatted_context_for_report", "")
        if estnltk_formatted_context:
            self.logger.debug(f"📖 Using saved formatted context for report: {len(estnltk_formatted_context)} chars")
        elif estnltk_context and estnltk_context.get("context_available", False):
            # Regenerate formatted context from raw context if available
            self.logger.warning(f"⚠️ Formatted context missing for report, regenerating from raw context")
            simple_context = self._simple_context_formatting(estnltk_context, original_text, None)
            estnltk_formatted_context = self._format_simple_context_for_prompt(simple_context)
        else:
            self.logger.debug(f"ℹ️ No context available for report generation")
        
        # Parse both analyses
        standard_data = self._parse_analysis_table(standard_analysis)
        enhanced_data = self._parse_analysis_table(enhanced_analysis)
        
        # Create EstNLTK lookup for easy access
        estnltk_lookup = {}
        if estnltk_context and estnltk_context.get("context_available", False):
            for word_analysis in estnltk_context.get("word_analyses", []):
                word = word_analysis.get("original_word", "")
                if word:
                    estnltk_lookup[word] = word_analysis
        
        if not standard_data or not enhanced_data:
            return f"# Comparison Report: {title}\n\nError: Could not parse analysis tables."
        
        # Find all words and compare
        all_words = set(standard_data.keys()) | set(enhanced_data.keys())
        
        comparison_data = []
        improvement_counts = {
            "same": 0,
            "enhanced": 0,
            "enhanced_detail": 0,
            "enhanced_etymology": 0,
            "corrected": 0,
            "corrected_estonian_lemma": 0,
            "refined": 0,
            "missing_data": 0
        }
        
        estnltk_agreement_count = 0
        estnltk_disagreement_count = 0
        
        for word in sorted(all_words):
            improvement_type = self._determine_improvement_type(standard_data, enhanced_data, word)
            improvement_counts[improvement_type] += 1
            
            if word in standard_data and word in enhanced_data:
                std = standard_data[word]
                enh = enhanced_data[word]
                
                # Get EstNLTK analysis for this word
                estnltk_data = estnltk_lookup.get(word, {})
                estnltk_lemma = estnltk_data.get("estnltk_lemma", "N/A")
                estnltk_confidence = estnltk_data.get("confidence", "N/A")
                estnltk_method = estnltk_data.get("method", "N/A")
                
                # Check EstNLTK agreement
                estnltk_agrees = "N/A"
                if estnltk_lemma != "N/A" and enh["lemma_modern"]:
                    if estnltk_lemma.lower() == enh["lemma_modern"].lower():
                        estnltk_agrees = "✓"
                        estnltk_agreement_count += 1
                    else:
                        estnltk_agrees = "✗"
                        estnltk_disagreement_count += 1
                
                comparison_data.append({
                    "word": word,
                    "standard_lemma": std["lemma_modern"],
                    "enhanced_lemma": enh["lemma_modern"],
                    "improvement_type": improvement_type,
                    "standard_comment": std["comment"],
                    "enhanced_comment": enh["comment"],
                    "estnltk_lemma": estnltk_lemma,
                    "estnltk_confidence": estnltk_confidence,
                    "estnltk_method": estnltk_method,
                    "estnltk_agrees": estnltk_agrees
                })
        
        # Generate enhanced lemmas dictionary for text overview (filter out punctuation)
        enhanced_lemmas = {}
        original_text = result_data.get("original_text", "")
        
        # Extract words without punctuation (include apostrophes for Estonian dialectal forms)
        clean_words = re.findall(r'[\w´\']+', original_text)
        for word in clean_words:
            if word in enhanced_data:
                enhanced_lemmas[word] = enhanced_data[word].get("lemma_modern", word)
        
        # Generate enhanced text overview
        enhanced_text_overview = self._generate_enhanced_text_overview(original_text, enhanced_lemmas)
        
        # Calculate EstNLTK agreement statistics
        total_estnltk_comparisons = estnltk_agreement_count + estnltk_disagreement_count
        estnltk_agreement_rate = (estnltk_agreement_count / total_estnltk_comparisons * 100) if total_estnltk_comparisons > 0 else 0
        
        report_lines = [
            f"# Analysis Comparison Report: {title}",
            "",
            "## Summary Statistics",
            "",
            f"- **Total words analyzed**: {len(all_words)}",
            f"- **Unchanged**: {improvement_counts['same']} ({improvement_counts['same']/len(all_words)*100:.1f}%)",
            f"- **Enhanced detail**: {improvement_counts['enhanced_detail']} ({improvement_counts['enhanced_detail']/len(all_words)*100:.1f}%)",
            f"- **Enhanced etymology**: {improvement_counts['enhanced_etymology']} ({improvement_counts['enhanced_etymology']/len(all_words)*100:.1f}%)",
            f"- **Improved Estonian lemmas**: {improvement_counts['corrected_estonian_lemma']} ({improvement_counts['corrected_estonian_lemma']/len(all_words)*100:.1f}%)",
            f"- **Other corrections**: {improvement_counts['corrected']} ({improvement_counts['corrected']/len(all_words)*100:.1f}%)",
            f"- **Refined**: {improvement_counts['refined']} ({improvement_counts['refined']/len(all_words)*100:.1f}%)",
            f"- **Other enhancements**: {improvement_counts['enhanced']} ({improvement_counts['enhanced']/len(all_words)*100:.1f}%)",
            "",
            "## EstNLTK Agreement Analysis",
            "",
            f"- **Enhanced analysis agrees with EstNLTK**: {estnltk_agreement_count} words",
            f"- **Enhanced analysis disagrees with EstNLTK**: {estnltk_disagreement_count} words",
            f"- **Agreement rate**: {estnltk_agreement_rate:.1f}% (when EstNLTK provided suggestions)",
            "",
            "## Enhanced Text Overview",
            "",
            "Text with enhanced analysis lemmas in parentheses:",
            "",
            enhanced_text_overview,
            "",
            "## Analysis Differences",
            ""
        ]
        
        # Show only actual lemma differences (filter out cases where lemmas are the same)
        significant_changes = [comp for comp in comparison_data 
                             if comp["standard_lemma"] != comp["enhanced_lemma"]]
        
        if significant_changes:
            report_lines.extend([
                "### Key Differences Between Standard and Enhanced Analysis",
                "",
                "| Word | Standard Lemma | Enhanced Lemma | EstNLTK Lemma | EstNLTK Agreement | Change Type | Standard Comment | Enhanced Comment |",
                "|------|----------------|----------------|---------------|-------------------|-------------|------------------|------------------|"
            ])
            
            for comp in significant_changes[:15]:  # Show top 15
                std_comment = comp["standard_comment"][:25] + "..." if len(comp["standard_comment"]) > 25 else comp["standard_comment"]
                enh_comment = comp["enhanced_comment"][:25] + "..." if len(comp["enhanced_comment"]) > 25 else comp["enhanced_comment"]
                change_type = comp["improvement_type"].replace('_', ' ').title()
                report_lines.append(f"| {comp['word']} | {comp['standard_lemma']} | {comp['enhanced_lemma']} | {comp['estnltk_lemma']} | {comp['estnltk_agrees']} | {change_type} | {std_comment} | {enh_comment} |")
        
        # Add detailed comparison table
        report_lines.extend([
            "",
            "## Complete Word-by-Word Comparison",
            "",
            "| Word | Standard Lemma | Enhanced Lemma | EstNLTK Lemma | EstNLTK Agreement | Change Type | Standard Comment | Enhanced Comment |",
            "|------|----------------|----------------|---------------|-------------------|-------------|------------------|------------------|"
        ])
        
        for comp in comparison_data:
            std_comment = comp["standard_comment"][:40] + "..." if len(comp["standard_comment"]) > 40 else comp["standard_comment"]
            enh_comment = comp["enhanced_comment"][:40] + "..." if len(comp["enhanced_comment"]) > 40 else comp["enhanced_comment"]
            
            report_lines.append(
                f"| {comp['word']} | {comp['standard_lemma']} | {comp['enhanced_lemma']} | "
                f"{comp['estnltk_lemma']} | {comp['estnltk_agrees']} | "
                f"{comp['improvement_type'].replace('_', ' ').title()} | {std_comment} | {enh_comment} |"
            )
        
        # Add EstNLTK context section if available
        if estnltk_formatted_context:
            report_lines.extend([
                "",
                "## EstNLTK Context Provided to Enhanced Analysis",
                "",
                "The following is the exact pre-analyzed EstNLTK+dictionary context that was provided to Claude for the enhanced analysis:",
                "",
                "```",
                estnltk_formatted_context.strip(),
                "```"
            ])
        
        return "\n".join(report_lines)
    
    def _generate_enhanced_text_overview(self, original_text: str, enhanced_lemmas: dict) -> str:
        """
        Generate text overview with enhanced lemmas in parentheses, filtering out punctuation.
        
        Args:
            original_text: The original runosong text
            enhanced_lemmas: Dictionary mapping words to their enhanced lemmas
            
        Returns:
            Text string with enhanced lemmas in parentheses, without punctuation
        """
        if not original_text:
            return ""
        
        # Extract only words (include apostrophes for Estonian dialectal forms) using regex
        words = re.findall(r'[\w´\']+', original_text)
        
        # Create overview with lemmas in parentheses
        overview_parts = []
        for word in words:
            enhanced_lemma = enhanced_lemmas.get(word, word)
            overview_parts.append(f"{word} ({enhanced_lemma})")
        
        return " ".join(overview_parts)
    
    def save_comparison_report(self, result_data: dict, output_file: str) -> None:
        """
        Save a human-readable comparison report to file.
        
        Args:
            result_data: The result data from process_with_comparison
            output_file: Path to save the comparison report
        """
        report = self.generate_human_readable_comparison(result_data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"Comparison report saved to {output_file}")

    def _load_corpus_database(self) -> dict:
        """Load the corpus database from JSON file."""
        try:
            if os.path.exists(self.corpus_file):
                with open(self.corpus_file, 'r', encoding='utf-8') as f:
                    corpus_data = json.load(f)
                # Handle both old and new corpus database formats
                if isinstance(corpus_data, dict) and "words" in corpus_data:
                    # New format with structure
                    word_count = len(corpus_data.get("words", {}))
                    processed_count = len(corpus_data.get("processed_texts", []))
                    self.logger.info(f"Loaded corpus database with {word_count} words and {processed_count} processed texts")
                else:
                    # Old format - just words dict, migrate to new format
                    old_words = corpus_data if isinstance(corpus_data, dict) else {}
                    corpus_data = {
                        "words": old_words,
                        "processed_texts": [],
                        "metadata": {
                            "created": datetime.now().strftime("%Y-%m-%d"),
                            "version": "v6",
                            "migrated_from": "v5_old_format"
                        }
                    }
                    self.logger.info(f"Migrated old corpus format with {len(old_words)} words to new structure")
                return corpus_data
            else:
                self.logger.info("No existing corpus database found, starting with empty corpus")
                return {}
        except Exception as e:
            self.logger.error(f"Error loading corpus database: {e}")
            return {}

    def _save_corpus_database(self):
        """Save the corpus database to JSON file."""
        try:
            with open(self.corpus_file, 'w', encoding='utf-8') as f:
                json.dump(self.corpus_database, f, ensure_ascii=False, indent=2)
            word_count = len(self.corpus_database.get("words", {}))
            text_count = len(self.corpus_database.get("processed_texts", []))
            self.logger.info(f"Saved corpus database with {word_count} words from {text_count} texts")
        except Exception as e:
            self.logger.error(f"Error saving corpus database: {e}")

    def _normalize_source_title(self, title: str) -> str:
        """Normalize source title to prevent duplicate entries in corpus.
        
        Args:
            title: Original title which may include metadata like "ars0104990010 - Setumaa (1927)"
            
        Returns:
            Normalized title like "ars0104990010" for consistent corpus operations
        """
        if not title:
            return title
        
        # Extract base poem ID by taking the part before the first " - " separator
        # This handles cases like "ars0104990010 - Setumaa (1927)" -> "ars0104990010"
        normalized = title.split(' - ')[0].strip()
        
        return normalized

    def _find_title_variants_in_processed_texts(self, target_title: str) -> list:
        """Find all variants of a title in processed_texts list.
        
        Handles both directions:
        - Given normalized title, find unnormalized variants
        - Given unnormalized title, find normalized variants
        
        Args:
            target_title: Title to find variants for
            
        Returns:
            List of all matching title variants found in processed_texts
        """
        processed_texts = self.corpus_database.get("processed_texts", [])
        found_variants = []
        
        normalized_target = self._normalize_source_title(target_title)
        
        for processed_title in processed_texts:
            normalized_processed = self._normalize_source_title(processed_title)
            
            # Match if normalized forms are the same
            if normalized_target == normalized_processed:
                found_variants.append(processed_title)
                
        return found_variants

    def _validate_corpus_cleaning(self, normalized_title: str):
        """Validate that corpus cleaning worked properly by checking for remaining source variants."""
        words_db = self.corpus_database.get("words", {})
        remaining_sources = set()
        words_with_variants = []
        
        # Check all word entries for any remaining source variants
        for word, entry in words_db.items():
            sources = entry.get("sources", [])
            for source in sources:
                if self._normalize_source_title(source) == normalized_title:
                    remaining_sources.add(source)
                    words_with_variants.append(word)
        
        if remaining_sources:
            self.logger.warning(f"CLEANING VALIDATION FAILED: Found {len(remaining_sources)} remaining source variants after cleaning:")
            for source in remaining_sources:
                self.logger.warning(f"  - Remaining source: '{source}'")
            self.logger.warning(f"Words still containing these sources: {words_with_variants[:5]}...")  # Show first 5
        else:
            self.logger.info(f"✓ CLEANING VALIDATION PASSED: No source variants found for '{normalized_title}'")

    def _get_corpus_context(self, current_words: list) -> dict:
        """Simple lookup - find words in corpus and return raw entries."""
        relevant_entries = {}
        lookup_count = 0
        
        words_db = self.corpus_database.get("words", {})
        for word in current_words:
            if word in words_db:
                relevant_entries[word] = words_db[word]
                lookup_count += 1
        
        self.logger.info(f"Found corpus entries for {len(relevant_entries)} words out of {len(current_words)}")
        return relevant_entries

    def _preprocess_corpus_context(self, raw_corpus_context: dict) -> dict:
        """
        Preprocess corpus context to remove unnecessary fields and limit contexts.
        
        Args:
            raw_corpus_context: Raw corpus entries from _get_corpus_context()
        
        Returns:
            Filtered corpus entries with only essential fields and limited contexts
        """
        processed_entries = {}
        
        for word, entry in raw_corpus_context.items():
            # Keep only essential fields, remove metadata
            processed_entry = {}
            
            # Always include lemmas and lemma_counts
            if "lemmas" in entry:
                processed_entry["lemmas"] = entry["lemmas"]
            if "lemma_counts" in entry:
                processed_entry["lemma_counts"] = entry["lemma_counts"]
            
            # Include translations if available
            if "translations" in entry:
                processed_entry["translations"] = entry["translations"]
            
            # Limit contexts to maximum 3, select most diverse ones
            if "contexts" in entry and entry["contexts"]:
                contexts = entry["contexts"]
                if len(contexts) <= 3:
                    processed_entry["contexts"] = contexts
                else:
                    # Select 3 most diverse contexts (avoid duplicates, prefer different lengths)
                    selected_contexts = []
                    used_texts = set()
                    
                    # First pass: select contexts with different text content
                    for context in contexts:
                        # Extract the key part of context (without the bracketed word)
                        context_key = context.replace(f"[{word}]", "").strip()
                        if context_key not in used_texts and len(selected_contexts) < 3:
                            selected_contexts.append(context)
                            used_texts.add(context_key)
                    
                    # If we still need more contexts, add remaining ones
                    for context in contexts:
                        if context not in selected_contexts and len(selected_contexts) < 3:
                            selected_contexts.append(context)
                    
                    processed_entry["contexts"] = selected_contexts[:3]
            
            processed_entries[word] = processed_entry
        
        self.logger.info(f"Preprocessed corpus context: removed sources/first_seen fields, limited contexts to max 3 per word")
        return processed_entries

    def _clean_corpus_entries_from_source(self, source_title: str):
        """Remove all corpus entries that came from the specified source and its variants.
        
        This function now handles both normalized and unnormalized source variants.
        For example, if source_title is "ars0104990010", it will also clean entries
        from "ars0104990010 - Setumaa (1927)" and vice versa.
        """
        if "words" not in self.corpus_database:
            return
        
        words_db = self.corpus_database["words"]
        words_to_remove = []
        cleaned_count = 0
        sources_removed_total = 0
        
        # Find all source variants that should be cleaned
        normalized_target = self._normalize_source_title(source_title)
        
        # Collect all sources from all word entries to find variants
        all_sources_in_corpus = set()
        for entry in words_db.values():
            all_sources_in_corpus.update(entry.get("sources", []))
        
        # Find all source variants that match our target
        sources_to_clean = set()
        for corpus_source in all_sources_in_corpus:
            if self._normalize_source_title(corpus_source) == normalized_target:
                sources_to_clean.add(corpus_source)
        
        self.logger.info(f"Found {len(sources_to_clean)} source variants to clean: {list(sources_to_clean)}")
        
        for word, entry in words_db.items():
            original_sources = entry.get("sources", []).copy()
            sources_found_for_word = []
            
            # Remove ALL matching source variants from this word entry
            for source_to_clean in sources_to_clean:
                if source_to_clean in entry.get("sources", []):
                    sources_found_for_word.append(source_to_clean)
                    entry["sources"] = [s for s in entry["sources"] if s != source_to_clean]
                    sources_removed_total += 1
            
            # If any sources were removed from this word
            if sources_found_for_word:
                # If this was the only source(s), remove the word entirely
                if not entry["sources"]:
                    words_to_remove.append(word)
                    cleaned_count += 1
                    self.logger.debug(f"Word '{word}' removed entirely (no sources remaining)")
                else:
                    # Reduce lemma counts proportionally (this is an approximation)
                    # In practice, we'll reset counts since we can't know which came from which source
                    for lemma in list(entry["lemma_counts"].keys()):
                        entry["lemma_counts"][lemma] = max(1, entry["lemma_counts"][lemma] - 1)
                    
                    # Remove contexts and translations that might be from this source
                    # (Conservative approach: keep them since we can't be sure which came from where)
                    cleaned_count += 1
                    self.logger.debug(f"Word '{word}' cleaned, removed sources: {sources_found_for_word}")
        
        # Remove words that have no sources left
        for word in words_to_remove:
            del words_db[word]
        
        if cleaned_count > 0:
            self.logger.info(f"Cleaned {cleaned_count} corpus word entries, removed {sources_removed_total} source references total")
        else:
            self.logger.info(f"No corpus entries found to clean for source '{source_title}'")

    def _update_corpus_from_analysis(self, analysis_text: str, source_title: str, original_text: str = "", force_update: bool = False):
        """Update corpus database with results from enhanced analysis.
        
        Args:
            analysis_text: The markdown table from enhanced analysis
            source_title: Title of the source text
            original_text: The original runosong text for context extraction
            force_update: If True, update even if text was already processed
        """
        try:
            if not analysis_text or "|" not in analysis_text:
                return
            
            # Note: Cleaning is now handled before context generation in the main processing loop
            # This ensures context is clean when enhanced analysis runs
            
            # Parse the markdown table
            lines = analysis_text.strip().split('\n')
            table_lines = [line for line in lines if line.strip().startswith('|') and 'original form' not in line and '---' not in line]
            
            update_count = 0
            processed_in_this_run = set()  # Track words processed in this analysis to avoid duplicates
            for line in table_lines:
                if not line.strip():
                    continue
                    
                # Split by | and clean up
                parts = [part.strip() for part in line.split('|')[1:-1]]  # Remove empty first/last parts
                
                if len(parts) >= 6:  # Ensure we have enough columns
                    original_form = parts[0]
                    english_translation = parts[2]
                    lemma_modern = parts[5]
                    
                    if original_form and lemma_modern:
                        # Skip if we've already processed this word+lemma combination in this analysis
                        word_lemma_key = f"{original_form}|{lemma_modern}"
                        if word_lemma_key in processed_in_this_run:
                            continue
                        processed_in_this_run.add(word_lemma_key)
                        
                        # Extract proper context from original text
                        context = self._extract_word_context(original_form, original_text)
                        self._update_corpus_entry(original_form, lemma_modern, english_translation, context, source_title)
                        update_count += 1
            
            # Add to processed texts list if not already there
            if source_title not in self.corpus_database.get("processed_texts", []):
                self.corpus_database.setdefault("processed_texts", []).append(source_title)
            
            self.logger.info(f"Updated corpus with {update_count} entries from {source_title}")
            self._save_corpus_database()
            
        except Exception as e:
            self.logger.error(f"Error updating corpus from analysis: {e}")

    def _update_corpus_entry(self, word: str, lemma: str, translation: str, context: str, source: str):
        """Update corpus with new analysis result."""
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Ensure words section exists
        if "words" not in self.corpus_database:
            self.corpus_database["words"] = {}
        
        words_db = self.corpus_database["words"]
        
        if word not in words_db:
            # New word entry
            words_db[word] = {
                "lemmas": [lemma],
                "lemma_counts": {lemma: 1},
                "translations": [translation] if translation else [],
                "contexts": [context] if context else [],
                "sources": [source] if source else [],
                "first_seen": today
            }
        else:
            # Update existing entry
            entry = words_db[word]
            
            # Update lemma tracking
            if lemma not in entry["lemmas"]:
                entry["lemmas"].append(lemma)
                entry["lemma_counts"][lemma] = 1
            else:
                entry["lemma_counts"][lemma] += 1
            
            # Update other fields (avoid duplicates)
            if translation and translation not in entry["translations"]:
                entry["translations"].append(translation)
            if context and context not in entry.get("contexts", []):
                if "contexts" not in entry:
                    entry["contexts"] = []
                entry["contexts"].append(context)
            if source and source not in entry["sources"]:
                entry["sources"].append(source)

    def _extract_word_context(self, word: str, original_text: str, context_window: int = 3) -> str:
        """Extract meaningful context around a word from original runosong text.
        
        Args:
            word: The target word to find context for
            original_text: The original runosong text
            context_window: Maximum number of words to show before/after target (default: 3)
            
        Returns:
            Context string with target word in brackets, e.g. "word1 word2 [target] word3 word4"
            Shows up to context_window words before and after, limited by text length.
        """
        if not original_text:
            return ""
        
        # Split text into words
        words = self._extract_words_from_text(original_text)
        
        # Find the word in the text (case-insensitive)
        word_lower = word.lower()
        for i, text_word in enumerate(words):
            if text_word.lower() == word_lower:
                # Extract context window around the word
                start = max(0, i - context_window)
                end = min(len(words), i + context_window + 1)
                context_words = words[start:end]
                
                # Highlight the target word with brackets
                if i - start < len(context_words):
                    context_words[i - start] = f"[{context_words[i - start]}]"
                
                return " ".join(context_words)
        
        # If word not found, return first few words as fallback
        return " ".join(words[:context_window]) if words else ""

    def _extract_words_from_text(self, text: str) -> list:
        """Extract words from text, preserving word boundaries for dialectal forms."""
        import re
        # Split by whitespace first to preserve word boundaries
        # This prevents EstNLTK from merging words like "Koolõ' õks" → "Koolõ'õks"
        raw_words = text.split()
        
        # Clean each word but preserve the core word
        cleaned_words = []
        for word in raw_words:
            # Remove leading/trailing punctuation but keep internal apostrophes/hyphens
            cleaned = re.sub(r'^[^\w\'´`-]+|[^\w\'´`-]+$', '', word)
            if cleaned and not re.match(r'^[^\w]+$', cleaned):  # Skip pure punctuation
                cleaned_words.append(cleaned)
        
        return cleaned_words


def main():
    """Command-line interface for Enhanced Estonian Prompt Tester v6."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Enhanced Estonian Prompt Tester v6 with Prompt Caching and Contextual Corpus Usage'
    )
    parser.add_argument('--annotation-dir', 
                       help='Directory containing annotation files for EstNLTK context')
    parser.add_argument('--corpus-file',
                       help='Corpus database JSON file (default: corpus_database.json)')
    parser.add_argument('--api-key',
                       help='Anthropic API key (or set ANTHROPIC_API_KEY env var)')
    parser.add_argument('--input', help='Input CSV file with texts')
    parser.add_argument('--output-dir', help='Output directory for results')
    parser.add_argument('--prompts', nargs='+', default=['tabular_analysis'],
                       help='Analysis prompts to test')
    parser.add_argument('--metadata-columns', nargs='*',
                       help='Additional metadata columns to include')
    parser.add_argument('--comparison-only', nargs='+',
                       help='Generate comparison report from existing JSON result file(s)')
    parser.add_argument('--setup', action='store_true', help='Run configuration setup')
    parser.add_argument('--force-corpus-update', action='store_true', 
                       help='Force corpus update even for already processed texts')
    
    # Row range selection arguments
    parser.add_argument('--start-row', type=int, default=1,
                       help='First row to process (1-based indexing, default: 1)')
    parser.add_argument('--end-row', type=int,
                       help='Last row to process (1-based indexing, default: process all rows)')
    parser.add_argument('--rows', type=str,
                       help='Row range specification like "5-10" or "1,3,5-8" (overrides start-row/end-row)')
    
    args = parser.parse_args()
    
    # Handle comparison-only mode
    if args.comparison_only:
        import glob
        import re
        
        # Expand glob patterns and collect all files
        all_files = []
        for pattern in args.comparison_only:
            if '*' in pattern or '?' in pattern:
                # This is a glob pattern
                matches = glob.glob(pattern)
                all_files.extend(matches)
            else:
                # Regular file path
                all_files.append(pattern)
        
        if not all_files:
            print(f"❌ No files found matching patterns: {args.comparison_only}")
            return 1
        
        print(f"📁 Found {len(all_files)} JSON file(s) to process")
        
        # Process each file
        for json_file in all_files:
            try:
                print(f"\n🔄 Processing: {json_file}")
                
                # Load existing JSON result
                with open(json_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                try:
                    result_data = json.loads(content)
                except json.JSONDecodeError as e:
                    print(f"❌ JSON parsing error: {e}")
                    continue
                
                # Initialize a minimal tester for comparison generation (no API key needed)
                tester = EnhancedEstonianPromptTesterV8(api_key="dummy_key_for_comparison_only", annotation_dir=None)
                
                # Generate comparison report
                report_file = json_file.replace('.json', '_comparison_report_v7.md')
                tester.save_comparison_report(result_data, report_file)
                
                print(f"✅ Comparison report generated: {report_file}")
                
            except FileNotFoundError:
                print(f"❌ File not found: {json_file}")
                continue
            except Exception as e:
                print(f"❌ Failed to generate comparison report for {json_file}: {e}")
                continue
        
        return 0
    
    # Handle setup
    if args.setup:
        config_manager = ConfigManager()
        config_manager.setup_wizard()
        return 0
    
    # Validate required arguments for normal processing
    if not args.input or not args.output_dir:
        print("❌ Error: --input and --output-dir are required for processing (or use --comparison-only)")
        parser.print_help()
        return 1
    
    # Get configuration
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    # Get API key
    api_key = args.api_key or config_manager.get_api_key()
    if not api_key:
        print("🔑 No API key found. Running setup...")
        api_key = config_manager.ensure_api_key()
    
    # Get annotation directory (now optional)
    annotation_dir = args.annotation_dir or config.get('annotation_dir')
    
    # Check if context enhancement is possible
    main_index_exists = Path("lemma_variant_index.json").exists()
    annotation_index_exists = annotation_dir and Path(annotation_dir) / "lemma_variant_index.json" if annotation_dir else False
    context_available = main_index_exists or annotation_index_exists
    
    print(f"📁 Annotation directory: {annotation_dir if annotation_dir else 'Not specified (using main directory)'}")
    print(f"🔑 API key: {'✅ Found' if api_key else '❌ Missing'}")
    print(f"🧠 Context enhancement: {'✅ Enabled' if context_available else '❌ Disabled'}")
    print(f"🔍 Context formatting: {'✅ Enabled' if context_available else '❌ Disabled'}")
    print(f"🔄 Corpus usage: {'✅ Contextual guidance' if context_available else '❌ Disabled'}")
    print(f"🚀 Version: Enhanced Estonian Prompt Tester v8 with Morphology-First Analysis and Contextual Corpus Usage")
    print(f"📊 Processing workflow: Translation → Quality improvement → Estonian context preparation → AI analysis → Comparison report")
    
    # Initialize enhanced Estonian prompt tester v7
    try:
        tester = EnhancedEstonianPromptTesterV8(
            api_key=api_key,
            annotation_dir=annotation_dir,
            corpus_file=args.corpus_file,
            force_corpus_update=args.force_corpus_update
        )
    except Exception as e:
        print(f"Failed to initialize enhanced tester v6: {e}")
        return 1
    
    # Process with comparison
    try:
        tester.process_with_comparison(
            input_file=args.input,
            output_dir=args.output_dir,
            prompt_names=args.prompts,
            metadata_columns=args.metadata_columns,
            start_row=args.start_row,
            end_row=args.end_row,
            rows_spec=args.rows
        )
        
        print(f"✅ Processing completed. Results saved to {args.output_dir}")
        print(f"📄 Generated files:")
        print(f"  - JSON analysis files: enhanced_analysis_*.json")
        print(f"  - Human-readable comparison reports: comparison_report_*.md")
        print(f"  - Processing statistics: enhancement_statistics.json")
        
        # Print statistics
        stats = tester.get_enhancement_statistics()
        print(f"\n📊 Enhancement Statistics:")
        print(f"Texts with context: {stats['texts_with_context']}")
        print(f"Texts without context: {stats['texts_without_context']}")
        print(f"Enhancement rate: {stats['enhancement_rate']:.1f}%")
        print(f"Context generation time: {stats['context_generation_time']:.2f}s")
        print(f"Pre-analysis time: {stats['pre_analysis_time']:.2f}s")
        print(f"Enhanced processing time: {stats['enhanced_processing_time']:.2f}s")
        print(f"Context cache hits: {stats['cache_hits']}")
        print(f"Context cache misses: {stats['cache_misses']}")
        print(f"Pre-analysis calls: {stats['pre_analysis_calls']}")
        print(f"Pre-analysis cache hits: {stats['pre_analysis_cache_hits']}")
        print(f"Pre-analysis cache misses: {stats['pre_analysis_cache_misses']}")
        
        print(f"\n📋 To generate additional comparison reports:")
        print(f"python enhanced_comb_3_v6.py --comparison-only {args.output_dir}/enhanced_analysis_*.json")
        
    except Exception as e:
        print(f"Processing failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())