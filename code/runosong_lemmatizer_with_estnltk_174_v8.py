#!/usr/bin/env python3
"""
Estonian Runosong Lemmatizer with EstNLTK 1.7.4 v8 


"""

import json
import argparse
import logging
import re
import time
import math
from pathlib import Path
from collections import defaultdict

# EstNLTK imports
from estnltk import Text
from estnltk.taggers import VabamorfTagger

# EstNLTK 1.7.4 enhanced taggers
try:
    from estnltk.taggers import CompoundWordTagger, MorphExtendedTagger
    ESTNLTK_174_FEATURES = True
except ImportError:
    ESTNLTK_174_FEATURES = False
    logging.warning("EstNLTK 1.7.4 enhanced features not available")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class RunosongLemmatizerWithEstNLTK174:
    """An enhanced lemmatizer for Estonian runosong texts with EstNLTK 1.7.4 features."""

    def __init__(self, annotation_dir, levenshtein_threshold=2, enable_174_features=True):
        self.annotation_dir = Path(annotation_dir)
        self.levenshtein_threshold = levenshtein_threshold
        self.lemma_to_variants = {}
        self.variant_to_lemmas = defaultdict(list)
        self.enable_174_features = enable_174_features and ESTNLTK_174_FEATURES
        self._load_data()

        self.vowels = set('aeiouüõöä')
        self.stats = defaultdict(int)
        
        # Initialize EstNLTK 1.7.4 taggers if available
        self._init_174_taggers()
        
        # Hardcoded known lemmatizations for common/problematic words
        self.known_lemmatizations = {}

    def _init_174_taggers(self):
        """Initialize EstNLTK 1.7.4 enhanced taggers."""
        self.compound_tagger = None
        self.morph_extended_tagger = None
        
        if self.enable_174_features:
            try:
                self.compound_tagger = CompoundWordTagger()
                logger.info("CompoundWordTagger initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize CompoundWordTagger: {e}")
            
            try:
                self.morph_extended_tagger = MorphExtendedTagger()
                logger.info("MorphExtendedTagger initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize MorphExtendedTagger: {e}")

    def _load_data(self):
        start_time = time.time()
        
        # Try loading from main directory first (better approach)
        main_index_path = Path("lemma_variant_index.json")
        annotation_index_path = self.annotation_dir / "lemma_variant_index.json"
        
        index_path = None
        if main_index_path.exists():
            index_path = main_index_path
            logger.info(f"Loading data from main directory: {main_index_path}")
        elif annotation_index_path.exists():
            index_path = annotation_index_path
            logger.info(f"Loading data from annotation directory: {annotation_index_path}")
        else:
            logger.warning(f"Lemma variant index not found in either main directory ({main_index_path}) or annotation directory ({annotation_index_path})")
            return
            
        # Load the data
        with open(index_path, 'r', encoding='utf-8') as f:
            self.lemma_to_variants = json.load(f)
        for lemma, variants in self.lemma_to_variants.items():
            for variant in variants:
                self.variant_to_lemmas[variant.lower().strip()].append(lemma.lower())
                
        logger.info(f"Loaded {len(self.lemma_to_variants)} lemmas with {len(self.variant_to_lemmas)} unique variants in {time.time()-start_time:.2f}s")

    def _levenshtein_distance(self, s1, s2):
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def _sort_lemmas_by_similarity(self, word, lemmas):
        distances = [(lemma, self._levenshtein_distance(word.lower(), lemma.lower())) for lemma in lemmas]
        return [lemma for lemma, _ in sorted(distances, key=lambda x: x[1])]

    def _get_estnltk_analysis_for_lemma(self, lemma_word):
        text_obj = Text(lemma_word.lower())
        text_obj.tag_layer(['words', 'sentences', 'morph_analysis'])
        if text_obj['morph_analysis']:
            analysis = text_obj['morph_analysis'][0].annotations[0]
            lemma = analysis.get('lemma', '').lower()
            pos = analysis.get('partofspeech', '')
            form = analysis.get('form', '').replace(' ', '_')
            return lemma, pos, form
        return lemma_word.lower(), '', ''

    def _get_compound_analysis(self, text_obj):
        """Get compound word analysis using EstNLTK 1.7.4 CompoundWordTagger."""
        if not self.compound_tagger:
            return {}
        
        try:
            self.compound_tagger.tag(text_obj)
            compound_info = {}
            
            if 'compound_words' in text_obj.layers:
                for i, word in enumerate(text_obj.words):
                    word_text = word.text.lower()
                    compound_info[word_text] = []
                    
                    # Find compound annotations for this word
                    for compound_span in text_obj.compound_words:
                        if compound_span.start <= word.start < compound_span.end:
                            compound_data = {
                                'type': compound_span.annotations[0].get('type', ''),
                                'components': compound_span.annotations[0].get('components', [])
                            }
                            compound_info[word_text].append(compound_data)
            
            return compound_info
        except Exception as e:
            logger.warning(f"Compound analysis failed: {e}")
            return {}

    def _get_extended_morph_analysis(self, text_obj):
        """Get extended morphological analysis using EstNLTK 1.7.4 MorphExtendedTagger."""
        if not self.morph_extended_tagger:
            return {}
        
        try:
            self.morph_extended_tagger.tag(text_obj)
            extended_info = {}
            
            if 'morph_extended' in text_obj.layers:
                for i, word in enumerate(text_obj.words):
                    word_text = word.text.lower()
                    extended_info[word_text] = []
                    
                    # Find extended morphological annotations for this word
                    if i < len(text_obj.morph_extended):
                        for annotation in text_obj.morph_extended[i].annotations:
                            extended_data = {
                                'lemma': annotation.get('lemma', ''),
                                'pos': annotation.get('partofspeech', ''),
                                'form': annotation.get('form', ''),
                                'clitic': annotation.get('clitic', ''),
                                'extended_pos': annotation.get('extended_pos', ''),
                                'extended_form': annotation.get('extended_form', '')
                            }
                            extended_info[word_text].append(extended_data)
            
            return extended_info
        except Exception as e:
            logger.warning(f"Extended morphological analysis failed: {e}")
            return {}

    def _try_h_variation(self, word):
        word_l = word.lower()
        candidate = None
        if word_l and word_l[0] in self.vowels:
            candidate = 'h' + word_l
        elif word_l.startswith('h') and len(word_l) > 1:
            candidate = word_l[1:]
        
        if candidate and candidate in self.variant_to_lemmas:
            return self.variant_to_lemmas[candidate][0]
        return None

    def _try_suffix_strip(self, word):
        # Enhanced suffix stripping with more dialectal suffixes
        suffixes = [
         "käinen", "käine",  "käne", "känä", 
         
         'kõnõ', 'keine', 'kane', 'kõne', 
            'kõine', 'kõni', 'kenne', 'kenä', 'kanõ', 'kono', 'kõno',
            

            'kesed', 'kest', 'kesta', 'kese', 'keste', 'kesi', 'keise',
            'kõnõsed', 'kõnõst', 'kõnõse', 'kõnõste', 'kõnõsi', 'kõnõsse'
         
         "kõsõ", "kesta", "kõstä",
            "kesel", "kestel", "kõsõd", "keised", "keise", "dessa", "desse",
            "tesse", "delle", "nuksin", "nuksid", "nuksime", "nuksite",
            "maie", "maje", "maiõ", "aasse", "eesse", "eeksi", "aaksi",
            "eeida", "aaida", "vada", "jada", "aksi", "eksi", "kene", "kane",
            "kese", "kest", "kesi", "kesed", "keste", "essa", "ssa", "jad",
            "ksi", "sed", "and", "lle", "sta", "lla", "llä", "ida", "nut",
            "tije", "lik", "tus", "lene", "dele", "tele", "delt", "telt",
            "deks", "teks", "deni", "teni", "dena", "tena", "deta", "teta",
            "dega", "tega", "dest", "test", "del", "tel", "des", "tes",
            "vad", "sin", "sid", "sime", "site", "nud", "ksin", "ksid",
            "ksime", "ksite", "takse", "gem", "vat", "tav", "mata", "mast",
            "mas", "maks", "kse",
            "da", "ta", "ga", "le", "lt", "st", "ni", "ks", "gi", "ki",
            "kene", "kese", "ke", "maie", "ie", "i", "u", "o", "e", "a",
            # Additional dialectal suffixes for runosongs
            "nõ", "õ", "ma", "mi", "se", "si", "de", "dõ"
        ]
        word_l = word.lower()

        for suffix in suffixes:
            if word_l.endswith(suffix) and len(word_l) > len(suffix):
                stem = word_l[:-len(suffix)]
                if stem in self.variant_to_lemmas:
                    # Prioritize exact stem match, otherwise take first lemma of variant
                    if stem in self.lemma_to_variants:
                        return stem # Return the stem itself as the lemma
                    else:
                        return self.variant_to_lemmas[stem][0]
        return None

    def _try_compound_split_enhanced(self, word, compound_info):
        """Enhanced compound splitting using EstNLTK 1.7.4 compound detection."""
        word_l = word.lower()
        
        # First try EstNLTK 1.7.4 compound detection
        if compound_info and word_l in compound_info:
            for compound_data in compound_info[word_l]:
                components = compound_data.get('components', [])
                if len(components) >= 2:
                    # Use EstNLTK detected components
                    component_lemmas = []
                    for component in components:
                        if component in self.variant_to_lemmas:
                            component_lemmas.append(self.variant_to_lemmas[component][0])
                        else:
                            component_lemmas.append(component)
                    if len(component_lemmas) >= 2:
                        return "+".join(component_lemmas)
        
        # Fallback to original compound splitting logic
        return self._try_compound_split(word)

    def _try_compound_split(self, word):
        # Original compound splitting logic (fallback)
        if len(word) <= 6:
            return None
            
        word_l = word.lower()
        for i in range(len(word_l) - 3, 3, -1):
            left, right = word_l[:i], word_l[i:]
            if len(left) >= 3 and len(right) >= 3 and \
               left in self.variant_to_lemmas and right in self.variant_to_lemmas:
                left_lemmas = self._sort_lemmas_by_similarity(left, self.variant_to_lemmas[left])
                right_lemmas = self._sort_lemmas_by_similarity(right, self.variant_to_lemmas[right])
                if left_lemmas and right_lemmas:
                    return f"{left_lemmas[0]}+{right_lemmas[0]}"
        return None

    def _try_levenshtein(self, word):
        word_l = word.lower()
        if not word_l or len(word_l) < 3:
            return None, 0, []
        
        best_dist = self.levenshtein_threshold + 1
        best_variants = []

        for variant in self.variant_to_lemmas.keys():
            dist = self._levenshtein_distance(word_l, variant)
            if dist < best_dist:
                best_dist = dist
                best_variants = [variant]
            elif dist == best_dist:
                best_variants.append(variant)
        
        if best_variants and best_dist <= self.levenshtein_threshold:
            candidate_lemmas = []
            for variant in best_variants:
                candidate_lemmas.extend(self.variant_to_lemmas[variant])
            
            sorted_candidate_lemmas = self._sort_lemmas_by_similarity(word, list(set(candidate_lemmas)))
            
            if sorted_candidate_lemmas:
                return sorted_candidate_lemmas[0], best_dist, sorted_candidate_lemmas[1:]
        return None, 0, []

    def _validate_compound_candidate(self, components):
        """Use provided diminutive knowledge to filter false compounds (v8 enhancement)"""
        if len(components) != 2:
            return False
        
        first, second = components
        
        # Use provided diminutive suffix knowledge (targeted application)
        diminutive_suffixes = {
            # Base forms from research
            'kene', 'kõnõ', 'keine', 'kane', 'kõne', 'käne', 'ken',
            'kõine', 'kõni', 'kenne', 'kenä', 'kanõ', 'kono', 'kõno',
            
            # Inflected forms from research  
            'kesed', 'kest', 'kesta', 'kese', 'keste', 'kesi', 'keise',
            'kõnõsed', 'kõnõst', 'kõnõse', 'kõnõste', 'kõnõsi', 'kõnõsse'
        }
        
        # Reject if second component is diminutive suffix
        if second.lower() in diminutive_suffixes:
            return False
        
        # Use known problematic stems from research
        known_diminutive_stems = {
            'neitsi', 'velle', 'noore', 'ema', 'eide', 'peiu', 'venna', 
            'vele', 'enne', 'neiu', 'linnu', 'poisi', 'memme', 'nooru', 
            'taadi', 'emä', 'tillu', 'naisu'
        }
        
        # Reject if first component is known diminutive stem
        if first.lower() in known_diminutive_stems:
            return False
        
        # Also check for other obvious morphological patterns
        other_morphological_suffixes = ['mähe', 'mahe', 'edu', 'ehe', 'ist']
        if second.lower() in other_morphological_suffixes:
            return False
        
        return True  # Allow as potential compound

    def lemmatize_word(self, word, disamb_analysis=None, compound_info=None, extended_info=None):
        self.stats["total_words"] += 1
        word_lower = word.lower()

        estnltk_lemma = disamb_analysis.get('lemma', '') if disamb_analysis else ''
        estnltk_pos = disamb_analysis.get('partofspeech', '') if disamb_analysis else ''
        estnltk_form = disamb_analysis.get('form', '').replace(' ', '_') if disamb_analysis else ''
        
        # Check for enhanced features
        if extended_info and word_lower in extended_info:
            # Use extended morphological information if available
            for ext_data in extended_info[word_lower]:
                if ext_data['lemma'] and not estnltk_lemma:
                    estnltk_lemma = ext_data['lemma']
                    estnltk_pos = ext_data['pos']
                    estnltk_form = ext_data['form']
                    break
        
        is_estnltk_normalized_only = (estnltk_lemma.lower() == word_lower)

        def calculate_confidence(method, original_word, best_lemma, num_alternatives=0, levenshtein_dist=0, estnltk_original_pos=None, best_lemma_pos=None):
            confidence = 0.0
            if method == "estnltk+dict":
                confidence = 1.0
            elif method == "estnltk":
                confidence = 0.95 if not is_estnltk_normalized_only else 0.70
            elif method == "dict":
                confidence = 0.85
                if estnltk_original_pos and best_lemma_pos and estnltk_original_pos != best_lemma_pos:
                    confidence -= 0.2
            elif method == "h_variation":
                confidence = 0.70
            elif method == "suffix_strip":
                confidence = 0.80
            elif method == "compound_enhanced":  # v8: Lowered confidence to deprioritize
                confidence = 0.35
            elif method == "levenshtein":
                confidence = max(0.0, 0.6 - (0.15 * levenshtein_dist))
                if estnltk_original_pos and best_lemma_pos and estnltk_original_pos != best_lemma_pos:
                    confidence -= 0.2
            elif method == "compound":
                confidence = 0.30
            elif method == "unknown":
                confidence = 0.0
            
            # Apply dynamic penalties
            if method in ["dict", "levenshtein"] and len(original_word) < 5:
                confidence -= 0.2
            
            if method == "dict" and len(best_lemma) < len(original_word) * 0.7:
                confidence -= 0.3

            if method == "compound" and word[0].isupper() and (disamb_analysis and disamb_analysis.get('sentence_start', False) == False):
                confidence -= 0.1

            return max(0.0, round(confidence, 3))

        def create_result(method, best_lemma, pos, form, alternatives, is_unknown=False, levenshtein_dist=0):
            self.stats[f"{method}_matches"] += 1
            
            best_lemma_pos = self._get_estnltk_analysis_for_lemma(best_lemma)[1] if best_lemma else ''
            
            confidence_score = calculate_confidence(method, word, best_lemma, len(alternatives), levenshtein_dist, estnltk_original_pos=estnltk_pos, best_lemma_pos=best_lemma_pos)
            return {
                "original_word": word,
                "best_lemma": best_lemma,
                "pos": pos,
                "form": form,
                "method": method,
                "alternatives": alternatives,
                "confidence_score": confidence_score,
                "is_unknown": is_unknown
            }

        # Enhanced fallback logic with EstNLTK 1.7.4 features

        # 1. EstNLTK + Dictionary Match (Gold Standard)
        if estnltk_lemma and estnltk_lemma.lower() in self.variant_to_lemmas:
            if word_lower in [v.lower() for v in self.lemma_to_variants.get(estnltk_lemma.lower(), [])]:
                alt_lemmas = [l for l in self.variant_to_lemmas.get(word_lower, []) if l.lower() != estnltk_lemma.lower()]
                return create_result("estnltk+dict", estnltk_lemma, estnltk_pos, estnltk_form, self._sort_lemmas_by_similarity(word, alt_lemmas))

        # 2. Confident EstNLTK Analysis
        if estnltk_lemma and estnltk_lemma.lower() != word_lower:
            return create_result("estnltk", estnltk_lemma, estnltk_pos, estnltk_form, [])

        # 3. Dictionary-Only Match
        dict_lemmas = self.variant_to_lemmas.get(word_lower, [])
        if word_lower in self.known_lemmatizations:
            known_lemma = self.known_lemmatizations[word_lower]
            if known_lemma not in dict_lemmas:
                dict_lemmas.insert(0, known_lemma)

        if dict_lemmas:
            sorted_dict_lemmas = self._sort_lemmas_by_similarity(word, dict_lemmas)
            primary_lemma = sorted_dict_lemmas[0]
            alt_lemmas = sorted_dict_lemmas[1:]
            _, pos, form = self._get_estnltk_analysis_for_lemma(primary_lemma)
            return create_result("dict", primary_lemma, pos, form, alt_lemmas)

        # 4. H-Variation (v8: moved up to prioritize morphological analysis)
        h_lemma = self._try_h_variation(word)
        if h_lemma:
            _, pos, form = self._get_estnltk_analysis_for_lemma(h_lemma)
            return create_result("h_variation", h_lemma, pos, form, [])
        
        # 5. Enhanced Suffix-Stripping (v8: moved up to prioritize morphological analysis)
        suffix_stripped_lemma = self._try_suffix_strip(word)
        if suffix_stripped_lemma:
            _, pos, form = self._get_estnltk_analysis_for_lemma(suffix_stripped_lemma)
            return create_result("suffix_strip", suffix_stripped_lemma, pos, form, [])

        # 6. Levenshtein (v8: moved up to prioritize morphological analysis)
        lev_lemma, lev_dist, lev_alt_lemmas = self._try_levenshtein(word)
        if lev_lemma:
            _, pos, form = self._get_estnltk_analysis_for_lemma(lev_lemma)
            return create_result("levenshtein", lev_lemma, pos, form, lev_alt_lemmas, levenshtein_dist=lev_dist)

        # 7. Fallback Compound Splitting
        compound_lemma = self._try_compound_split(word)
        if compound_lemma:
            _, pos, form = self._get_estnltk_analysis_for_lemma(compound_lemma)
            return create_result("compound", compound_lemma, pos, form, [])

        # 8. Enhanced Compound Splitting with Validation (v8: moved to end, with validation)
        if compound_info:
            compound_lemma = self._try_compound_split_enhanced(word, compound_info)
            if compound_lemma:
                # v8: Apply morphological validation
                components = compound_lemma.split('+')
                if self._validate_compound_candidate(components):
                    _, pos, form = self._get_estnltk_analysis_for_lemma(components[0])
                    return create_result("compound_enhanced", compound_lemma, pos, form, [])

        # 9. Final Fallback: Unknown
        self.stats["fallback_unknown"] += 1
        return create_result("unknown", word, "", "", [], is_unknown=True)

    def lemmatize_text(self, text_str):
        # PUNCTUATION FIX: Remove problematic punctuation
        punctuation_chars = '´`\'''""„"‚'
        cleaned_text = text_str
        for char in punctuation_chars:
            cleaned_text = cleaned_text.replace(char, '')
        
        text_obj = Text(cleaned_text)
        text_obj.tag_layer(['words', 'sentences', 'morph_analysis'])
        
        # Standard disambiguation
        tagger = VabamorfTagger(postdisambiguate=True, output_layer='morph_analysis_disamb')
        tagger.tag(text_obj)
        
        # Get EstNLTK 1.7.4 enhanced analyses
        compound_info = self._get_compound_analysis(text_obj) if self.enable_174_features else {}
        extended_info = self._get_extended_morph_analysis(text_obj) if self.enable_174_features else {}
        
        # Extract original words from input
        original_words = []
        for word in text_str.split():
            if re.search(r'\w', word):
                original_words.append(word)
        original_word_index = 0

        results = []
        i = 0
        while i < len(text_obj.words):
            word_span = text_obj.words[i]
            word = word_span.text
            
            # Skip pure punctuation
            if re.fullmatch(r'[^\w\s]+', word):
                i += 1
                continue
            
            disamb_analysis = text_obj.morph_analysis_disamb[i].annotations[0] if i < len(text_obj.morph_analysis_disamb) and text_obj.morph_analysis_disamb[i].annotations else {}
            
            original_word = original_words[original_word_index] if original_word_index < len(original_words) else word
            word_result = self.lemmatize_word(original_word, disamb_analysis, compound_info, extended_info)
            results.append(word_result)
            
            original_word_index += 1
            i += 1
        return results

    def get_statistics(self):
        total = self.stats.get("total_words", 0)
        if total == 0: return self.stats
        stats_percent = self.stats.copy()
        for key, value in self.stats.items():
            if key != "total_words":
                stats_percent[f"{key}_percent"] = round(100 * value / total, 1)
        return stats_percent

def main():
    parser = argparse.ArgumentParser(description='Estonian Runosong Lemmatizer with EstNLTK 1.7.4 Enhancements')
    parser.add_argument('--annotation-dir', type=str, required=True, help='Directory containing annotation files')
    parser.add_argument('--text', type=str, help='Text to lemmatize')
    parser.add_argument('--input', type=str, help='Input file to lemmatize')
    parser.add_argument('--output', type=str, help='Output file for the result')
    parser.add_argument('--pretty-print', action='store_true', help='Pretty-print the JSON output')
    parser.add_argument('--disable-174-features', action='store_true', help='Disable EstNLTK 1.7.4 features')
    parser.add_argument('--output-format', type=str, default='full', choices=['full', 'simple-json', 'inline-text'],
                        help='Output format: "full" (default), "simple-json", or "inline-text"')
    args = parser.parse_args()

    lemmatizer = RunosongLemmatizerWithEstNLTK174(args.annotation_dir, enable_174_features=not args.disable_174_features)

    text_to_process = ""
    if args.text:
        text_to_process = args.text
    elif args.input:
        try:
            with open(args.input, 'r', encoding='utf-8') as f:
                text_to_process = f.read()
        except Exception as e:
            logger.error(f"Error reading input file: {e}")
            return 1
    else:
        parser.print_help()
        return 1

    lemmatization_results = lemmatizer.lemmatize_text(text_to_process)
    stats = lemmatizer.get_statistics()
    
    output_content = ""
    if args.output_format == 'full':
        final_output = {
            "text": text_to_process,
            "lemmatization_results": lemmatization_results,
            "statistics": stats
        }
        output_content = json.dumps(final_output, ensure_ascii=False, indent=(4 if args.pretty_print else None))
    elif args.output_format == 'simple-json':
        simple_results = []
        for res in lemmatization_results:
            simple_results.append({
                "original_word": res["original_word"],
                "best_lemma": res["best_lemma"],
                "method": res["method"],
                "confidence_score": res["confidence_score"]
            })
        final_output = {
            "text": text_to_process,
            "lemmatization_results": simple_results,
            "statistics": stats
        }
        output_content = json.dumps(final_output, ensure_ascii=False, indent=(4 if args.pretty_print else None))
    elif args.output_format == 'inline-text':
        inline_parts = []
        for res in lemmatization_results:
            inline_parts.append(f"{res['original_word']} ({res['best_lemma']}) [{res['confidence_score']}]")
        output_content = " ".join(inline_parts)

    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output_content + "\n")
            logger.info(f"Output successfully saved to {args.output}")
        except Exception as e:
            logger.error(f"Error writing to output file: {e}")
    else:
        print(output_content)

if __name__ == "__main__":
    main()