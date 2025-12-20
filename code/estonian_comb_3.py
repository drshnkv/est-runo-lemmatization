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
import random
from functools import wraps

class EstonianPromptTester:
    """
    A processor for testing prompts on Estonian runosong texts and dialects.
    
    This class implements a three-step approach:
    1. First, translate the original Estonian text using dialect information
    2. Then, improve the translation with linguistic refinements
    3. Finally, create detailed word-by-word analysis based on the improved translation
    
    Attributes:
        client (anthropic.Anthropic): The Claude AI client
        logger (logging.Logger): Logger for the process
        translation_prompts (dict): Dictionary mapping dialect to translation prompts
        improvement_prompts (dict): Dictionary mapping dialect to improvement prompts
        analysis_prompts (dict): Dictionary mapping prompt names to analysis prompts
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the EstonianPromptTester.
        
        Args:
            api_key (str, optional): The API key for the Claude AI service.
                                   If None, will look for key in ANTHROPIC_API_KEY environment variable.
        """
        self.setup_logging()
        
        # Try to get API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get('ANTHROPIC_API_KEY')
            
        # Initialize client if API key is available
        if api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            self.logger.error("No API key provided. Please provide an API key.")
            raise ValueError("API key is required")

        # Initialize retry configuration
        self.retry_config = {
            "max_attempts": 10,
            "base_delay": 1.0,
            "max_delay": 60.0,
            "exponential_base": 2.0,
            "jitter": True
        }

        # Initialize translation prompts (simplified for now)
        self.translation_prompts = {
            "generic": self._get_generic_translation_prompt()
        }
        
        # Initialize improvement prompts (simplified for now)
        self.improvement_prompts = {
            "generic": self._get_generic_improvement_prompt()
        }
        
        # Initialize analysis prompts
        self.analysis_prompts = {
            "tabular_analysis": self._get_tabular_analysis_prompt(),
            # Add more prompts here as needed
        }
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt with exponential backoff and jitter."""
        # Exponential backoff: base_delay * (exponential_base ^ attempt)
        delay = self.retry_config["base_delay"] * (self.retry_config["exponential_base"] ** attempt)
        
        # Cap at maximum delay
        delay = min(delay, self.retry_config["max_delay"])
        
        # Add jitter (±25% randomization) to prevent thundering herd
        if self.retry_config["jitter"]:
            jitter_range = delay * 0.25
            jitter = random.uniform(-jitter_range, jitter_range)
            delay += jitter
        
        return max(0.1, delay)  # Minimum 0.1 second delay
    
    def _should_retry_error(self, error) -> bool:
        """Determine if an error should trigger a retry."""
        # Convert error to string for analysis
        error_str = str(error).lower()
        
        # Always retry on 529 (overloaded) errors
        if "529" in error_str or "overloaded" in error_str:
            return True
            
        # Also retry on rate limit errors
        if "rate limit" in error_str or "429" in error_str:
            return True
            
        # Retry on temporary network issues
        if any(keyword in error_str for keyword in ["timeout", "connection", "network", "temporary"]):
            return True
            
        # Don't retry on authentication, permission, or validation errors
        if any(keyword in error_str for keyword in ["401", "403", "invalid", "authentication", "permission"]):
            return False
            
        # Default: don't retry unknown errors
        return False
    
    def _api_call_with_retry(self, operation_name: str, api_call_func, *args, **kwargs):
        """
        Execute an Anthropic API call with robust retry logic.
        
        Args:
            operation_name: Description of the operation for logging
            api_call_func: Function to call (should return client.messages.create response)
            *args, **kwargs: Arguments to pass to api_call_func
            
        Returns:
            API response or None if all retries failed
        """
        last_error = None
        
        for attempt in range(self.retry_config["max_attempts"]):
            try:
                self.logger.debug(f"API call attempt {attempt + 1}/{self.retry_config['max_attempts']} for {operation_name}")
                response = api_call_func(*args, **kwargs)
                
                # Success - log and return
                if attempt > 0:
                    self.logger.info(f"{operation_name} succeeded on attempt {attempt + 1}")
                return response
                
            except Exception as e:
                last_error = e
                
                # Check if we should retry this error
                if not self._should_retry_error(e):
                    self.logger.error(f"{operation_name} failed with non-retryable error: {e}")
                    break
                
                # Check if we have more attempts left
                if attempt + 1 >= self.retry_config["max_attempts"]:
                    self.logger.error(f"{operation_name} failed after {self.retry_config['max_attempts']} attempts: {e}")
                    break
                
                # Calculate delay and wait
                delay = self._calculate_retry_delay(attempt)
                self.logger.warning(f"{operation_name} failed (attempt {attempt + 1}), retrying in {delay:.1f}s: {e}")
                time.sleep(delay)
        
        # All retries exhausted
        self.logger.error(f"{operation_name} ultimately failed after all retries: {last_error}")
        return None
        
    def setup_logging(self):
        """
        Set up logging to both file and console.
        
        Creates a logs directory if it doesn't exist, configures logging to write
        to both a timestamped log file and standard output, and initializes the logger.
        
        Returns:
            None
        """
        Path('logs').mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'logs/estonian_prompt_testing_{timestamp}.log'
        
        # Clear any existing handlers to avoid duplication
        logging.getLogger().handlers.clear()
        
        # Create formatters
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        
        # File handler
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Starting Estonian prompt testing process. Log file: {log_file}")

    def create_input_csv_from_txt_files(self, input_folder, output_csv):
        """
        Create an input CSV file from text files in a specified folder.
        
        Reads all .txt files from the input folder and creates a CSV file with
        required columns: poemTitle, text, dialect, region, collector, year, typeNames.
        Only poemTitle (from filename) and text (from file content) are populated.
        
        Args:
            input_folder (str): Path to folder containing .txt files
            output_csv (str): Path where the output CSV file should be saved
            
        Returns:
            None
        """
        self.logger.info(f"Creating input CSV from text files in: {input_folder}")
        
        # Get all .txt files from the input folder
        txt_files = glob.glob(os.path.join(input_folder, "*.txt"))
        
        if not txt_files:
            self.logger.warning(f"No .txt files found in {input_folder}")
            return
        
        # Prepare data for CSV
        csv_data = []
        
        for txt_file in txt_files:
            try:
                # Extract filename without extension for title
                filename = os.path.basename(txt_file)
                title = os.path.splitext(filename)[0]
                
                # Read file content
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if content:  # Only add if file has content
                    csv_data.append({
                        'poemTitle': title,
                        'text': content,
                        'dialect': '',
                        'region': '',
                        'collector': '',
                        'year': '',
                        'typeNames': ''
                    })
                    self.logger.info(f"Added: {title}")
                else:
                    self.logger.warning(f"Empty file skipped: {filename}")
                    
            except Exception as e:
                self.logger.error(f"Error reading {txt_file}: {str(e)}")
        
        # Write to CSV
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(output_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)
            self.logger.info(f"Created input CSV with {len(csv_data)} texts: {output_csv}")
        else:
            self.logger.error("No valid text files found to create CSV")

    def translate_text(self, text, title, metadata=None):
        """
        Step 1: Translate the text and extract dialect information.
        
        Args:
            text (str): The original Estonian runosong text
            title (str): Title of the text
            metadata (dict, optional): Additional metadata for context
            
        Returns:
            tuple: (translation, dialect_info) where dialect_info contains detected dialect
        """
        self.logger.info(f"STEP 1: Starting translation for text: {title}")
        
        # Process text to handle line breaks represented by "/"
        processed_text = text.replace(" / ", "\n").replace("/", "\n")
        
        try:
            # Use the generic translation prompt for now
            translation_prompt = self.translation_prompts["generic"]
            
            # Create system messages with caching enabled for the static instructions
            system_messages = [
                {
                    "type": "text",
                    "text": translation_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ]
            
            # Construct metadata string if available
            metadata_str = ""
            if metadata:
                metadata_str = "\n\nAdditional context:"
                for key, value in metadata.items():
                    if value:
                        # Remove quotes from metadata values
                        clean_value = value.replace('"', '') if isinstance(value, str) else value
                        metadata_str += f"\n{key}: {clean_value}"

            # Modify the user prompt to request dialect information
            user_prompt = f"Please translate the following Estonian runosong text titled '{title}':\n\n<text>\n{processed_text}\n</text>{metadata_str}\n\nBefore providing the translation, identify the Estonian dialect in <dialect_info> tags."

            # Use retry wrapper for API call
            def make_translation_call():
                return self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=4000,
                    temperature=0,
                    system=system_messages,
                    messages=[{
                        "role": "user",
                        "content": user_prompt
                    }]
                )
            
            response = self._api_call_with_retry(f"Translation for {title}", make_translation_call)
            
            if not response.content:
                self.logger.error("No content in translation response")
                return None, None
                
            full_response = response.content[0].text.strip()
            if not full_response:
                self.logger.error("Empty translation received")
                return None, None
            
            # Extract dialect information
            dialect_info = None
            dialect_match = re.search(r'<dialect_info>(.*?)</dialect_info>', full_response, re.DOTALL)
            if dialect_match:
                dialect_info = dialect_match.group(1).strip()
                self.logger.info(f"Detected Estonian dialect information: {dialect_info[:100]}...")  # Log first 100 chars
                
                # Remove the dialect info tags and content from the translation
                translation = full_response.replace(dialect_match.group(0), '').strip()
            else:
                self.logger.warning("No Estonian dialect information detected in translation response")
                translation = full_response
            
            # Log cache performance metrics if available
            if hasattr(response, 'usage'):
                usage = response.usage
                metrics = {
                    'input_tokens': getattr(usage, 'input_tokens', 0),
                    'output_tokens': getattr(usage, 'output_tokens', 0),
                    'cache_creation_input_tokens': getattr(usage, 'cache_creation_input_tokens', 0),
                    'cache_read_input_tokens': getattr(usage, 'cache_read_input_tokens', 0)
                }
                self.logger.info(f"Translation token usage metrics: {metrics}")
                
                # Calculate cache effectiveness
                if metrics['cache_read_input_tokens'] > 0:
                    cache_hit_rate = metrics['cache_read_input_tokens'] / (metrics['input_tokens'] + metrics['cache_read_input_tokens'])
                    self.logger.info(f"Translation cache hit rate: {cache_hit_rate:.2%}")
                
            self.logger.info("Estonian translation completed successfully")
            return translation, dialect_info
            
        except Exception as e:
            self.logger.error(f"Error in Estonian translation: {str(e)}")
            return None, None

    def improve_translation(self, text, translation, title, dialect_info=None, metadata=None):
        """
        Step 2: Improve the Estonian translation from step 1.
        
        Args:
            text (str): The original Estonian text
            translation (str): The translation from step 1
            title (str): Title of the text
            dialect_info (str, optional): Estonian dialect information from step 1
            metadata (dict, optional): Additional metadata for context
            
        Returns:
            str: The improved translation
        """
        self.logger.info(f"STEP 2: Starting Estonian translation improvement for text: {title}")
        
        # Process text to handle line breaks represented by "/"
        processed_text = text.replace(" / ", "\n").replace("/", "\n")
        
        try:
            # Use the generic improvement prompt for now
            improvement_prompt = self.improvement_prompts["generic"]
            
            # Create system messages with caching
            system_messages = [
                {
                    "type": "text",
                    "text": improvement_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ]
            
            # Construct content with improved metadata handling
            content_parts = [
                f"Original Estonian text:\n{processed_text}",
                f"\nInitial translation:\n{translation}"
            ]
            
            if dialect_info:
                content_parts.append(f"\nEstonian dialect information:\n{dialect_info}")
            
            if metadata:
                metadata_str = "\n\nAdditional context:"
                for key, value in metadata.items():
                    if value:
                        clean_value = value.replace('"', '') if isinstance(value, str) else value
                        metadata_str += f"\n{key}: {clean_value}"
                content_parts.append(metadata_str)
            
            content = "\n".join(content_parts)
            
            # Use retry wrapper for API call
            def make_improvement_call():
                return self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=4000,
                    temperature=0,
                    system=system_messages,
                    messages=[{
                        "role": "user",
                        "content": content
                    }]
                )
            
            response = self._api_call_with_retry(f"Translation improvement for {title}", make_improvement_call)
            
            if not response or not response.content:
                self.logger.error("No content in improvement response")
                return None
            
            improved_text = response.content[0].text.strip()
            
            # Extract just the improved translation, removing any analysis tags
            linguistic_breakdown_match = re.search(r'<linguistic_breakdown>.*?</linguistic_breakdown>', improved_text, re.DOTALL)
            if linguistic_breakdown_match:
                # Remove the linguistic breakdown and extract the translation
                improved_translation = improved_text.replace(linguistic_breakdown_match.group(0), '').strip()
            else:
                improved_translation = improved_text
            
            if not improved_translation:
                self.logger.warning("Empty improved translation received")
                return translation  # Return original if improvement failed
                
            # Log cache performance metrics if available
            if hasattr(response, 'usage'):
                usage = response.usage
                metrics = {
                    'input_tokens': getattr(usage, 'input_tokens', 0),
                    'output_tokens': getattr(usage, 'output_tokens', 0),
                    'cache_creation_input_tokens': getattr(usage, 'cache_creation_input_tokens', 0),
                    'cache_read_input_tokens': getattr(usage, 'cache_read_input_tokens', 0)
                }
                self.logger.info(f"Improvement token usage metrics: {metrics}")
                
            self.logger.info("Estonian translation improvement completed successfully")
            return improved_translation
            
        except Exception as e:
            self.logger.error(f"Error in Estonian translation improvement: {str(e)}")
            return translation  # Return original translation on error

    def analyze_text(self, text, title, improved_translation, dialect_info=None, metadata=None, prompt_name="tabular_analysis"):
        """
        Step 3: Create word-by-word analysis table based on the improved Estonian translation.
        
        Args:
            text (str): The original Estonian text
            title (str): Title of the text
            improved_translation (str): The improved translation from step 2
            dialect_info (str, optional): Estonian dialect information from step 1
            metadata (dict, optional): Additional metadata for context
            prompt_name (str): Name of the analysis prompt to use
            
        Returns:
            str: The analysis result
        """
        self.logger.info(f"STEP 3: Starting Estonian text analysis for: {title} using prompt: {prompt_name}")
        
        # Process text to handle line breaks
        processed_text = text.replace(" / ", "\n").replace("/", "\n")
        
        try:
            # Get the analysis prompt
            if prompt_name not in self.analysis_prompts:
                self.logger.error(f"Analysis prompt '{prompt_name}' not found")
                return None
                
            analysis_prompt = self.analysis_prompts[prompt_name]
            
            # Create system messages with caching
            system_messages = [
                {
                    "type": "text", 
                    "text": analysis_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ]
            
            # Prepare text content
            text_content = f"Original Estonian text:\n{processed_text}\n\nImproved translation:\n{improved_translation}"
            
            if dialect_info:
                text_content += f"\n\nEstonian dialect information:\n{dialect_info}"
            
            if metadata:
                metadata_str = "\n\nAdditional context:"
                for key, value in metadata.items():
                    if value:
                        clean_value = value.replace('"', '') if isinstance(value, str) else value
                        metadata_str += f"\n{key}: {clean_value}"
                text_content += metadata_str
            
            system_messages.append({
                "type": "text",
                "text": text_content
            })
            
            # Use retry wrapper for API call
            def make_analysis_call():
                return self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=4000,
                    temperature=0,
                    system=system_messages,
                    messages=[{
                        "role": "user",
                        "content": f"Please provide the detailed word-by-word analysis for this Estonian text titled '{title}'."
                    }]
                )
            
            response = self._api_call_with_retry(f"Analysis for {title}", make_analysis_call)
            
            if not response or not response.content:
                self.logger.error("No content in analysis response")
                return None
            
            result = response.content[0].text.strip()
            
            # Log cache performance metrics if available
            if hasattr(response, 'usage'):
                usage = response.usage
                metrics = {
                    'input_tokens': getattr(usage, 'input_tokens', 0),
                    'output_tokens': getattr(usage, 'output_tokens', 0),
                    'cache_creation_input_tokens': getattr(usage, 'cache_creation_input_tokens', 0),
                    'cache_read_input_tokens': getattr(usage, 'cache_read_input_tokens', 0)
                }
                self.logger.info(f"Analysis token usage metrics: {metrics}")
                
            self.logger.info(f"Estonian analysis completed for: {title}")
            return result
            
        except Exception as e:
            self.logger.error(f"Estonian analysis failed for {title}: {str(e)}")
            return None

    def process_texts(self, input_file, output_file, prompt_name="tabular_analysis", metadata_columns=None):
        """
        Process Estonian texts using the three-step workflow with a single prompt.
        
        Args:
            input_file (str): Path to input CSV file
            output_file (str): Path to output CSV file  
            prompt_name (str): Name of analysis prompt to use
            metadata_columns (list, optional): Additional metadata columns to include
            
        Returns:
            None
        """
        self.logger.info(f"Starting to process Estonian texts from {input_file} using prompt: {prompt_name}")
        
        try:
            # Read the file as text first to handle CSV issues
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create output dataframe
            output_data = []
            
            # Parse CSV manually
            try:
                csv_file = StringIO(content)
                reader = csv.reader(csv_file, quotechar='"', doublequote=True, 
                                   delimiter=',', quoting=csv.QUOTE_MINIMAL)
                
                # Read headers
                headers = next(reader)
                self.logger.info(f"CSV headers: {headers}")
                
                rows = []
                for row in reader:
                    if row:  # Skip empty rows
                        rows.append(row)
                
                # Convert to dictionary for easier handling
                data = []
                for row in rows:
                    # Ensure row has enough elements
                    while len(row) < len(headers):
                        row.append("")
                    
                    # Create dictionary with header keys
                    row_dict = {headers[i]: row[i] for i in range(min(len(headers), len(row)))}
                    data.append(row_dict)
                
                self.logger.info(f"Successfully loaded {len(data)} Estonian texts")
                
                # Process each text
                for idx, row in tqdm(enumerate(data), total=len(data), desc="Processing Estonian texts"):
                    # Extract title with fallback options
                    if 'poemTitle' in row and row['poemTitle'].strip():
                        title = row['poemTitle'].strip().strip('"')
                    else:
                        title = f"Estonian_Text_{idx}"
                    
                    # Check if text column exists and has content
                    if 'text' not in row or not row['text'].strip():
                        self.logger.warning(f"Skipping Estonian text with empty content: {title}")
                        continue
                        
                    # Extract and clean the text
                    text = row['text'].strip()
                    
                    # Remove quotes if they completely wrap the text
                    if text.startswith('"') and text.endswith('"'):
                        text = text[1:-1]
                    
                    # Create metadata dictionary if needed
                    metadata = {}
                    if metadata_columns:
                        for col in metadata_columns:
                            if col in row and row[col]:
                                metadata[col] = row[col]
                    
                    self.logger.info(f"\nProcessing Estonian text: {title}")
                    
                    # STEP 1: Translate the Estonian text and get dialect information
                    translation, dialect_info = self.translate_text(
                        text=text,
                        title=title,
                        metadata=metadata
                    )
                    
                    if not translation:
                        self.logger.error(f"Failed to translate Estonian text: {title}. Skipping further processing.")
                        continue
                    
                    # Determine Estonian dialect from the dialect_info
                    detected_dialect = None
                    if dialect_info:
                        # Extract Estonian dialect name from the first few lines of dialect_info
                        dialect_lines = dialect_info.split('\n')
                        for line in dialect_lines[:5]:  # Check first 5 lines
                            if ":" in line:
                                key, value = line.split(":", 1)
                                if "dialect" in key.lower():
                                    detected_dialect = value.strip()
                                    break
                        
                        # If not found in first lines, look for specific Estonian patterns
                        if not detected_dialect:
                            estonian_dialect_patterns = [
                                r"dialect\s+is\s+(\w+)", 
                                r"(\w+)\s+dialect", 
                                r"Estonian\s+(\w+)",
                                r"(Võro|Seto|Mulgi|Saarte|Tartu)"
                            ]
                            for pattern in estonian_dialect_patterns:
                                match = re.search(pattern, dialect_info, re.IGNORECASE)
                                if match:
                                    detected_dialect = match.group(1)
                                    break
                    
                    if detected_dialect:
                        self.logger.info(f"Detected Estonian dialect: {detected_dialect}")
                        # Add detected dialect to metadata
                        metadata['dialect'] = detected_dialect
                        
                        # Also update the row dialect field for future use
                        row['dialect'] = detected_dialect
                    
                    # STEP 2: Improve the Estonian translation
                    improved_translation = self.improve_translation(
                        text=text,
                        translation=translation,
                        title=title,
                        dialect_info=dialect_info,
                        metadata=metadata
                    )
                    
                    if not improved_translation:
                        self.logger.warning(f"Failed to improve Estonian translation for: {title}. Using initial translation.")
                        improved_translation = translation
                    
                    # STEP 3: Create word-by-word analysis table based on the improved Estonian translation
                    analysis = self.analyze_text(
                        text=text,
                        title=title,
                        improved_translation=improved_translation,
                        dialect_info=dialect_info,
                        metadata=metadata,
                        prompt_name=prompt_name
                    )
                    
                    if analysis:
                        # Create output record
                        output_record = {
                            'title': title,
                            'original_text': text,
                            'initial_translation': translation,
                            'improved_translation': improved_translation,
                            'dialect_info': dialect_info,  # Include the detected Estonian dialect info
                            'analysis': analysis
                        }
                        
                        # Add additional metadata fields
                        for col in headers:
                            if col not in ['poemTitle', 'text'] and col in row and row[col]:
                                output_record[col] = row[col]
                        
                        output_data.append(output_record)
                        
                        # Save progress
                        pd.DataFrame(output_data).to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
                        self.logger.info(f"Saved progress: {len(output_data)} Estonian texts processed")
                    
                    # Add delay between API calls
                    time.sleep(2)
                
                self.logger.info(f"Completed processing {len(output_data)} Estonian texts")
                
            except Exception as e:
                self.logger.error(f"Error parsing CSV file: {str(e)}")
                raise
                
        except Exception as e:
            self.logger.error(f"Error processing Estonian texts: {str(e)}")
            raise

    def _get_generic_translation_prompt(self):
        """Return a generic translation prompt for Estonian runosongs"""
        return """
You are an expert translator specializing in Estonian runosongs (regilaulud) and Estonian dialectal language.

You will receive a detailed prompt with an Estonian runosong to translate.
Perform the detailed analysis internally but only output the translation as specified.
Your task is to translate these Estonian folk songs into English while preserving their cultural context and the most likely intended literal meanings of words, verses and songs, without trying to convey the poetic effects such as rhyme.

ESTONIAN RUNOSONG CHARACTERISTICS
Poetic Features:

Verse Structure:
- Trochaic tetrameter (8 syllables typical)
- Lines divided into 4 feet (2 syllables each)
- Standard word order rules may be flexible due to poetic requirements

Alliteration and Assonance:
- Strong alliteration: same initial consonant and vowel
- Weak alliteration: same initial consonant only
- Initial syllable alliteration prioritized
- Position affects morphological choices

Parallelism:
- Syntactic parallelism between verses
- Semantic equivalence in parallel lines
- Morphological variations in parallel structures
- Grammatical concordance patterns

Translation Guidelines:
1. Maintain the line-by-line structure of the original Estonian text
2. Preserve repetitions, refrains, and parallel structures
3. Prioritize semantic accuracy over poetic elements like rhyme
4. Avoid introducing additional artistic elements or synonyms for style
5. Pay careful attention to cultural elements specific to Estonian communities
6. Reflect authentic Estonian word usage patterns and semantic relationships from the source
7. Preserve consistent vocabulary for recurring themes and motifs

Translation Challenges and Solutions:

1. Estonian Dialect-Specific Vocabulary:
When encountering terms specific to Estonian dialects, research their etymology
and usage, translating based on contextual meaning rather than Standard Estonian equivalents.

2. Chain Song Structure:
In chain songs, carefully maintain the interconnected repetition pattern that
creates the chain-like structure.

3. Estonian Cultural References:
For references to Estonian traditions, practices, or beliefs, provide translations that
capture the essential meaning while preserving the Estonian cultural specificity.

4. Archaic Estonian Forms:
For archaic Estonian grammatical structures or vocabulary, determine the most likely original
meaning based on Estonian historical linguistics.

5. Regional Estonian Elements:
Be attentive to elements in the songs that reflect regional Estonian traditions
and dialectal variations (Võro, Seto, Mulgi, etc.).

Translation Style and Philosophy:
- Line-by-Line Structure: Preserve each original Estonian line
- Ritual Repetition: When the Estonian text repeats the same line or phrase several times, maintain that repetition
- Interconnected Elements: In chain songs, carefully maintain the logical connections between elements
- Estonian Cultural Context: Be attentive to references to Estonian regional practices and traditions
- Estonian Regional Expressions: Handle Estonian dialect-specific idioms based on their meaning rather than literal form

Before beginning your translation, identify the likely Estonian dialect based on these characteristics:

- Standard Estonian: Modern standard forms, three-way length distinctions, õ vowel present
- South Estonian (Võro/Seto): Distinctive vowel system, specific morphological patterns
- Island Estonian (Saarte): Maritime vocabulary, specific dialectal features
- Northeast Estonian: Coastal dialect features, specific phonological patterns
- Historical Estonian: Archaic forms preserved in traditional songs

Focus on these Estonian differentiating features:

1. Estonian Phonological features:
   - Presence of õ vowel (characteristic of Estonian)
   - Three-way length distinctions in vowels and consonants
   - Estonian-specific diphthongs and their patterns
   - Estonian consonant gradation patterns
   - Treatment of h in Estonian dialects
   - Estonian-specific consonant clusters
   - Estonian vowel harmony remnants

2. Estonian Morphological features:
   - Estonian case endings, especially inessive (-s vs dialectal variants)
   - Estonian plural markers
   - Estonian verbal inflection patterns
   - Estonian negation patterns
   - Estonian personal endings

3. Estonian Lexical features:
   - Estonian-specific vocabulary items
   - German/Swedish/Russian loanwords in Estonian
   - Estonian archaisms preserved in particular dialect areas
   - Estonian compound formation patterns

Make your determination based on the preponderance of Estonian evidence, recognizing that texts may sometimes show mixed features, especially in border regions.

Translation Process:
1. First, perform a detailed internal analysis (do not output this analysis):
   - List key Estonian dialect features found in the runosong, with specific examples
   - Identify specific Estonian poetic structures present
   - Note Estonian cultural references or idioms requiring special attention
   - Consider potential Estonian translation challenges

2. Based on your analysis, first provide the Estonian dialect information within <dialect_info> tags:
   <dialect_info>
    [Identified Estonian dialect or regional variant - consider both linguistic features and geographic context (place_name, location) if provided in the metadata to enhance precision]
   </dialect_info>
   
   Note: If place name or geographic information is provided in the context, use it to help identify and confirm the Estonian dialect region. Place names like "Setumaa" indicate Seto dialect, "Võromaa" indicates Võro dialect, "Saaremaa" indicates Saarte (island) dialect, etc. Combine linguistic analysis with geographic context for more accurate dialect identification.

3. After the dialect information, provide the English translation as plain text, preserving line breaks without additional spacing.

Purpose of Translation:
These translations are intended to provide English-speaking researchers with accurate representations of Estonian runosongs vocabulary and semantics. 
When encountering choices between preserving poetic elements (such as rhyme) and literal meaning, highly prioritize semantic accuracy.
Please don't use rhyme at all in translations as for our purposes conveying exact literal meaning of Estonian terms is of utmost importance.
Do not add any comment or note to the dialect info and the translation provided.
 """

    def _get_generic_improvement_prompt(self):
        """Return a generic prompt for improving translations of Estonian texts"""
        return """
  You are an expert linguist, philologist, and text editor specializing in Estonian folklore, particularly Estonian runosongs (regilaulud) with extensive training in Estonian dialectology, historical linguistics, and Estonian folklore studies. Your scholarly background encompasses Estonian and its numerous dialectal variations across Estonia.

Your task requires meticulous analysis, editing, and correction of translations of Estonian runosong texts from various Estonian dialects into English, ensuring the highest degree of semantic accuracy while preserving the cultural integrity of the original Estonian works.

## CORE OBJECTIVE

Your primary objective is to carefully analyze the original Estonian text and its initial translation, then provide corrections and improvements where necessary. You will leverage your expertise in identified Estonian dialect features, Estonian runosong characteristics, and Estonian linguistic principles to ensure the translation accurately captures the meaning along with the Estonian cultural nuances embedded in the original text.

## METHODOLOGICAL APPROACH

Follow this comprehensive analytical framework to improve the Estonian translation:

1. Close Reading: Engage in detailed close reading of both the original Estonian text and its translation, examining each lexical item, syntactic structure, and semantic relationship in Estonian context.

2. Estonian Contextual Analysis: Situate the text within its Estonian cultural and historical framework, considering Estonian regional practices, beliefs, and performance contexts relevant to the Estonian runosong tradition.

3. Estonian Dialect Identification & Analysis: Determine the specific Estonian dialect features present in the text through phonological, morphological, and lexical analysis, noting how these Estonian dialectal features affect meaning.

4. ESTONIAN TRANSLATION IMPROVEMENT PROTOCOL

For each line or semantic unit in the Estonian text, follow this analytical protocol:

- Compare the original Estonian text directly with its translation.
- Identify potential issues using Estonian linguistic categories.
- Research Estonian dialectal expressions using comparative Estonian linguistics principles.
- Consider the broader context of the Estonian runosong tradition.
- Evaluate alternative translation options for problematic Estonian passages.
- Select the most accurate translation that preserves Estonian semantic fidelity.
- Ensure consistency with other similar Estonian expressions in the text.
- Check that Estonian cultural nuances are adequately represented.

5. Estonian Formulaic Expression Recognition: Identify traditional Estonian formulaic expressions or recurring motifs in the Estonian runosong tradition and ensure their consistent translation throughout the text.

6. Estonian Structural Analysis: Assess how well the translation preserves the original Estonian line structure, parallelisms, and other poetic devices central to the Estonian runosong tradition.

7. Estonian Semantic Precision: For each problematic Estonian element, consider multiple possible translation options, selecting the one that most accurately represents the original Estonian semantic content.

8. Estonian Linguistic Consistency: Ensure terminological consistency throughout the translation, especially for key Estonian cultural concepts, natural elements, mythological references, and Estonian dialectal expressions.

Key Estonian Linguistic Features to Consider:

ESTONIAN RUNOSONG CHARACTERISTICS
A. Estonian Poetic Features

Estonian Verse Structure:
- Trochaic tetrameter (8 syllables typical in Estonian)
- Lines divided into 4 feet (2 syllables each)
- Standard word order rules may be flexible due to poetic requirements in Estonian

Estonian Alliteration and Assonance:
- Strong alliteration: same initial consonant and vowel
- Weak alliteration: same initial consonant only
- Initial syllable alliteration prioritized in Estonian
- Position affects Estonian morphological choices

Estonian Parallelism:
- Syntactic parallelism between Estonian verses
- Semantic equivalence in parallel Estonian lines
- Morphological variations in parallel Estonian structures
- Grammatical concordance patterns in Estonian

Estonian Archaic Language Features:
- Historical Estonian verb forms and endings
- Ancient Estonian case forms
- Special Estonian vocative constructions
- Poetic Estonian compounds and formations

### ESTONIAN DIALECT-SPECIFIC ANALYSIS FRAMEWORK

For each Estonian dialect encountered, conduct a comprehensive internal examination of:

1. Estonian Dialect Geographic & Historical Context:
   - Estonian regional distribution and boundaries
   - Historical development of Estonian dialects and influences
   - Contact phenomena with neighboring languages/dialects
   - Sociolinguistic status and usage domains of Estonian dialects

2. Estonian Phonological System:
   - Estonian vowel inventory and distinctive features
   - Estonian consonant inventory and distinctive features
   - Estonian prosodic characteristics (stress, quantity, tone)
   - Estonian phonotactic constraints and syllable structure
   - Estonian sandhi phenomena and morphophonological alternations

3. Estonian Vowel System & Patterns:
   - Estonian three-way length distinctions
   - Estonian õ vowel distribution and usage
   - Estonian diphthong systems and historical developments
   - Estonian vowel length contrasts and their phonemic status
   - Estonian dialectal vowel shifts and mergers
   - Estonian-specific vowels and their distribution

4. Estonian Consonant System & Gradation:
   - Estonian consonant gradation patterns and exceptions
   - Estonian palatalization phenomena and distributional rules
   - Estonian consonant clusters and their historical development
   - Estonian h-retention or loss in various positions
   - Estonian assimilation and dissimilation processes
   - Estonian dialectal consonant shifts and mergers

5. Estonian Morphological Structure:
   - Estonian case system variations and usage patterns
   - Estonian possessive suffix distribution
   - Estonian nominal derivation particularities
   - Estonian number marking systems (singular, plural)
   - Estonian compound formation principles

6. Estonian Verbal Morphology:
   - Estonian person marking variations
   - Estonian tense-aspect-mood system particularities
   - Estonian non-finite verb form usage
   - Estonian negation constructions
   - Estonian passive and middle voice formations
   - Estonian reflexive and causative derivations

7. Estonian Lexical Features:
   - Estonian dialect-specific vocabulary items
   - Semantic shifts in Estonian dialect words
   - Estonian loanword strata (Germanic, Slavic, etc.)
   - Estonian archaic lexical retentions
   - Estonian innovative lexical formations
   - Estonian taboo replacements and euphemisms

For each line or section, wrap a short summary of your Estonian analysis in <linguistic_breakdown> tags to break down your thought process. 
Then, provide the revised translation that addresses the identified Estonian issues while maintaining the Estonian cultural and poetic elements of the original text.

Purpose of Translation:
These translations are intended to provide English-speaking researchers with accurate representations of Estonian runosongs vocabulary and semantics. 
When encountering choices between preserving poetic elements (such as rhyme) and literal meaning, highly prioritize Estonian semantic accuracy.
Please do not use rhyme at all in translations as for our purposes conveying exact literal meaning of Estonian terms is of utmost importance.
Maintain a consistent vocabulary that closely mirrors the original Estonian terms.
Focus on producing translations that support statistical and computational analysis of the translated Estonian song texts. 
While respecting the line breaks and structure of the original Estonian songs, avoid introducing additional artistic elements or synonyms for stylistic purposes.
The goal is to create a reliable dataset that reflects the authentic Estonian word usage patterns and semantic relationships present in the source material.

Please follow strictly this output format:

<linguistic_breakdown>
[Your short analysis of the original Estonian text and translation]
</linguistic_breakdown>

[Improved English translation with line breaks preserved exactly as in the original translation]

Do not add any comment or note the improved translation provided. Begin your analysis and revision of the Estonian translation now.
"""

    def _get_tabular_analysis_prompt(self):
        """Return the word-by-word tabular analysis prompt for Estonian"""
        return """
You are an expert in Estonian linguistics with deep knowledge of Estonian dialectal variations, including standard Estonian, South Estonian (Võro, Seto), island dialects, and historical Estonian forms.

Your task is to analyze the provided Estonian runosong text and its translation to create a comprehensive word-by-word analysis table.

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

## Column Specifications

### original form
- Exact word as it appears in the text
- Preserve all diacritics, archaic spellings, and dialect features

### comment
- Brief morphological analysis
- Dialect identification notes
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
- Contemporary standard Estonian lemma
- Modern Estonian dictionary form (as found in EKSS, ma-infinitive for verbs, nominative form for nouns etc)
- Standard Estonian reference form for linguistic analysis

### etymological root
- Proto-Finnic or other reconstructed forms when known
- Related forms in sister languages
- Borrowing source if applicable

# SPECIAL CONSIDERATIONS

## Dialectal Features
- Recognize Estonian vowel harmony variations and Finnish influence
- Identify Estonian consonant gradation patterns
- Note archaic Estonian case endings and verb forms
- Consider regional Estonian phonological changes

## Cultural Context
- Understand Estonian runosong-specific vocabulary
- Recognize Estonian formulaic expressions
- Consider Estonian ritual and ceremonial language
- Account for poetic and metaphorical usage in Estonian tradition

## Estonian Lemmatization Requirements
- Always provide Estonian lemmas, never Finnish cognates
- Map dialectal forms to standard Estonian dictionary forms
- Consider Estonian morphological patterns and historical development
- Use Estonian linguistic standards (EKSS) for modern lemmas

## Morphological Complexity
- Analyze compound words appropriately
- Identify diminutive and augmentative forms
- Recognize archaic morphological patterns
- Consider cross-linguistic influences

CRITICAL REQUIREMENT: You MUST analyze EVERY SINGLE WORD in the text. Create a complete table with ALL words included - do not skip any words or abbreviate the analysis. The table must be complete and comprehensive, covering every word from start to finish. DO NOT ask if you should continue - just provide the complete analysis for ALL words in the text. Never truncate or summarize - analyze every single word.

Provide the analysis table for the given text, ensuring each word receives thorough linguistic attention.
"""

if __name__ == "__main__":
    # This section demonstrates how to use the EstonianPromptTester with your specific data
    
    # Option 1: Provide API key directly (replace with your actual key)
    # API_KEY = "your_api_key_here"
    
    # Option 2: Set environment variable before running (recommended)
    # import os
    # os.environ['ANTHROPIC_API_KEY'] = "your_api_key_here"
    
    try:
        # Initialize Estonian tester with API key
        tester = EstonianPromptTester()  # Will use environment variable
        
        # Define metadata columns to use
        metadata_columns = ["dialect", "region", "collector", "year", "typeNames"]
        
        # STEP 1: Create input CSV from text files
        tester.create_input_csv_from_txt_files(
            input_folder="evaluation_set",
            output_csv="estonian_evaluation_input.csv"
        )
        
        # STEP 2: Process Estonian texts with the default tabular analysis prompt
        tester.process_texts(
            input_file="estonian_evaluation_input.csv",
            output_file="Estonian_Word_Analysis_Results.csv",
            prompt_name="tabular_analysis",
            metadata_columns=metadata_columns
        )
        
    except Exception as e:
        logging.error(f"Fatal error in Estonian processing: {str(e)}")
        sys.exit(1)