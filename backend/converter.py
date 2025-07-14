"""
Sanskrit text format converter
Converts between IAST (English diacritics) and Russian diacritics
"""

import re
from typing import Dict, Union

# Unicode combining marks
MACRON = "\u0304"        # ̄ (combining macron)
DOT_BELOW = "\u0323"     # ̣ (combining dot below)
DOT_ABOVE = "\u0307"     # ̇ (combining dot above)
ACUTE = "\u0301"         # ́ (combining acute accent)
TILDE = "\u0303"         # ̃ (combining tilde)

# Russian to IAST mapping (reverse of TypeScript mapping)
RUS_TO_IAST: Dict[str, str] = {
    # Diphthongs
    'аи': 'ai',
    'ау': 'au',
    
    # Aspirated / compounds (need to be checked first due to length)
    'кх': 'kh',
    'гх': 'gh',
    'чх': 'ch',
    'джх': 'jh',
    f'т{DOT_BELOW}х': 'ṭh',
    f'д{DOT_BELOW}х': 'ḍh',
    'тх': 'th',
    'дх': 'dh',
    'пх': 'ph',
    'бх': 'bh',
    
    # Single consonants w/ diacritics
    f'н{DOT_ABOVE}': 'ṅ',
    f'н{TILDE}': 'ñ',
    f'т{DOT_BELOW}': 'ṭ',
    f'д{DOT_BELOW}': 'ḍ',
    f'н{DOT_BELOW}': 'ṇ',
    f'ш{ACUTE}': 'ś',
    f'ш{DOT_BELOW}': 'ṣ',
    f'х{DOT_BELOW}': 'ḥ',
    f'м{DOT_BELOW}': 'ṃ',
    f'м{DOT_ABOVE}': 'ṁ',
    
    # Plain consonants
    'к': 'k',
    'г': 'g',
    'ч': 'c',
    'дж': 'j',
    'т': 't',
    'д': 'd',
    'н': 'n',
    'п': 'p',
    'б': 'b',
    'м': 'm',
    'й': 'y',
    'р': 'r',
    'л': 'l',
    'в': 'v',
    'с': 's',
    'х': 'h',
    
    # Vowels with diacritics
    f'а{MACRON}': 'ā',
    f'и{MACRON}': 'ī',
    f'у{MACRON}': 'ū',
    f'р{DOT_BELOW}{MACRON}': 'ṝ',
    f'р{DOT_BELOW}': 'ṛ',
    f'л{DOT_BELOW}{MACRON}': 'ḹ',
    f'л{DOT_BELOW}': 'ḷ',
    
    # Plain vowels
    'а': 'a',
    'и': 'i',
    'у': 'u',
    'е': 'e',
    'о': 'o',
}

# IAST to Russian mapping (from TypeScript)
IAST_TO_RUS: Dict[str, str] = {
    # Diphthongs
    'ai': 'аи',
    'au': 'ау',
    
    # Aspirated / compounds
    'kh': 'кх',
    'gh': 'гх',
    'ch': 'чх',
    'jh': 'джх',
    'ṭh': f'т{DOT_BELOW}х',
    'ḍh': f'д{DOT_BELOW}х',
    'th': 'тх',
    'dh': 'дх',
    'ph': 'пх',
    'bh': 'бх',
    
    # Single consonants w/ diacritics
    'ṅ': f'н{DOT_ABOVE}',
    'ñ': f'н{TILDE}',
    'ṭ': f'т{DOT_BELOW}',
    'ḍ': f'д{DOT_BELOW}',
    'ṇ': f'н{DOT_BELOW}',
    'ś': f'ш{ACUTE}',
    'ṣ': f'ш{DOT_BELOW}',
    'ḥ': f'х{DOT_BELOW}',
    'ṃ': f'м{DOT_BELOW}',
    'ṁ': f'м{DOT_ABOVE}',
    
    # Plain consonants
    'k': 'к',
    'g': 'г',
    'c': 'ч',
    'j': 'дж',
    't': 'т',
    'd': 'д',
    'n': 'н',
    'p': 'п',
    'b': 'б',
    'm': 'м',
    'y': 'й',
    'r': 'р',
    'l': 'л',
    'v': 'в',
    's': 'с',
    'h': 'х',
    
    # Vowels with diacritics
    'ā': f'а{MACRON}',
    'ī': f'и{MACRON}',
    'ū': f'у{MACRON}',
    'ṝ': f'р{DOT_BELOW}{MACRON}',
    'ṛ': f'р{DOT_BELOW}',
    'ḹ': f'л{DOT_BELOW}{MACRON}',
    'ḷ': f'л{DOT_BELOW}',
    
    # Plain vowels
    'a': 'а',
    'i': 'и',
    'u': 'у',
    'e': 'е',
    'o': 'о',
}

def _apply_case(sample: str, replacement: str) -> str:
    """Apply case transformation based on the original sample"""
    if not sample or not replacement:
        return replacement
    
    # If sample is all caps
    if sample.isupper():
        return replacement.upper()
    # If only first letter capitalized
    elif sample[0].isupper():
        return replacement[0].upper() + replacement[1:]
    
    return replacement

def _create_regex_pattern(mapping: Dict[str, str]) -> re.Pattern:
    """Create regex pattern from mapping dictionary, longest first"""
    # Sort keys by length descending to ensure digraphs handled first
    sorted_keys = sorted(mapping.keys(), key=len, reverse=True)
    # Escape special regex characters
    escaped_keys = [re.escape(key) for key in sorted_keys]
    pattern = '|'.join(escaped_keys)
    return re.compile(pattern, re.IGNORECASE)

# Pre-compile regex patterns for efficiency
IAST_TO_RUS_PATTERN = _create_regex_pattern(IAST_TO_RUS)
RUS_TO_IAST_PATTERN = _create_regex_pattern(RUS_TO_IAST)

def convert_iast_to_russian(text: str) -> str:
    """
    Convert IAST Sanskrit string to Russian diacritic notation.
    
    Args:
        text: Input text in IAST format
        
    Returns:
        Text converted to Russian diacritics
    """
    if not text:
        return text
    
    # Normalize the input
    normalized_text = text.normalize('NFC') if hasattr(text, 'normalize') else text
    
    def replace_func(match):
        matched_text = match.group(0)
        lower_matched = matched_text.lower()
        replacement = IAST_TO_RUS.get(lower_matched, matched_text)
        return _apply_case(matched_text, replacement)
    
    return IAST_TO_RUS_PATTERN.sub(replace_func, normalized_text)

def convert_russian_to_iast(text: str) -> str:
    """
    Convert Russian diacritic Sanskrit string to IAST notation.
    
    Args:
        text: Input text in Russian diacritics format
        
    Returns:
        Text converted to IAST format
    """
    if not text:
        return text
    
    # Normalize the input
    normalized_text = text.normalize('NFC') if hasattr(text, 'normalize') else text
    
    def replace_func(match):
        matched_text = match.group(0)
        lower_matched = matched_text.lower()
        replacement = RUS_TO_IAST.get(lower_matched, matched_text)
        return _apply_case(matched_text, replacement)
    
    return RUS_TO_IAST_PATTERN.sub(replace_func, normalized_text)

def detect_text_format(text: str) -> str:
    """
    Detect if text is in IAST or Russian diacritics format.
    
    Args:
        text: Input text to analyze
        
    Returns:
        'iast', 'russian', or 'unknown'
    """
    if not text:
        return 'unknown'
    
    # Count IAST-specific characters
    iast_chars = set('āīūṛṝḷḹṅñṭḍṇśṣḥṃṁ')
    iast_count = sum(1 for char in text if char in iast_chars)
    
    # Count Russian-specific characters (Cyrillic)
    cyrillic_chars = set('абвгдежзийклмнопрстуфхцчшщъыьэюя')
    cyrillic_count = sum(1 for char in text.lower() if char in cyrillic_chars)
    
    # Count combining diacritics (used in Russian format)
    combining_diacritics = set([MACRON, DOT_BELOW, DOT_ABOVE, ACUTE, TILDE])
    diacritic_count = sum(1 for char in text if char in combining_diacritics)
    
    if iast_count > 0 and cyrillic_count == 0:
        return 'iast'
    elif (cyrillic_count > 0 or diacritic_count > 0) and iast_count == 0:
        return 'russian'
    elif cyrillic_count > iast_count:
        return 'russian'
    elif iast_count > cyrillic_count:
        return 'iast'
    else:
        return 'unknown'

def convert_text_format(text: str, target_format: str, source_format: str = None) -> str:
    """
    Convert text between IAST and Russian diacritics formats.
    
    Args:
        text: Input text to convert
        target_format: Target format ('iast' or 'russian')
        source_format: Source format (if None, will be auto-detected)
        
    Returns:
        Converted text
    """
    if not text:
        return text
    
    if source_format is None:
        source_format = detect_text_format(text)
    
    if source_format == target_format:
        return text
    
    if target_format == 'russian':
        if source_format == 'iast':
            return convert_iast_to_russian(text)
        else:
            # Already in Russian or unknown, return as is
            return text
    elif target_format == 'iast':
        if source_format == 'russian':
            return convert_russian_to_iast(text)
        else:
            # Already in IAST or unknown, return as is
            return text
    else:
        raise ValueError(f"Unsupported target format: {target_format}")

# Create a singleton instance for easy import
text_converter = type('TextConverter', (), {
    'convert_iast_to_russian': staticmethod(convert_iast_to_russian),
    'convert_russian_to_iast': staticmethod(convert_russian_to_iast),
    'detect_text_format': staticmethod(detect_text_format),
    'convert_text_format': staticmethod(convert_text_format),
})() 