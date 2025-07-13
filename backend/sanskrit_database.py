"""
Sanskrit Text Database Module
Manages collection and analysis of Sanskrit texts for OCR improvement
"""

import sqlite3
import re
import unicodedata
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz, process
import logging

logger = logging.getLogger(__name__)

@dataclass
class SanskritText:
    """Represents a Sanskrit text entry"""
    id: Optional[int] = None
    text: str = ""
    source_book: str = ""
    source_chapter: str = ""
    source_verse: str = ""
    text_type: str = ""  # verse, prose, mantra, etc.
    language_script: str = ""  # iast, devanagari, russian_diacritics
    word_count: int = 0
    character_count: int = 0

@dataclass
class TextMatch:
    """Represents a match found in the database"""
    matched_text: str
    original_text: str
    source_book: str
    source_chapter: str
    source_verse: str
    confidence: float
    match_type: str  # exact, fuzzy, partial

class SanskritDatabase:
    """Manages Sanskrit text database operations"""
    
    def __init__(self, db_path: str = "sanskrit_texts.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create main texts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sanskrit_texts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                normalized_text TEXT NOT NULL,
                source_book TEXT NOT NULL,
                source_chapter TEXT DEFAULT '',
                source_verse TEXT DEFAULT '',
                text_type TEXT DEFAULT 'verse',
                language_script TEXT DEFAULT 'russian_diacritics',
                word_count INTEGER DEFAULT 0,
                character_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create words table for faster searching
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sanskrit_words (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT NOT NULL,
                normalized_word TEXT NOT NULL,
                text_id INTEGER,
                frequency INTEGER DEFAULT 1,
                FOREIGN KEY (text_id) REFERENCES sanskrit_texts (id)
            )
        ''')
        
        # Create phrases table for common phrases
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sanskrit_phrases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phrase TEXT NOT NULL,
                normalized_phrase TEXT NOT NULL,
                text_id INTEGER,
                phrase_length INTEGER DEFAULT 0,
                FOREIGN KEY (text_id) REFERENCES sanskrit_texts (id)
            )
        ''')
        
        # Create indexes for faster searching
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_normalized_text ON sanskrit_texts(normalized_text)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_normalized_word ON sanskrit_words(normalized_word)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_normalized_phrase ON sanskrit_phrases(normalized_phrase)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_source_book ON sanskrit_texts(source_book)')
        
        conn.commit()
        conn.close()
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for better matching"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize Unicode characters
        text = unicodedata.normalize('NFC', text)
        
        # Convert to lowercase for comparison
        text = text.lower()
        
        # Remove punctuation for normalization
        text = re.sub(r'[^\w\s\u0900-\u097F\u0400-\u04FF]', ' ', text)
        
        return text
    
    def extract_words(self, text: str) -> List[str]:
        """Extract individual words from text"""
        normalized = self.normalize_text(text)
        words = re.findall(r'\b\w+\b', normalized)
        return [word for word in words if len(word) > 2]  # Filter short words
    
    def extract_phrases(self, text: str, min_length: int = 3, max_length: int = 8) -> List[str]:
        """Extract phrases of different lengths"""
        words = self.extract_words(text)
        phrases = []
        
        for length in range(min_length, min(max_length + 1, len(words) + 1)):
            for i in range(len(words) - length + 1):
                phrase = ' '.join(words[i:i + length])
                phrases.append(phrase)
        
        return phrases
    
    def add_text(self, sanskrit_text: SanskritText) -> int:
        """Add a new Sanskrit text to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Normalize text
        normalized_text = self.normalize_text(sanskrit_text.text)
        
        # Calculate statistics
        words = self.extract_words(sanskrit_text.text)
        word_count = len(words)
        character_count = len(sanskrit_text.text)
        
        # Insert main text
        cursor.execute('''
            INSERT INTO sanskrit_texts 
            (text, normalized_text, source_book, source_chapter, source_verse, 
             text_type, language_script, word_count, character_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            sanskrit_text.text, normalized_text, sanskrit_text.source_book,
            sanskrit_text.source_chapter, sanskrit_text.source_verse,
            sanskrit_text.text_type, sanskrit_text.language_script,
            word_count, character_count
        ))
        
        text_id = cursor.lastrowid
        
        # Insert words
        word_freq = {}
        for word in words:
            normalized_word = self.normalize_text(word)
            word_freq[normalized_word] = word_freq.get(normalized_word, 0) + 1
        
        for word, freq in word_freq.items():
            cursor.execute('''
                INSERT INTO sanskrit_words (word, normalized_word, text_id, frequency)
                VALUES (?, ?, ?, ?)
            ''', (word, word, text_id, freq))
        
        # Insert phrases
        phrases = self.extract_phrases(sanskrit_text.text)
        for phrase in phrases:
            normalized_phrase = self.normalize_text(phrase)
            cursor.execute('''
                INSERT INTO sanskrit_phrases (phrase, normalized_phrase, text_id, phrase_length)
                VALUES (?, ?, ?, ?)
            ''', (phrase, normalized_phrase, text_id, len(phrase.split())))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Added text from {sanskrit_text.source_book} with {word_count} words")
        return text_id
    
    def find_best_match(self, input_text: str, min_confidence: float = 0.7) -> Optional[TextMatch]:
        """Find the best matching text in the database"""
        normalized_input = self.normalize_text(input_text)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        best_match = None
        best_confidence = 0.0
        
        # Try exact match first
        cursor.execute('''
            SELECT text, source_book, source_chapter, source_verse
            FROM sanskrit_texts
            WHERE normalized_text = ?
        ''', (normalized_input,))
        
        result = cursor.fetchone()
        if result:
            conn.close()
            return TextMatch(
                matched_text=result[0],
                original_text=input_text,
                source_book=result[1],
                source_chapter=result[2],
                source_verse=result[3],
                confidence=1.0,
                match_type="exact"
            )
        
        # Try fuzzy matching on phrases
        input_phrases = self.extract_phrases(input_text, min_length=4)
        
        for phrase in input_phrases:
            normalized_phrase = self.normalize_text(phrase)
            
            cursor.execute('''
                SELECT p.phrase, t.text, t.source_book, t.source_chapter, t.source_verse
                FROM sanskrit_phrases p
                JOIN sanskrit_texts t ON p.text_id = t.id
                WHERE p.normalized_phrase LIKE ?
                ORDER BY p.phrase_length DESC
                LIMIT 10
            ''', (f'%{normalized_phrase}%',))
            
            results = cursor.fetchall()
            
            for result in results:
                db_phrase, full_text, book, chapter, verse = result
                
                # Calculate similarity
                similarity = fuzz.ratio(normalized_phrase, self.normalize_text(db_phrase)) / 100.0
                
                if similarity > best_confidence and similarity >= min_confidence:
                    best_confidence = similarity
                    best_match = TextMatch(
                        matched_text=full_text,
                        original_text=input_text,
                        source_book=book,
                        source_chapter=chapter,
                        source_verse=verse,
                        confidence=similarity,
                        match_type="fuzzy"
                    )
        
        # Try partial matching on full texts
        if not best_match or best_confidence < 0.8:
            cursor.execute('''
                SELECT text, source_book, source_chapter, source_verse
                FROM sanskrit_texts
                WHERE normalized_text LIKE ?
                ORDER BY character_count DESC
                LIMIT 5
            ''', (f'%{normalized_input}%',))
            
            results = cursor.fetchall()
            
            for result in results:
                db_text, book, chapter, verse = result
                
                # Calculate similarity
                similarity = fuzz.partial_ratio(normalized_input, self.normalize_text(db_text)) / 100.0
                
                if similarity > best_confidence and similarity >= min_confidence:
                    best_confidence = similarity
                    best_match = TextMatch(
                        matched_text=db_text,
                        original_text=input_text,
                        source_book=book,
                        source_chapter=chapter,
                        source_verse=verse,
                        confidence=similarity,
                        match_type="partial"
                    )
        
        conn.close()
        return best_match
    
    def get_statistics(self) -> Dict[str, int]:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM sanskrit_texts')
        text_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT normalized_word) FROM sanskrit_words')
        unique_words = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT source_book) FROM sanskrit_texts')
        book_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT SUM(word_count) FROM sanskrit_texts')
        total_words = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            'texts': text_count,
            'unique_words': unique_words,
            'books': book_count,
            'total_words': total_words
        }
    
    def search_by_source(self, book_name: str) -> List[SanskritText]:
        """Search texts by source book"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, text, source_book, source_chapter, source_verse, 
                   text_type, language_script, word_count, character_count
            FROM sanskrit_texts
            WHERE source_book LIKE ?
            ORDER BY source_chapter, source_verse
        ''', (f'%{book_name}%',))
        
        results = cursor.fetchall()
        conn.close()
        
        texts = []
        for result in results:
            texts.append(SanskritText(
                id=result[0],
                text=result[1],
                source_book=result[2],
                source_chapter=result[3],
                source_verse=result[4],
                text_type=result[5],
                language_script=result[6],
                word_count=result[7],
                character_count=result[8]
            ))
        
        return texts

# Global database instance
sanskrit_db = SanskritDatabase() 