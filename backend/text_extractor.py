"""
Text Extraction Module
Extracts text from various file formats for Sanskrit text analysis
"""

import io
import re
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import chardet

# Import libraries for different file formats
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import textract
    TEXTRACT_AVAILABLE = True
except ImportError:
    TEXTRACT_AVAILABLE = False

from sanskrit_database import SanskritText

logger = logging.getLogger(__name__)

class TextExtractor:
    """Handles text extraction from various file formats"""
    
    def __init__(self):
        self.supported_formats = {
            '.txt': self.extract_from_txt,
            '.pdf': self.extract_from_pdf if PDF_AVAILABLE else None,
            '.docx': self.extract_from_docx if DOCX_AVAILABLE else None,
            '.doc': self.extract_from_doc if TEXTRACT_AVAILABLE else None,
            '.rtf': self.extract_from_rtf if TEXTRACT_AVAILABLE else None,
            '.odt': self.extract_from_odt if TEXTRACT_AVAILABLE else None,
        }
    
    def detect_encoding(self, file_content: bytes) -> str:
        """Detect file encoding"""
        try:
            result = chardet.detect(file_content)
            return result['encoding'] or 'utf-8'
        except:
            return 'utf-8'
    
    def extract_from_txt(self, file_content: bytes, filename: str) -> List[str]:
        """Extract text from TXT file"""
        try:
            encoding = self.detect_encoding(file_content)
            text = file_content.decode(encoding)
            
            # Split by paragraphs or verses
            paragraphs = self.split_into_paragraphs(text)
            return paragraphs
            
        except Exception as e:
            logger.error(f"Error extracting from TXT {filename}: {e}")
            return []
    
    def extract_from_pdf(self, file_content: bytes, filename: str) -> List[str]:
        """Extract text from PDF file"""
        if not PDF_AVAILABLE:
            logger.error("PyPDF2 not available for PDF extraction")
            return []
        
        try:
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            all_text = ""
            for page in pdf_reader.pages:
                all_text += page.extract_text() + "\n"
            
            paragraphs = self.split_into_paragraphs(all_text)
            return paragraphs
            
        except Exception as e:
            logger.error(f"Error extracting from PDF {filename}: {e}")
            return []
    
    def extract_from_docx(self, file_content: bytes, filename: str) -> List[str]:
        """Extract text from DOCX file"""
        if not DOCX_AVAILABLE:
            logger.error("python-docx not available for DOCX extraction")
            return []
        
        try:
            docx_file = io.BytesIO(file_content)
            doc = Document(docx_file)
            
            paragraphs = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    paragraphs.append(paragraph.text.strip())
            
            return paragraphs
            
        except Exception as e:
            logger.error(f"Error extracting from DOCX {filename}: {e}")
            return []
    
    def extract_from_doc(self, file_content: bytes, filename: str) -> List[str]:
        """Extract text from DOC file using textract"""
        if not TEXTRACT_AVAILABLE:
            logger.error("textract not available for DOC extraction")
            return []
        
        try:
            # Save to temporary file for textract
            temp_path = f"/tmp/{filename}"
            with open(temp_path, 'wb') as f:
                f.write(file_content)
            
            text = textract.process(temp_path).decode('utf-8')
            paragraphs = self.split_into_paragraphs(text)
            
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)
            
            return paragraphs
            
        except Exception as e:
            logger.error(f"Error extracting from DOC {filename}: {e}")
            return []
    
    def extract_from_rtf(self, file_content: bytes, filename: str) -> List[str]:
        """Extract text from RTF file"""
        return self.extract_from_doc(file_content, filename)
    
    def extract_from_odt(self, file_content: bytes, filename: str) -> List[str]:
        """Extract text from ODT file"""
        return self.extract_from_doc(file_content, filename)
    
    def split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into meaningful paragraphs/verses"""
        # Clean up text
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\r', '\n', text)
        
        # Split by double newlines (paragraph breaks)
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Filter out empty paragraphs and very short ones
        filtered_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if len(para) > 20:  # Minimum length for meaningful text
                filtered_paragraphs.append(para)
        
        return filtered_paragraphs
    
    def detect_text_structure(self, text: str) -> Dict[str, str]:
        """Detect structure elements in text (chapters, verses, etc.)"""
        structure = {
            'chapter': '',
            'verse': '',
            'type': 'prose'
        }
        
        # Look for chapter indicators
        chapter_patterns = [
            r'(?:Chapter|Глава|Adhyaya)\s*(\d+)',
            r'(?:Ch\.|Гл\.)\s*(\d+)',
            r'Адхьяя\s*(\d+)',
        ]
        
        for pattern in chapter_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                structure['chapter'] = match.group(1)
                break
        
        # Look for verse indicators
        verse_patterns = [
            r'(?:Verse|Стих|Шлока)\s*(\d+)',
            r'(?:V\.|Ст\.)\s*(\d+)',
            r'(\d+)\.',  # Simple numbering
        ]
        
        for pattern in verse_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                structure['verse'] = match.group(1)
                break
        
        # Detect if it's verse or prose
        lines = text.split('\n')
        verse_indicators = 0
        
        for line in lines:
            line = line.strip()
            # Check for verse characteristics
            if len(line) > 20 and len(line) < 80:  # Typical verse length
                verse_indicators += 1
            if re.search(r'[ṃṅṇṭḍśṣḥ]', line):  # Sanskrit diacritics
                verse_indicators += 1
        
        if verse_indicators > len(lines) * 0.3:
            structure['type'] = 'verse'
        
        return structure
    
    def extract_book_metadata(self, text: str, filename: str) -> Dict[str, str]:
        """Extract book metadata from text"""
        metadata = {
            'title': '',
            'author': '',
            'translator': '',
            'publisher': '',
            'language': 'russian_diacritics'
        }
        
        # Look for title in first few lines
        lines = text.split('\n')[:10]
        for line in lines:
            line = line.strip()
            if len(line) > 10 and len(line) < 100:
                # Likely title if it's not too long/short
                if not metadata['title'] and any(word in line.lower() for word in ['гита', 'пурана', 'упанишад', 'веда']):
                    metadata['title'] = line
                    break
        
        # If no title found, use filename
        if not metadata['title']:
            metadata['title'] = Path(filename).stem
        
        # Look for author patterns
        author_patterns = [
            r'(?:Author|Автор|By)\s*:?\s*(.+)',
            r'(?:Translated by|Перевод)\s*:?\s*(.+)',
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata['author'] = match.group(1).strip()
                break
        
        # Detect language script
        if re.search(r'[āīūṛṝḷḹṅñṭḍṇśṣḥṁṃ]', text):
            metadata['language'] = 'iast'
        elif re.search(r'[äïüëöçñåøæ]', text):
            metadata['language'] = 'gaura_pt'
        elif re.search(r'[а-яё]', text):
            metadata['language'] = 'russian_diacritics'
        
        return metadata
    
    def process_file(self, file_content: bytes, filename: str) -> List[SanskritText]:
        """Process a file and extract Sanskrit texts"""
        file_extension = Path(filename).suffix.lower()
        
        if file_extension not in self.supported_formats:
            logger.error(f"Unsupported file format: {file_extension}")
            return []
        
        extractor = self.supported_formats[file_extension]
        if not extractor:
            logger.error(f"No extractor available for {file_extension}")
            return []
        
        # Extract paragraphs
        paragraphs = extractor(file_content, filename)
        if not paragraphs:
            return []
        
        # Get full text for metadata extraction
        full_text = '\n\n'.join(paragraphs)
        metadata = self.extract_book_metadata(full_text, filename)
        
        # Convert paragraphs to SanskritText objects
        sanskrit_texts = []
        
        for i, paragraph in enumerate(paragraphs):
            # Detect structure for this paragraph
            structure = self.detect_text_structure(paragraph)
            
            sanskrit_text = SanskritText(
                text=paragraph,
                source_book=metadata['title'],
                source_chapter=structure['chapter'],
                source_verse=structure['verse'] or str(i + 1),
                text_type=structure['type'],
                language_script=metadata['language']
            )
            
            sanskrit_texts.append(sanskrit_text)
        
        logger.info(f"Extracted {len(sanskrit_texts)} texts from {filename}")
        return sanskrit_texts
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        return [ext for ext, extractor in self.supported_formats.items() if extractor is not None]

# Global extractor instance
text_extractor = TextExtractor() 