import { Translations } from '../types'

export const enTranslations: Translations = {
  // Header
  'app.title': 'Sanskrit OCR & Converter',
  'app.subtitle': 'OCR, Text Processing & Format Conversion',
  'app.description': 'Process images or text • Convert between Russian and IAST diacritics • Database lookup',
  
  // Navigation
  'nav.ocr': 'OCR Recognition',
  'nav.database': 'Database Management',
  'nav.admin': 'Admin',
  
  // Input Types
  'input.type.title': 'Input Type',
  'input.type.image': 'Image',
  'input.type.text': 'Text',
  
  // Input Format
  'input.format.title': 'Image Text Format',
  'input.format.english': 'English Diacritics',
  'input.format.russian': 'Russian Diacritics',
  'input.format.description': 'Select the type of diacritics in your image: English (ā ī ū ṛ ś ṣ) or Russian (а̄ ӣ ӯ р̣ ш́)',
  
  // Output Format
  'output.format.title': 'Output Format',
  'output.format.russian': 'Russian Diacritics',
  'output.format.iast': 'IAST (English)',
  
  // Image Upload
  'upload.drag.title': 'Drag & drop your Sanskrit image here',
  'upload.drag.subtitle': 'or click to select a file (max 10MB)',
  'upload.select': 'Select Image',
  'upload.processing': 'Processing...',
  'upload.process': 'Process Image',
  
  // Text Input
  'text.input.label': 'Enter Sanskrit Text ({format} or any format)',
  'text.input.placeholder': 'Enter Sanskrit text in any format (IAST, Russian diacritics, etc.)...',
  'text.characters': 'Characters: {count}',
  'text.process': 'Process Text',
  
  // Results
  'results.corrected': 'Corrected Text (from Database)',
  'results.processed': 'Processed Text ({format})',
  'results.copy': 'Copy',
  'results.clear': 'Clear',
  
  // Source Info
  'source.found': '📚 Source Found in Database',
  'source.book': 'Book:',
  'source.chapter': 'Chapter:',
  'source.verse': 'Verse:',
  'source.confidence': 'Confidence:',
  'source.match': 'Match Type:',
  
  // Database Stats
  'stats.texts': 'Texts',
  'stats.books': 'Books',
  'stats.unique_words': 'Unique Words',
  'stats.total_words': 'Total Words',
  
  // Admin Panel
  'admin.access.granted': '✅ Admin Access Granted',
  'admin.access.description': 'You can now upload books to the database.',
  'admin.logout': 'Logout',
  'admin.access.required': '🔐 Admin Access Required',
  'admin.access.restriction': 'Book upload is restricted to administrators only. Please log in to access database management features.',
  'admin.login': 'Admin Login',
  'admin.cancel': 'Cancel',
  'admin.password.placeholder': 'Enter admin password',
  'admin.logging': 'Logging in...',
  'admin.login.button': 'Login',
  
  // Book Upload
  'book.upload.title': 'Drag & drop your Sanskrit books here',
  'book.upload.subtitle': 'Supports PDF, DOCX, TXT, DOC, RTF, ODT formats (max 50MB)',
  'book.select': 'Select Book',
  'book.upload.processing': 'Processing...',
  'book.upload.button': 'Upload Book',
  'book.upload.success': 'Successfully added {count} texts from {filename}',
  
  // Footer
  'footer.description': 'Image OCR • Text Processing • Format Conversion (IAST ⇄ Russian) • Database Lookup • Multiple Input Methods',
  'footer.contact': 'For any questions, please contact us via Telegram or email:',
  
  // Error Messages
  'error.image.format': 'Please select an image file',
  'error.file.size': 'File size must be less than 10MB',
  'error.file.size.book': 'File size must be less than 50MB',
  'error.document.format': 'Please select a supported document file (PDF, DOCX, TXT, DOC, RTF, ODT)',
  'error.process.image': 'Failed to process image',
  'error.process.text': 'Failed to process text',
  'error.process.book': 'Failed to process book',
  'error.copy.clipboard': 'Failed to copy text to clipboard'
} 