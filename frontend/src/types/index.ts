/**
 * Type definitions for Sanskrit OCR application
 */

export interface OCRResponse {
  success: boolean
  text: string
  original_ocr: string
  source_info?: {
    source_book: string
    source_chapter: string
    source_verse: string
    confidence: number
    match_type: string
  }
  image_info: {
    width: number
    height: number
    mode: string
    stats: ImageStats
  }
  processing_info: {
    database_match: boolean
    variants_processed: number
    estimated_lines: number
    text_density: number
    quality_score: number
    format_converted?: boolean
    detected_format?: string
    output_format?: string
    processing_method?: string
  }
}

export interface TextProcessResponse {
  success: boolean
  text: string
  original_text: string
  source_info?: {
    source_book: string
    source_chapter: string
    source_verse: string
    confidence: number
    match_type: string
  }
  text_info: {
    word_count: number
    character_count: number
    detected_format: string
    output_format: string
  }
  processing_info: {
    database_match: boolean
    format_converted: boolean
    input_type: string
  }
}

export interface ConvertTextResponse {
  success: boolean
  original_text: string
  converted_text: string
  source_format: string
  target_format: string
  detected_format: string
}

export interface ImageStats {
  mean_intensity: number
  std_intensity: number
  contrast: number
  text_density: number
  line_count: number
  character_density: number
}

export interface BookUploadResponse {
  success: boolean
  message: string
  texts_added: number
  total_texts_extracted: number
  database_stats: DatabaseStats
}

export interface DatabaseStats {
  texts: number
  unique_words: number
  books: number
  total_words: number
}

export interface LoginResponse {
  success: boolean
  token?: string
  message: string
}

export interface SanskritText {
  id: number
  text: string
  source_book: string
  source_chapter: string
  source_verse: string
  text_type: string
  language_script: string
  word_count: number
}

export interface SearchResult {
  id: number
  text: string
  source_book: string
  source_chapter: string
  source_verse: string
  text_type: string
  language_script: string
  word_count: number
}

// Internationalization types
export type SupportedLanguage = 'ru' | 'en'

export interface LanguageContextType {
  currentLanguage: SupportedLanguage
  setLanguage: (language: SupportedLanguage) => void
  t: (key: string) => string
}

export interface Translations {
  [key: string]: string
}

export interface SearchResponse {
  success: boolean
  results: SearchResult[]
  total_found: number
}

export interface SupportedFormatsResponse {
  success: boolean
  supported_formats: string[]
}

/**
 * Available tab types for the application
 */
export type TabType = 'ocr' | 'database'

/**
 * Available input types for processing
 */
export type InputType = 'image' | 'text'

/**
 * Available output formats for text conversion
 */
export type OutputFormat = 'russian' | 'iast'

/**
 * Available input formats for image OCR - format/language of text on the image
 */
export type InputFormat = 'russian_diacritics' | 'english_diacritics'

export interface AppState {
  activeTab: TabType
  selectedFile: File | null
  selectedBook: File | null
  recognizedText: string
  originalOCR: string
  sourceInfo: OCRResponse['source_info'] | null
  isProcessing: boolean
  isUploadingBook: boolean
  error: string
  bookError: string
  dragActive: boolean
  bookDragActive: boolean
  dbStats: DatabaseStats | null
  uploadSuccess: string
  
  // Admin authentication state
  isAdminAuthenticated: boolean
  adminToken: string
  adminPassword: string
  isLoggingIn: boolean
  loginError: string
  showAdminLogin: boolean
} 