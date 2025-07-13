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
  }
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

export interface SearchResponse {
  success: boolean
  results: SearchResult[]
  total_found: number
}

export interface SupportedFormatsResponse {
  success: boolean
  supported_formats: string[]
}

export type TabType = 'ocr' | 'database'

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