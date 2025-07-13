/**
 * Utility functions for the application
 */

/**
 * Copy text to clipboard
 */
export const copyToClipboard = async (text: string): Promise<boolean> => {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch (err) {
    console.error('Failed to copy text:', err)
    return false
  }
}

/**
 * Format file size in human readable format
 */
export const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes'
  
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

/**
 * Check if file is an image
 */
export const isImageFile = (file: File): boolean => {
  return file.type.startsWith('image/')
}

/**
 * Check if file is a supported document format
 */
export const isSupportedDocumentFile = (file: File): boolean => {
  const supportedExtensions = ['.pdf', '.docx', '.txt', '.doc', '.rtf', '.odt']
  const fileName = file.name.toLowerCase()
  return supportedExtensions.some(ext => fileName.endsWith(ext))
}

/**
 * Get file extension from filename
 */
export const getFileExtension = (filename: string): string => {
  return filename.split('.').pop()?.toLowerCase() || ''
}

/**
 * Validate admin password format
 */
export const validatePassword = (password: string): boolean => {
  return password.length >= 3 // Basic validation
}

/**
 * Format confidence percentage
 */
export const formatConfidence = (confidence: number): string => {
  return `${(confidence * 100).toFixed(1)}%`
}

/**
 * Truncate text to specified length
 */
export const truncateText = (text: string, maxLength: number): string => {
  if (text.length <= maxLength) return text
  return text.substring(0, maxLength) + '...'
}

/**
 * Debounce function for search inputs
 */
export const debounce = <T extends (...args: any[]) => any>(
  func: T,
  wait: number
): ((...args: Parameters<T>) => void) => {
  let timeout: NodeJS.Timeout | null = null
  
  return (...args: Parameters<T>) => {
    if (timeout) clearTimeout(timeout)
    timeout = setTimeout(() => func(...args), wait)
  }
}

/**
 * Check if text contains Sanskrit diacritics
 */
export const containsSanskritDiacritics = (text: string): boolean => {
  const diacriticChars = new Set('а̄ӣӯр̣л̣н̣т̣д̣ш́ṃāīūṛṝḷḹṅñṭḍṇśṣḥṁṃäïüëöçñåøæ')
  return [...text].some(char => diacriticChars.has(char))
}

/**
 * Get text statistics
 */
export const getTextStats = (text: string) => {
  const lines = text.split('\n').filter(line => line.trim())
  const words = text.split(/\s+/).filter(word => word.trim())
  const characters = text.length
  const hasDiacritics = containsSanskritDiacritics(text)
  
  return {
    lines: lines.length,
    words: words.length,
    characters,
    hasDiacritics
  }
}

/**
 * Format date for display
 */
export const formatDate = (date: Date): string => {
  return new Intl.DateTimeFormat('ru-RU', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  }).format(date)
}

/**
 * Generate random ID
 */
export const generateId = (): string => {
  return Math.random().toString(36).substr(2, 9)
}

/**
 * Validate file size (max 10MB)
 */
export const validateFileSize = (file: File, maxSizeMB: number = 10): boolean => {
  const maxSizeBytes = maxSizeMB * 1024 * 1024
  return file.size <= maxSizeBytes
}

/**
 * Get error message from error object
 */
export const getErrorMessage = (error: unknown): string => {
  if (error instanceof Error) {
    return error.message
  }
  if (typeof error === 'string') {
    return error
  }
  return 'Unknown error occurred'
} 