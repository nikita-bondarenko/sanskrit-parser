/**
 * Book Upload Component
 */

import { useState, useCallback } from 'preact/hooks'
import { DatabaseStats } from '../types'
import { useAPI } from '../hooks/useAPI'
import { useLanguage } from '../contexts/LanguageContext'
import { formatFileSize, isSupportedDocumentFile, validateFileSize, getErrorMessage } from '../utils'

interface BookUploadProps {
  adminToken: string
  onStatsUpdate: (stats: DatabaseStats) => void
}

export const BookUpload = ({ adminToken, onStatsUpdate }: BookUploadProps) => {
  const [selectedBook, setSelectedBook] = useState<File | null>(null)
  const [isUploadingBook, setIsUploadingBook] = useState<boolean>(false)
  const [bookError, setBookError] = useState<string>('')
  const [uploadSuccess, setUploadSuccess] = useState<string>('')
  const [bookDragActive, setBookDragActive] = useState<boolean>(false)

  const { uploadBook } = useAPI()
  const { t } = useLanguage()

  const handleBookSelect = useCallback((event: Event) => {
    const target = event.target as HTMLInputElement
    const file = target.files?.[0]
    if (file) {
      if (!isSupportedDocumentFile(file)) {
        setBookError('Please select a supported document file (PDF, DOCX, TXT, DOC, RTF, ODT)')
        return
      }
      if (!validateFileSize(file, 50)) { // 50MB limit for books
        setBookError('File size must be less than 50MB')
        return
      }
      setSelectedBook(file)
      setBookError('')
      setUploadSuccess('')
    }
  }, [])

  const handleBookDrop = useCallback((event: DragEvent) => {
    event.preventDefault()
    setBookDragActive(false)
    
    const file = event.dataTransfer?.files[0]
    if (file) {
      if (!isSupportedDocumentFile(file)) {
        setBookError('Please select a supported document file (PDF, DOCX, TXT, DOC, RTF, ODT)')
        return
      }
      if (!validateFileSize(file, 50)) {
        setBookError('File size must be less than 50MB')
        return
      }
      setSelectedBook(file)
      setBookError('')
      setUploadSuccess('')
    }
  }, [])

  const handleBookDragOver = useCallback((event: DragEvent) => {
    event.preventDefault()
    setBookDragActive(true)
  }, [])

  const handleBookDragLeave = useCallback((event: DragEvent) => {
    event.preventDefault()
    setBookDragActive(false)
  }, [])

  const handleUploadBook = useCallback(async () => {
    if (!selectedBook || !adminToken) return

    setIsUploadingBook(true)
    setBookError('')
    setUploadSuccess('')

    try {
      const data = await uploadBook(selectedBook, adminToken)
      
      if (data.success) {
        setUploadSuccess(`Successfully added ${data.texts_added} texts from ${selectedBook.name}`)
        onStatsUpdate(data.database_stats)
        setSelectedBook(null)
      } else {
        setBookError('Failed to process book')
      }
    } catch (err) {
      setBookError(getErrorMessage(err))
    } finally {
      setIsUploadingBook(false)
    }
  }, [selectedBook, adminToken, uploadBook, onStatsUpdate])

  return (
    <>
      {/* Book Upload Area */}
      <div
        className={`border-2 border-dashed rounded-xl p-8 mb-6 text-center transition-all duration-300 ${
          bookDragActive 
            ? 'border-purple-400 bg-purple-50/10' 
            : 'border-white/30 hover:border-white/50'
        }`}
        onDrop={handleBookDrop}
        onDragOver={handleBookDragOver}
        onDragLeave={handleBookDragLeave}
      >
        <div className="text-white/80 mb-4">
          <svg 
            className="w-12 h-12 mx-auto mb-4" 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={2} 
              d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" 
            />
          </svg>
          <p className="text-lg mb-2">
            Drag & drop your Sanskrit books here
          </p>
          <p className="text-sm text-white/60">
            Supports PDF, DOCX, TXT, DOC, RTF, ODT formats (max 50MB)
          </p>
        </div>
        
        <input
          type="file"
          accept=".pdf,.docx,.txt,.doc,.rtf,.odt"
          onChange={handleBookSelect}
          className="hidden"
          id="book-input"
        />
        <label
          htmlFor="book-input"
          className="inline-block bg-purple-500 hover:bg-purple-600 text-white px-6 py-2 rounded-lg cursor-pointer transition-colors"
        >
          Select Book
        </label>
      </div>

      {/* Selected Book Info */}
      {selectedBook && (
        <div className="bg-white/10 rounded-lg p-4 mb-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-white font-medium">{selectedBook.name}</p>
              <p className="text-white/60 text-sm">
                {formatFileSize(selectedBook.size)}
              </p>
            </div>
            <button
              onClick={handleUploadBook}
              disabled={isUploadingBook}
              className="bg-green-500 hover:bg-green-600 disabled:bg-gray-400 text-white px-6 py-2 rounded-lg transition-colors flex items-center gap-2"
            >
              {isUploadingBook ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
                  Processing...
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                  Upload Book
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {/* Book Upload Messages */}
      {bookError && (
        <div className="bg-red-500/20 border border-red-500/50 rounded-lg p-4 mb-6">
          <p className="text-red-200">{bookError}</p>
        </div>
      )}

      {uploadSuccess && (
        <div className="bg-green-500/20 border border-green-500/50 rounded-lg p-4 mb-6">
          <p className="text-green-200">{uploadSuccess}</p>
        </div>
      )}
    </>
  )
} 