/**
 * OCR Tab Component
 */

import { useState, useCallback } from 'preact/hooks'
import { OCRResponse } from '../types'
import { useAPI } from '../hooks/useAPI'
import { copyToClipboard, formatFileSize, isImageFile, validateFileSize, getErrorMessage } from '../utils'

interface OCRTabProps {
  className?: string
}

export const OCRTab = ({ className = '' }: OCRTabProps) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [recognizedText, setRecognizedText] = useState<string>('')
  const [originalOCR, setOriginalOCR] = useState<string>('')
  const [sourceInfo, setSourceInfo] = useState<OCRResponse['source_info'] | null>(null)
  const [isProcessing, setIsProcessing] = useState<boolean>(false)
  const [error, setError] = useState<string>('')
  const [dragActive, setDragActive] = useState<boolean>(false)

  const { processImage } = useAPI()

  const handleFileSelect = useCallback((event: Event) => {
    const target = event.target as HTMLInputElement
    const file = target.files?.[0]
    if (file) {
      if (!isImageFile(file)) {
        setError('Please select an image file')
        return
      }
      if (!validateFileSize(file)) {
        setError('File size must be less than 10MB')
        return
      }
      setSelectedFile(file)
      setError('')
    }
  }, [])

  const handleDrop = useCallback((event: DragEvent) => {
    event.preventDefault()
    setDragActive(false)
    
    const file = event.dataTransfer?.files[0]
    if (file) {
      if (!isImageFile(file)) {
        setError('Please select an image file')
        return
      }
      if (!validateFileSize(file)) {
        setError('File size must be less than 10MB')
        return
      }
      setSelectedFile(file)
      setError('')
    }
  }, [])

  const handleDragOver = useCallback((event: DragEvent) => {
    event.preventDefault()
    setDragActive(true)
  }, [])

  const handleDragLeave = useCallback((event: DragEvent) => {
    event.preventDefault()
    setDragActive(false)
  }, [])

  const processImageFile = useCallback(async () => {
    if (!selectedFile) return

    setIsProcessing(true)
    setError('')

    try {
      const data = await processImage(selectedFile)
      
      if (data.success) {
        setRecognizedText(data.text)
        setOriginalOCR(data.original_ocr)
        setSourceInfo(data.source_info || null)
      } else {
        setError('Failed to process image')
      }
    } catch (err) {
      setError(getErrorMessage(err))
    } finally {
      setIsProcessing(false)
    }
  }, [selectedFile, processImage])

  const handleCopyToClipboard = useCallback(async () => {
    if (recognizedText) {
      const success = await copyToClipboard(recognizedText)
      if (!success) {
        setError('Failed to copy text to clipboard')
      }
    }
  }, [recognizedText])

  const clearAll = useCallback(() => {
    setSelectedFile(null)
    setRecognizedText('')
    setOriginalOCR('')
    setSourceInfo(null)
    setError('')
  }, [])

  return (
    <div className={`glass-effect rounded-2xl p-8 mb-6 ${className}`}>
      {/* File Upload Area */}
      <div
        className={`border-2 border-dashed rounded-xl p-8 mb-6 text-center transition-all duration-300 ${
          dragActive 
            ? 'border-blue-400 bg-blue-50/10' 
            : 'border-white/30 hover:border-white/50'
        }`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
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
              d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" 
            />
          </svg>
          <p className="text-lg mb-2">
            Drag & drop your Sanskrit image here
          </p>
          <p className="text-sm text-white/60">
            or click to select a file (max 10MB)
          </p>
        </div>
        
        <input
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          className="hidden"
          id="file-input"
        />
        <label
          htmlFor="file-input"
          className="inline-block bg-blue-500 hover:bg-blue-600 text-white px-6 py-2 rounded-lg cursor-pointer transition-colors"
        >
          Select Image
        </label>
      </div>

      {/* Selected File Info */}
      {selectedFile && (
        <div className="bg-white/10 rounded-lg p-4 mb-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-white font-medium">{selectedFile.name}</p>
              <p className="text-white/60 text-sm">
                {formatFileSize(selectedFile.size)}
              </p>
            </div>
            <button
              onClick={processImageFile}
              disabled={isProcessing}
              className="bg-green-500 hover:bg-green-600 disabled:bg-gray-400 text-white px-6 py-2 rounded-lg transition-colors flex items-center gap-2"
            >
              {isProcessing ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
                  Processing...
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  Process Image
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="bg-red-500/20 border border-red-500/50 rounded-lg p-4 mb-6">
          <p className="text-red-200">{error}</p>
        </div>
      )}

      {/* Source Information */}
      {sourceInfo && (
        <div className="bg-green-500/20 border border-green-500/50 rounded-lg p-4 mb-6">
          <h4 className="text-green-200 font-semibold mb-2">📚 Source Found in Database</h4>
          <div className="text-green-100 text-sm">
            <p><strong>Book:</strong> {sourceInfo.source_book}</p>
            {sourceInfo.source_chapter && <p><strong>Chapter:</strong> {sourceInfo.source_chapter}</p>}
            {sourceInfo.source_verse && <p><strong>Verse:</strong> {sourceInfo.source_verse}</p>}
            <p><strong>Confidence:</strong> {(sourceInfo.confidence * 100).toFixed(1)}%</p>
            <p><strong>Match Type:</strong> {sourceInfo.match_type}</p>
          </div>
        </div>
      )}

      {/* Results */}
      {recognizedText && (
        <div className="bg-white/10 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white">
              {sourceInfo ? 'Corrected Text (from Database)' : 'Recognized Text (Russian Diacritics)'}
            </h3>
            <div className="flex gap-2">
              <button
                onClick={handleCopyToClipboard}
                className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg transition-colors flex items-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                Copy
              </button>
              <button
                onClick={clearAll}
                className="bg-gray-500 hover:bg-gray-600 text-white px-4 py-2 rounded-lg transition-colors"
              >
                Clear
              </button>
            </div>
          </div>
          <div className="bg-black/20 rounded-lg p-4">
            <pre className="text-white whitespace-pre-wrap font-mono text-sm leading-relaxed">
              {recognizedText}
            </pre>
          </div>
          
          {/* Show original OCR if different */}
          {originalOCR && originalOCR !== recognizedText && (
            <div className="mt-4">
              <h4 className="text-white/80 text-sm mb-2">Original OCR Output:</h4>
              <div className="bg-black/20 rounded-lg p-4">
                <pre className="text-white/60 whitespace-pre-wrap font-mono text-sm leading-relaxed">
                  {originalOCR}
                </pre>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
} 