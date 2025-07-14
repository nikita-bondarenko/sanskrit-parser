/**
 * OCR Tab Component - Enhanced with Input Type and Output Format Selection
 */

import { useState, useCallback, useEffect } from 'preact/hooks'
import { OCRResponse, TextProcessResponse, InputType, OutputFormat, InputFormat } from '../types'
import { useAPI } from '../hooks/useAPI'
import { useLanguage } from '../contexts/LanguageContext'
import { copyToClipboard, formatFileSize, isImageFile, validateFileSize, getErrorMessage } from '../utils'

interface OCRTabProps {
  className?: string
}

export const OCRTab = ({ className = '' }: OCRTabProps) => {
  // Input type and format selection
  const [inputType, setInputType] = useState<InputType>('image')
  const [outputFormat, setOutputFormat] = useState<OutputFormat>('russian')
  const [inputFormat, setInputFormat] = useState<InputFormat>('english_diacritics')
  
  // Image input state
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [dragActive, setDragActive] = useState<boolean>(false)
  
  // Text input state
  const [inputText, setInputText] = useState<string>('')
  
  // Common result state
  const [recognizedText, setRecognizedText] = useState<string>('')
  const [originalOCR, setOriginalOCR] = useState<string>('')
  const [sourceInfo, setSourceInfo] = useState<OCRResponse['source_info'] | null>(null)
  const [isProcessing, setIsProcessing] = useState<boolean>(false)
  const [error, setError] = useState<string>('')

  const { processImage, processText } = useAPI()
  const { t } = useLanguage()

  // File handling for image input
  const handleFileSelect = useCallback((event: Event) => {
    const target = event.target as HTMLInputElement
    const file = target.files?.[0]
    if (file) {
      if (!isImageFile(file)) {
        setError(t('error.image.format'))
        return
      }
      if (!validateFileSize(file)) {
        setError(t('error.file.size'))
        return
      }
      setSelectedFile(file)
      setError('')
      // Clear previous results when new file is selected
      setRecognizedText('')
      setOriginalOCR('')
      setSourceInfo(null)
    }
  }, [t])

  const handleDrop = useCallback((event: DragEvent) => {
    event.preventDefault()
    setDragActive(false)
    
    const file = event.dataTransfer?.files[0]
    if (file) {
      if (!isImageFile(file)) {
        setError(t('error.image.format'))
        return
      }
      if (!validateFileSize(file)) {
        setError(t('error.file.size'))
        return
      }
      setSelectedFile(file)
      setError('')
      // Clear previous results when new file is selected
      setRecognizedText('')
      setOriginalOCR('')
      setSourceInfo(null)
    }
  }, [t])

  const handleDragOver = useCallback((event: DragEvent) => {
    event.preventDefault()
    setDragActive(true)
  }, [])

  const handleDragLeave = useCallback((event: DragEvent) => {
    event.preventDefault()
    setDragActive(false)
  }, [])

  // Process image input
  const processImageFile = useCallback(async () => {
    if (!selectedFile) return

    setIsProcessing(true)
    setError('')
    
    // Debug logging
    console.log('Processing image with formats:', { inputFormat, outputFormat })

    try {
      const data = await processImage(selectedFile, outputFormat, inputFormat)
      
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
  }, [selectedFile, outputFormat, inputFormat, processImage])

  // Process text input
  const processTextInput = useCallback(async () => {
    if (!inputText.trim()) return

    setIsProcessing(true)
    setError('')

    try {
      const data = await processText(inputText, outputFormat)
      
      if (data.success) {
        setRecognizedText(data.text)
        setOriginalOCR(data.original_text)
        setSourceInfo(data.source_info || null)
      } else {
        setError('Failed to process text')
      }
    } catch (err) {
      setError(getErrorMessage(err))
    } finally {
      setIsProcessing(false)
    }
  }, [inputText, outputFormat, processText])

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
    setInputText('')
    setRecognizedText('')
    setOriginalOCR('')
    setSourceInfo(null)
    setError('')
  }, [])

  const handleInputTypeChange = useCallback((newInputType: InputType) => {
    setInputType(newInputType)
    clearAll()
  }, [clearAll])

  const formatDisplayName = useCallback((format: OutputFormat) => {
    return format === 'russian' ? t('output.format.russian') : t('output.format.iast')
  }, [t])

  // Clear results when format settings change
  useEffect(() => {
    if (recognizedText) {
      setRecognizedText('')
      setOriginalOCR('')
      setSourceInfo(null)
    }
  }, [outputFormat, inputFormat])

  return (
    <div className={`glass-effect rounded-2xl p-8 mb-6 ${className}`}>
            {/* Input Type and Format Selection */}
      <div className="mb-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Input Type Selection */}
          <div>
            <label className="block text-sm font-medium text-white mb-2">{t('input.type.title')}</label>
            <select
              value={inputType}
              onChange={(e) => handleInputTypeChange((e.target as HTMLSelectElement).value as InputType)}
              className="w-full bg-white/10 border border-white/30 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-400"
            >
              <option value="image" className="bg-gray-800 text-white">{t('input.type.image')}</option>
              <option value="text" className="bg-gray-800 text-white">{t('input.type.text')}</option>
            </select>
          </div>

          {/* Input Format Selection (only for images) */}
          {inputType === 'image' && (
            <div>
              <label className="block text-sm font-medium text-white mb-2">{t('input.format.title')}</label>
              <select
                value={inputFormat}
                onChange={(e) => setInputFormat((e.target as HTMLSelectElement).value as InputFormat)}
                className="w-full bg-white/10 border border-white/30 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-purple-400 focus:ring-1 focus:ring-purple-400"
              >
                <option value="english_diacritics" className="bg-gray-800 text-white">{t('input.format.english')}</option>
                <option value="russian_diacritics" className="bg-gray-800 text-white">{t('input.format.russian')}</option>
              </select>
              <p className="text-xs text-white/60 mt-1">
                {t('input.format.description')}
              </p>
            </div>
          )}

          {/* Output Format Selection */}
          <div>
            <label className="block text-sm font-medium text-white mb-2">{t('output.format.title')}</label>
            <select
              value={outputFormat}
              onChange={(e) => setOutputFormat((e.target as HTMLSelectElement).value as OutputFormat)}
              className="w-full bg-white/10 border border-white/30 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-green-400 focus:ring-1 focus:ring-green-400"
            >
              <option value="russian" className="bg-gray-800 text-white">{t('output.format.russian')}</option>
              <option value="iast" className="bg-gray-800 text-white">{t('output.format.iast')}</option>
            </select>
          </div>
        </div>
      </div>

      {/* Input Area */}
      {inputType === 'image' ? (
        <>
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
                {t('upload.drag.title')}
              </p>
              <p className="text-sm text-white/60">
                {t('upload.drag.subtitle')}
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
              {t('upload.select')}
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
                      {t('upload.processing')}
                    </>
                  ) : (
                    <>
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                      {t('upload.process')}
                    </>
                  )}
                </button>
              </div>
            </div>
          )}
        </>
      ) : (
        <>
          {/* Text Input Area */}
          <div className="mb-6">
            <label className="block text-white text-sm font-medium mb-2">
              Enter Sanskrit Text ({formatDisplayName(outputFormat === 'russian' ? 'iast' : 'russian')} or any format)
            </label>
            <textarea
              value={inputText}
              onChange={(e) => setInputText((e.target as HTMLTextAreaElement).value)}
              placeholder="Enter Sanskrit text in any format (IAST, Russian diacritics, etc.)..."
              className="w-full h-32 p-4 bg-black/20 border border-white/30 rounded-lg text-white placeholder-white/50 focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-400 resize-none"
            />
            <div className="flex justify-between items-center mt-2">
              <p className="text-white/60 text-sm">
                Characters: {inputText.length}
              </p>
              <button
                onClick={processTextInput}
                disabled={isProcessing || !inputText.trim()}
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
                    Process Text
                  </>
                )}
              </button>
            </div>
          </div>
        </>
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
              {sourceInfo ? t('results.corrected') : t('results.processed', { format: formatDisplayName(outputFormat) })}
            </h3>
            <div className="flex gap-2">
              <button
                onClick={handleCopyToClipboard}
                className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg transition-colors flex items-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                                  </svg>
                  {t('results.copy')}
                </button>
                <button
                  onClick={clearAll}
                  className="bg-gray-500 hover:bg-gray-600 text-white px-4 py-2 rounded-lg transition-colors"
                >
                  {t('results.clear')}
                </button>
            </div>
          </div>
          <div className="bg-black/20 rounded-lg p-4">
            <pre className="text-white whitespace-pre-wrap font-mono text-sm leading-relaxed">
              {recognizedText}
            </pre>
          </div>
          
          {/* Show original input if different from result */}
          {originalOCR && originalOCR !== recognizedText && (
            <div className="mt-4">
              <h4 className="text-white/80 text-sm mb-2">
                {inputType === 'image' ? 'Original OCR Output:' : 'Original Input:'}
              </h4>
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