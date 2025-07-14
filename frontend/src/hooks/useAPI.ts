/**
 * Custom hooks for API calls
 */

import { useCallback } from 'preact/hooks'
import { OCRResponse, TextProcessResponse, ConvertTextResponse, BookUploadResponse, LoginResponse, DatabaseStats, SearchResponse, SupportedFormatsResponse, OutputFormat, InputFormat } from '../types'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export const useAPI = () => {
  const processImage = useCallback(async (file: File, outputFormat: OutputFormat, inputFormat: InputFormat): Promise<OCRResponse> => {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('output_format', outputFormat)
    formData.append('input_format', inputFormat)
    
    // Debug logging
    console.log('API call with formats:', { outputFormat, inputFormat })

    const response = await fetch(`${API_URL}/ocr`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    return response.json()
  }, [])

  const processText = useCallback(async (text: string, outputFormat: OutputFormat): Promise<TextProcessResponse> => {
    const response = await fetch(`${API_URL}/process-text`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text,
        output_format: outputFormat,
      }),
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    return response.json()
  }, [])

  const convertText = useCallback(async (text: string, targetFormat: OutputFormat, sourceFormat?: string): Promise<ConvertTextResponse> => {
    const response = await fetch(`${API_URL}/convert-text`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text,
        target_format: targetFormat,
        source_format: sourceFormat,
      }),
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    return response.json()
  }, [])

  const uploadBook = useCallback(async (file: File, token: string): Promise<BookUploadResponse> => {
    const formData = new FormData()
    formData.append('file', file)

    const response = await fetch(`${API_URL}/upload-book`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
      },
      body: formData,
    })

    if (!response.ok) {
      if (response.status === 401) {
        throw new Error('Authentication required. Please log in as admin.')
      }
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    return response.json()
  }, [])

  const adminLogin = useCallback(async (password: string): Promise<LoginResponse> => {
    const response = await fetch(`${API_URL}/admin/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ password }),
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    return response.json()
  }, [])

  const getDatabaseStats = useCallback(async (): Promise<DatabaseStats> => {
    const response = await fetch(`${API_URL}/database-stats`)
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const data = await response.json()
    return data.stats
  }, [])

  const searchBooks = useCallback(async (bookName: string): Promise<SearchResponse> => {
    const response = await fetch(`${API_URL}/search-books?book_name=${encodeURIComponent(bookName)}`)
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    return response.json()
  }, [])

  const getSupportedFormats = useCallback(async (): Promise<SupportedFormatsResponse> => {
    const response = await fetch(`${API_URL}/supported-formats`)
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    return response.json()
  }, [])

  return {
    processImage,
    processText,
    convertText,
    uploadBook,
    adminLogin,
    getDatabaseStats,
    searchBooks,
    getSupportedFormats,
  }
} 