import { useState, useCallback } from 'preact/hooks'
import { SupportedLanguage, LanguageContextType } from '../types'
import { getTranslation, interpolate } from '../i18n'

// Simple global state for language management without Context API
let globalLanguage: SupportedLanguage = 'ru'
let globalSetLanguage: ((language: SupportedLanguage) => void) | null = null
let globalT: ((key: string, params?: Record<string, string | number>) => string) | null = null

// Initialize language from localStorage
try {
  const savedLanguage = localStorage.getItem('sanskrit-ocr-language') as SupportedLanguage
  if (savedLanguage && (savedLanguage === 'ru' || savedLanguage === 'en')) {
    globalLanguage = savedLanguage
  }
} catch (error) {
  console.warn('Failed to load language preference:', error)
}

export const useLanguage = (): LanguageContextType => {
  const [currentLanguage, setCurrentLanguage] = useState<SupportedLanguage>(globalLanguage)

  const setLanguage = useCallback((language: SupportedLanguage) => {
    setCurrentLanguage(language)
    globalLanguage = language
    // Persist language preference in localStorage
    try {
      localStorage.setItem('sanskrit-ocr-language', language)
    } catch (error) {
      console.warn('Failed to save language preference:', error)
    }
    // Notify other components
    if (globalSetLanguage) {
      globalSetLanguage(language)
    }
  }, [])

  const t = useCallback((key: string, params?: Record<string, string | number>) => {
    const translation = getTranslation(currentLanguage, key)
    return params ? interpolate(translation, params) : translation
  }, [currentLanguage])

  // Update global references
  globalSetLanguage = setLanguage
  globalT = t

  return {
    currentLanguage,
    setLanguage,
    t
  }
} 