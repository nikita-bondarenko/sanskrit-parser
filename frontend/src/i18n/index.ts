import { SupportedLanguage, Translations } from '../types'
import { ruTranslations } from './ru'
import { enTranslations } from './en'

export const translations: Record<SupportedLanguage, Translations> = {
  ru: ruTranslations,
  en: enTranslations
}

export const getTranslation = (language: SupportedLanguage, key: string): string => {
  return translations[language]?.[key] || key
}

export const interpolate = (text: string, params: Record<string, string | number>): string => {
  return text.replace(/\{(\w+)\}/g, (match, key) => {
    return params[key]?.toString() || match
  })
}

export * from './ru'
export * from './en' 