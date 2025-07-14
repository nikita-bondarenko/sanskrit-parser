/**
 * Database Statistics Component
 */

import { DatabaseStats as DatabaseStatsType } from '../types'
import { useLanguage } from '../contexts/LanguageContext'

interface DatabaseStatsProps {
  stats: DatabaseStatsType | null
  className?: string
}

export const DatabaseStats = ({ stats, className = '' }: DatabaseStatsProps) => {
  const { t } = useLanguage()
  if (!stats) return null

  return (
    <div className={`grid grid-cols-2 md:grid-cols-4 gap-4 mb-8 ${className}`}>
      <div className="bg-white/10 rounded-lg p-4 text-center">
        <div className="text-2xl font-bold text-white">{stats.texts}</div>
        <div className="text-white/60 text-sm">{t('stats.texts')}</div>
      </div>
      <div className="bg-white/10 rounded-lg p-4 text-center">
        <div className="text-2xl font-bold text-white">{stats.books}</div>
        <div className="text-white/60 text-sm">{t('stats.books')}</div>
      </div>
      <div className="bg-white/10 rounded-lg p-4 text-center">
        <div className="text-2xl font-bold text-white">{stats.unique_words}</div>
        <div className="text-white/60 text-sm">{t('stats.unique_words')}</div>
      </div>
      <div className="bg-white/10 rounded-lg p-4 text-center">
        <div className="text-2xl font-bold text-white">{stats.total_words}</div>
        <div className="text-white/60 text-sm">{t('stats.total_words')}</div>
      </div>
    </div>
  )
} 