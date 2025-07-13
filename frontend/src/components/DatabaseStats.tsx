/**
 * Database Statistics Component
 */

import { DatabaseStats as DatabaseStatsType } from '../types'

interface DatabaseStatsProps {
  stats: DatabaseStatsType | null
  className?: string
}

export const DatabaseStats = ({ stats, className = '' }: DatabaseStatsProps) => {
  if (!stats) return null

  return (
    <div className={`grid grid-cols-2 md:grid-cols-4 gap-4 mb-8 ${className}`}>
      <div className="bg-white/10 rounded-lg p-4 text-center">
        <div className="text-2xl font-bold text-white">{stats.texts}</div>
        <div className="text-white/60 text-sm">Texts</div>
      </div>
      <div className="bg-white/10 rounded-lg p-4 text-center">
        <div className="text-2xl font-bold text-white">{stats.books}</div>
        <div className="text-white/60 text-sm">Books</div>
      </div>
      <div className="bg-white/10 rounded-lg p-4 text-center">
        <div className="text-2xl font-bold text-white">{stats.unique_words}</div>
        <div className="text-white/60 text-sm">Unique Words</div>
      </div>
      <div className="bg-white/10 rounded-lg p-4 text-center">
        <div className="text-2xl font-bold text-white">{stats.total_words}</div>
        <div className="text-white/60 text-sm">Total Words</div>
      </div>
    </div>
  )
} 