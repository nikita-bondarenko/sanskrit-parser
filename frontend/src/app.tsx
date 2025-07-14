/**
 * Main Application Component - Refactored
 */

import { useState, useCallback } from 'preact/hooks'
import { TabType, DatabaseStats as DatabaseStatsType } from './types'
import { useAPI } from './hooks/useAPI'
import { useLanguage } from './contexts/LanguageContext'
import { OCRTab } from './components/OCRTab'
import { AdminPanel } from './components/AdminPanel'
import { DatabaseStats } from './components/DatabaseStats'
import { BookUpload } from './components/BookUpload'

export function App() {
  const [activeTab, setActiveTab] = useState<TabType>('ocr')
  const [isAdminAuthenticated, setIsAdminAuthenticated] = useState<boolean>(false)
  const [adminToken, setAdminToken] = useState<string>('')
  const [dbStats, setDbStats] = useState<DatabaseStatsType | null>(null)

  const { getDatabaseStats } = useAPI()
  const { currentLanguage, setLanguage, t } = useLanguage()

  const loadDatabaseStats = useCallback(async () => {
    try {
      const stats = await getDatabaseStats()
      setDbStats(stats)
    } catch (err) {
      console.error('Failed to load database stats:', err)
    }
  }, [getDatabaseStats])

  const handleAuthChange = useCallback((authenticated: boolean, token: string) => {
    setIsAdminAuthenticated(authenticated)
    setAdminToken(token)
  }, [])

  const handleStatsUpdate = useCallback((stats: DatabaseStatsType) => {
    setDbStats(stats)
  }, [])

  // Load database stats on component mount and tab change
  useState(() => {
    loadDatabaseStats()
  })

  return (
    <div className="min-h-screen gradient-bg">
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="text-center mb-8">
            <div className="flex justify-between items-start mb-4">
              <div></div> {/* Spacer */}
              <div className="flex-1">
                <h1 className="text-4xl font-bold text-white mb-2">
                  {t('app.title')}
                </h1>
                <p className="text-xl text-white/80">
                  {t('app.subtitle')}
                </p>
                <p className="text-sm text-white/60 mt-2">
                  {t('app.description')}
                </p>
              </div>
              {/* Language Switcher */}
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setLanguage('ru')}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                    currentLanguage === 'ru'
                      ? 'bg-white/20 text-white'
                      : 'text-white/70 hover:text-white hover:bg-white/10'
                  }`}
                >
                  RU
                </button>
                <button
                  onClick={() => setLanguage('en')}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                    currentLanguage === 'en'
                      ? 'bg-white/20 text-white'
                      : 'text-white/70 hover:text-white hover:bg-white/10'
                  }`}
                >
                  EN
                </button>
              </div>
            </div>
          </div>

          {/* Tab Navigation */}
          <div className="flex justify-center mb-8">
            <div className="glass-effect rounded-lg p-1 flex">
              <button
                onClick={() => setActiveTab('ocr')}
                className={`px-6 py-2 rounded-md transition-all ${
                  activeTab === 'ocr'
                    ? 'bg-blue-500 text-white'
                    : 'text-white/70 hover:text-white'
                }`}
              >
                {t('nav.ocr')}
              </button>
              <button
                onClick={() => {
                  setActiveTab('database')
                  loadDatabaseStats()
                }}
                className={`px-6 py-2 rounded-md transition-all ${
                  activeTab === 'database'
                    ? 'bg-blue-500 text-white'
                    : 'text-white/70 hover:text-white'
                }`}
              >
                {t('nav.database')}
                {isAdminAuthenticated && (
                  <span className="ml-2 text-xs bg-green-500 text-white px-2 py-1 rounded-full">
                    {t('nav.admin')}
                  </span>
                )}
              </button>
            </div>
          </div>

          {/* OCR Tab */}
          {activeTab === 'ocr' && <OCRTab />}

          {/* Database Tab */}
          {activeTab === 'database' && (
            <div className="glass-effect rounded-2xl p-8 mb-6">
              {/* Admin Panel */}
              <AdminPanel
                isAuthenticated={isAdminAuthenticated}
                onAuthChange={handleAuthChange}
                dbStats={dbStats}
                onStatsUpdate={handleStatsUpdate}
              />

              {/* Database Stats */}
              <DatabaseStats stats={dbStats} />

              {/* Book Upload - Only for authenticated admins */}
              {isAdminAuthenticated && (
                <BookUpload
                  adminToken={adminToken}
                  onStatsUpdate={handleStatsUpdate}
                />
              )}
            </div>
          )}

          {/* Footer */}
          <div className="text-center text-white/60 text-sm space-y-3">
            <p>
              {t('footer.description')}
            </p>
            <div className="border-t border-white/20 pt-3">
              <p className="mb-2">
                {t('footer.contact')}
              </p>
              <div className="flex justify-center items-center gap-4 flex-wrap">
                <a 
                  href="https://t.me/NikitaBondarenkoDev" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-blue-300 hover:text-blue-200 transition-colors flex items-center gap-1"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                  </svg>
                  @NikitaBondarenkoDev
                </a>
                <span className="text-white/40">•</span>
                <a 
                  href="mailto:brajbas3@gmail.com" 
                  className="text-blue-300 hover:text-blue-200 transition-colors flex items-center gap-1"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 4.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                  </svg>
                  brajbas3@gmail.com
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
} 