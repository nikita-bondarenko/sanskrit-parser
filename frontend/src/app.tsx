/**
 * Main Application Component - Refactored
 */

import { useState, useCallback } from 'preact/hooks'
import { TabType, DatabaseStats as DatabaseStatsType } from './types'
import { useAPI } from './hooks/useAPI'
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
            <h1 className="text-4xl font-bold text-white mb-2">
              Sanskrit OCR
            </h1>
            <p className="text-xl text-white/80">
              Russian Diacritic Helper with Database
            </p>
            <p className="text-sm text-white/60 mt-2">
              Upload Sanskrit text images or manage your text database
            </p>
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
                OCR Recognition
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
                Database Management
                {isAdminAuthenticated && (
                  <span className="ml-2 text-xs bg-green-500 text-white px-2 py-1 rounded-full">
                    Admin
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
          <div className="text-center text-white/60 text-sm">
            <p>
              Supports IAST and Gaura PT formats • Powered by Neural Networks • Database-Enhanced OCR
            </p>
          </div>
        </div>
      </div>
    </div>
  )
} 