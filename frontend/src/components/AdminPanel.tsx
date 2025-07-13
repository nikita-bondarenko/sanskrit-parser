/**
 * Admin Panel Component
 */

import { useState, useCallback } from 'preact/hooks'
import { useAPI } from '../hooks/useAPI'
import { DatabaseStats } from '../types'
import { getErrorMessage } from '../utils'

interface AdminPanelProps {
  isAuthenticated: boolean
  onAuthChange: (authenticated: boolean, token: string) => void
  dbStats: DatabaseStats | null
  onStatsUpdate: (stats: DatabaseStats) => void
}

export const AdminPanel = ({ 
  isAuthenticated, 
  onAuthChange, 
  dbStats, 
  onStatsUpdate 
}: AdminPanelProps) => {
  const [adminPassword, setAdminPassword] = useState<string>('')
  const [isLoggingIn, setIsLoggingIn] = useState<boolean>(false)
  const [loginError, setLoginError] = useState<string>('')
  const [showAdminLogin, setShowAdminLogin] = useState<boolean>(false)

  const { adminLogin } = useAPI()

  const handleAdminLogin = useCallback(async () => {
    if (!adminPassword) return

    setIsLoggingIn(true)
    setLoginError('')

    try {
      const data = await adminLogin(adminPassword)
      
      if (data.success && data.token) {
        onAuthChange(true, data.token)
        setShowAdminLogin(false)
        setAdminPassword('')
        setLoginError('')
      } else {
        setLoginError(data.message || 'Login failed')
      }
    } catch (err) {
      setLoginError(getErrorMessage(err))
    } finally {
      setIsLoggingIn(false)
    }
  }, [adminPassword, adminLogin, onAuthChange])

  const handleAdminLogout = useCallback(() => {
    onAuthChange(false, '')
    setAdminPassword('')
    setLoginError('')
    setShowAdminLogin(false)
  }, [onAuthChange])

  const handleKeyPress = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleAdminLogin()
    }
  }, [handleAdminLogin])

  if (isAuthenticated) {
    return (
      <div className="bg-green-500/20 border border-green-500/50 rounded-lg p-4 mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-green-200 font-semibold">✅ Admin Access Granted</h3>
            <p className="text-green-100 text-sm">You can now upload books to the database.</p>
          </div>
          <button
            onClick={handleAdminLogout}
            className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded-lg transition-colors"
          >
            Logout
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-yellow-500/20 border border-yellow-500/50 rounded-lg p-6 mb-8">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-yellow-200 font-semibold mb-2">🔐 Admin Access Required</h3>
          <p className="text-yellow-100 text-sm">
            Book upload is restricted to administrators only. Please log in to access database management features.
          </p>
        </div>
        <button
          onClick={() => setShowAdminLogin(!showAdminLogin)}
          className="bg-yellow-500 hover:bg-yellow-600 text-white px-4 py-2 rounded-lg transition-colors"
        >
          {showAdminLogin ? 'Cancel' : 'Admin Login'}
        </button>
      </div>
      
      {showAdminLogin && (
        <div className="mt-4 p-4 bg-black/20 rounded-lg">
          <div className="flex items-center gap-4">
            <input
              type="password"
              placeholder="Enter admin password"
              value={adminPassword}
              onChange={(e) => setAdminPassword((e.target as HTMLInputElement).value)}
              onKeyPress={handleKeyPress}
              className="flex-1 bg-white/10 border border-white/30 rounded-lg px-4 py-2 text-white placeholder-white/60 focus:outline-none focus:border-white/50"
            />
            <button
              onClick={handleAdminLogin}
              disabled={isLoggingIn || !adminPassword}
              className="bg-green-500 hover:bg-green-600 disabled:bg-gray-400 text-white px-6 py-2 rounded-lg transition-colors flex items-center gap-2"
            >
              {isLoggingIn ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
                  Logging in...
                </>
              ) : (
                'Login'
              )}
            </button>
          </div>
          {loginError && (
            <p className="text-red-200 text-sm mt-2">{loginError}</p>
          )}
        </div>
      )}
    </div>
  )
} 