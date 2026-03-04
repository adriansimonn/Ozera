/**
 * Settings page — account, credits, and UI preferences.
 * Tabbed interface with three separate views.
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import { NavBar } from '../components/common/NavBar'
import { useAuthStore } from '../stores/authStore'
import { useSettingsStore } from '../stores/settingsStore'
import { apiClient } from '../api/axios'
import { supabase } from '../lib/supabase'
import {
  Eye,
  EyeOff,
  Check,
  AlertCircle,
  Loader2,
  Lock,
  Plus,
} from 'lucide-react'

type SettingsTab = 'account' | 'credits' | 'ui'

export default function SettingsPage() {
  const { user, isAuthenticated, refreshUser } = useAuthStore()
  const { settings, fetchSettings, updateSettings } = useSettingsStore()

  const [activeTab, setActiveTab] = useState<SettingsTab>('account')

  // Account fields
  const [displayName, setDisplayName] = useState('')
  const [nameSaveStatus, setNameSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle')

  // Password fields
  const [currentPassword, setCurrentPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [passwordStatus, setPasswordStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle')
  const [passwordError, setPasswordError] = useState('')
  const [showCurrentPassword, setShowCurrentPassword] = useState(false)
  const [showNewPassword, setShowNewPassword] = useState(false)

  // Local state for continuous inputs
  const [lowBalanceAlert, setLowBalanceAlert] = useState<number | null>(null)

  // Debounce timer ref for settings auto-save
  const debounceRef = useRef<ReturnType<typeof setTimeout>>()

  // Initialize display name from user
  useEffect(() => {
    if (user) {
      setDisplayName(user.full_name || '')
    }
  }, [user])

  // Sync low balance alert from settings
  useEffect(() => {
    if (settings?.credits.low_balance_alert != null) {
      setLowBalanceAlert(settings.credits.low_balance_alert)
    }
  }, [settings?.credits.low_balance_alert])

  // Debounced settings update
  const debouncedUpdate = useCallback(
    (updates: Record<string, any>) => {
      if (debounceRef.current) clearTimeout(debounceRef.current)
      debounceRef.current = setTimeout(() => {
        updateSettings(updates)
      }, 600)
    },
    [updateSettings],
  )

  // Debounced display name save
  const debouncedNameSave = useCallback(
    (name: string) => {
      if (debounceRef.current) clearTimeout(debounceRef.current)
      debounceRef.current = setTimeout(async () => {
        setNameSaveStatus('saving')
        try {
          await supabase.auth.updateUser({ data: { full_name: name } })
          await apiClient.patch('/settings/profile', { display_name: name })
          await refreshUser()
          setNameSaveStatus('saved')
          setTimeout(() => setNameSaveStatus('idle'), 2000)
        } catch {
          setNameSaveStatus('error')
          setTimeout(() => setNameSaveStatus('idle'), 2000)
        }
      }, 600)
    },
    [refreshUser],
  )

  const handleDisplayNameChange = (value: string) => {
    setDisplayName(value)
    if (value.trim()) {
      debouncedNameSave(value.trim())
    }
  }

  const handlePasswordChange = async () => {
    setPasswordError('')
    if (newPassword !== confirmPassword) {
      setPasswordError('Passwords do not match')
      return
    }
    if (newPassword.length < 6) {
      setPasswordError('Password must be at least 6 characters')
      return
    }

    setPasswordStatus('saving')
    try {
      // Verify current password
      const { error: signInError } = await supabase.auth.signInWithPassword({
        email: user!.email,
        password: currentPassword,
      })
      if (signInError) {
        setPasswordError('Current password is incorrect')
        setPasswordStatus('error')
        setTimeout(() => setPasswordStatus('idle'), 2000)
        return
      }

      // Update password
      const { error: updateError } = await supabase.auth.updateUser({ password: newPassword })
      if (updateError) {
        setPasswordError(updateError.message)
        setPasswordStatus('error')
        setTimeout(() => setPasswordStatus('idle'), 2000)
        return
      }

      setPasswordStatus('saved')
      setCurrentPassword('')
      setNewPassword('')
      setConfirmPassword('')
      setTimeout(() => setPasswordStatus('idle'), 2000)
    } catch (err: any) {
      setPasswordError(err.message || 'Failed to update password')
      setPasswordStatus('error')
      setTimeout(() => setPasswordStatus('idle'), 2000)
    }
  }

  const handleAddCredits = () => {
    if ((window as any).showPurchaseCreditsModal) {
      (window as any).showPurchaseCreditsModal()
    }
  }

  // Setting change handlers — immediate for discrete inputs, debounced for continuous
  const handleCreditsSetting = (key: string, value: any, debounce = false) => {
    if (debounce) {
      debouncedUpdate({ credits: { [key]: value } })
    } else {
      updateSettings({ credits: { [key]: value } })
    }
  }

  const handleUISetting = (key: string, value: any) => {
    updateSettings({ ui: { [key]: value } })
  }

  if (!isAuthenticated || !user) {
    return (
      <div className="min-h-screen">
        <NavBar />
        <div className="max-w-2xl mx-auto px-6 pt-24">
          <div className="bg-white/[0.03] border border-white/10 p-8 text-center">
            <p className="text-gray-400">Please log in to access settings.</p>
          </div>
        </div>
      </div>
    )
  }

  const tabs: { key: SettingsTab; label: string }[] = [
    { key: 'account', label: 'Account' },
    { key: 'credits', label: 'Credit Balance' },
    { key: 'ui', label: 'UI Preferences' },
  ]

  return (
    <div className="min-h-screen">
      <NavBar />

      <div className="max-w-2xl mx-auto px-6 pt-24 pb-16">
        <h1 className="text-2xl font-bold text-white mb-6">Settings</h1>

        {/* Tab Buttons */}
        <div className="flex gap-1 mb-6 border-b border-white/10 pb-px">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={`px-4 py-2.5 text-sm font-medium transition-colors border-b-2 -mb-px ${
                activeTab === tab.key
                  ? 'border-white text-white'
                  : 'border-transparent text-gray-500 hover:text-gray-300'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Account Tab */}
        {activeTab === 'account' && (
          <div className="bg-white/[0.03] border border-white/10 p-5 space-y-4">
            {/* Display Name */}
            <div>
              <label className="block text-sm text-gray-400 mb-1">Display Name</label>
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  value={displayName}
                  onChange={(e) => handleDisplayNameChange(e.target.value)}
                  className="flex-1 bg-white/5 border border-white/10 px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-blue-500/50"
                  placeholder="Your name"
                />
                {nameSaveStatus === 'saving' && <Loader2 size={14} className="text-gray-400 animate-spin" />}
                {nameSaveStatus === 'saved' && <Check size={14} className="text-green-400" />}
                {nameSaveStatus === 'error' && <AlertCircle size={14} className="text-red-400" />}
              </div>
            </div>

            {/* Email (read-only) */}
            <div>
              <label className="block text-sm text-gray-400 mb-1">Email</label>
              <input
                type="email"
                value={user.email}
                disabled
                className="w-full bg-white/5 border border-white/10 px-3 py-2 text-sm text-gray-500 cursor-not-allowed"
              />
            </div>

            {/* Change Password */}
            <div className="pt-2 border-t border-white/5">
              <div className="flex items-center gap-2 mb-3">
                <Lock size={14} className="text-gray-400" />
                <span className="text-sm font-medium text-gray-300">Change Password</span>
              </div>
              <div className="space-y-3">
                <div className="relative">
                  <input
                    type={showCurrentPassword ? 'text' : 'password'}
                    value={currentPassword}
                    onChange={(e) => setCurrentPassword(e.target.value)}
                    className="w-full bg-white/5 border border-white/10 px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-blue-500/50 pr-10"
                    placeholder="Current password"
                  />
                  <button
                    type="button"
                    onClick={() => setShowCurrentPassword(!showCurrentPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300"
                  >
                    {showCurrentPassword ? <EyeOff size={14} /> : <Eye size={14} />}
                  </button>
                </div>
                <div className="relative">
                  <input
                    type={showNewPassword ? 'text' : 'password'}
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    className="w-full bg-white/5 border border-white/10 px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-blue-500/50 pr-10"
                    placeholder="New password"
                  />
                  <button
                    type="button"
                    onClick={() => setShowNewPassword(!showNewPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300"
                  >
                    {showNewPassword ? <EyeOff size={14} /> : <Eye size={14} />}
                  </button>
                </div>
                <input
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  className="w-full bg-white/5 border border-white/10 px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-blue-500/50"
                  placeholder="Confirm new password"
                />
                {passwordError && (
                  <p className="text-xs text-red-400">{passwordError}</p>
                )}
                <button
                  onClick={handlePasswordChange}
                  disabled={!currentPassword || !newPassword || !confirmPassword || passwordStatus === 'saving'}
                  className="flex items-center gap-2 px-4 py-2 text-sm text-gray-400 hover:text-white border border-white/10 hover:border-white/20 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  {passwordStatus === 'saving' && <Loader2 size={14} className="animate-spin" />}
                  {passwordStatus === 'saved' && <Check size={14} className="text-green-400" />}
                  Update Password
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Credit Balance Tab */}
        {activeTab === 'credits' && (
          <div className="bg-white/[0.03] border border-white/10 p-5 space-y-4">
            {/* Low Balance Alert */}
            <div>
              <label className="block text-sm text-gray-400 mb-1">Low Balance Alert Threshold ($)</label>
              <input
                type="number"
                min={0}
                step={0.5}
                value={lowBalanceAlert ?? settings?.credits.low_balance_alert ?? 5}
                onChange={(e) => {
                  const val = parseFloat(e.target.value) || 0
                  setLowBalanceAlert(val)
                  handleCreditsSetting('low_balance_alert', val, true)
                }}
                className="w-32 bg-white/5 border border-white/10 px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500/50"
              />
              <p className="text-xs text-gray-500 mt-1">Warn when balance drops below this amount</p>
            </div>

            {/* Show Balance in Navbar */}
            <div className="flex items-center justify-between">
              <div>
                <label className="text-sm text-gray-300">Show balance in navbar</label>
                <p className="text-xs text-gray-500">Display credit balance next to your profile</p>
              </div>
              <button
                onClick={() => handleCreditsSetting('show_balance_in_navbar', !(settings?.credits.show_balance_in_navbar ?? true))}
                className={`w-10 h-5 rounded-full transition-colors relative ${
                  (settings?.credits.show_balance_in_navbar ?? true) ? 'bg-blue-600' : 'bg-gray-600'
                }`}
              >
                <span
                  className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform ${
                    (settings?.credits.show_balance_in_navbar ?? true) ? 'left-5' : 'left-0.5'
                  }`}
                />
              </button>
            </div>

            {/* Add Credits */}
            <div>
              <button
                onClick={handleAddCredits}
                className="flex items-center gap-2 px-4 py-2 text-sm text-gray-400 hover:text-white border border-white/10 hover:border-white/20 transition-colors"
              >
                <Plus size={14} />
                Add Credits
              </button>
            </div>
          </div>
        )}

        {/* UI Preferences Tab */}
        {activeTab === 'ui' && (
          <div className="bg-white/[0.03] border border-white/10 p-5 space-y-4">
            {/* Theme */}
            <div>
              <label className="block text-sm text-gray-400 mb-1">Theme</label>
              <div className="flex gap-2">
                {(['dark', 'light', 'liquid_glass', 'system'] as const).map((theme) => (
                  <button
                    key={theme}
                    onClick={() => handleUISetting('theme', theme)}
                    className={`px-4 py-2 text-sm transition-colors border ${
                      (settings?.ui.theme ?? 'dark') === theme
                        ? 'bg-blue-600/20 border-blue-500/50 text-blue-300'
                        : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
                    }`}
                  >
                    {theme === 'liquid_glass' ? 'Liquid Glass' : theme.charAt(0).toUpperCase() + theme.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            {/* Default View Mode */}
            <div>
              <label className="block text-sm text-gray-400 mb-1">Default View Mode</label>
              <div className="flex gap-2">
                {(['split', 'single'] as const).map((mode) => (
                  <button
                    key={mode}
                    onClick={() => handleUISetting('default_view_mode', mode)}
                    className={`px-4 py-2 text-sm capitalize transition-colors border ${
                      (settings?.ui.default_view_mode ?? 'split') === mode
                        ? 'bg-blue-600/20 border-blue-500/50 text-blue-300'
                        : 'bg-white/5 border-white/10 text-gray-400 hover:bg-white/10'
                    }`}
                  >
                    {mode}
                  </button>
                ))}
              </div>
            </div>

            {/* Default Landing Page */}
            <div>
              <label className="block text-sm text-gray-400 mb-1">Default Landing Page</label>
              <select
                value={settings?.ui.default_page ?? '/'}
                onChange={(e) => handleUISetting('default_page', e.target.value)}
                className="bg-white/5 border border-white/10 px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500/50"
              >
                <option value="/">Generate</option>
                <option value="/training">Custom Models</option>
                <option value="/patching">Activation Patching</option>
                <option value="/analysis">Attention Analysis</option>
                <option value="/sae">SAE Analysis</option>
              </select>
            </div>
          </div>
        )}

      </div>
    </div>
  )
}
