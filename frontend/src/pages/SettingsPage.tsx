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
import { Dropdown } from '../components/common/Dropdown'

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
      <div className="min-h-screen" style={{ minHeight: '125vh' }}>
        <NavBar />
        <div className="max-w-3xl mx-auto px-8 pt-28">
          <div className="bg-white/[0.03] border border-white/10 p-10 text-center">
            <p className="text-base text-gray-300">Please log in to access settings.</p>
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
    <div className="min-h-screen" style={{ minHeight: '125vh' }}>
      <NavBar />

      <div className="max-w-3xl mx-auto px-8 pt-28 pb-20">
        <h1 className="text-3xl font-bold text-white mb-8">Settings</h1>

        {/* Tab Buttons */}
        <div className="flex gap-1 mb-8 border-b border-white/10 pb-px">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={`px-5 py-3 text-base font-medium transition-colors border-b-2 -mb-px ${
                activeTab === tab.key
                  ? 'border-white text-white'
                  : 'border-transparent text-gray-400 hover:text-gray-200'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Account Tab */}
        {activeTab === 'account' && (
          <div className="bg-white/[0.03] border border-white/10 p-7 space-y-6">
            {/* Display Name */}
            <div>
              <label className="block text-base text-gray-300 mb-2">Display Name</label>
              <div className="flex items-center gap-3">
                <input
                  type="text"
                  value={displayName}
                  onChange={(e) => handleDisplayNameChange(e.target.value)}
                  className="flex-1 bg-white/5 border border-white/10 px-4 py-3 text-base text-white placeholder-gray-400 focus:outline-none focus:border-white/30"
                  placeholder="Your name"
                />
                {nameSaveStatus === 'saving' && <Loader2 size={18} className="text-gray-300 animate-spin" />}
                {nameSaveStatus === 'saved' && <Check size={18} className="text-green-400" />}
                {nameSaveStatus === 'error' && <AlertCircle size={18} className="text-red-400" />}
              </div>
            </div>

            {/* Email (read-only) */}
            <div>
              <label className="block text-base text-gray-300 mb-2">Email</label>
              <input
                type="email"
                value={user.email}
                disabled
                className="w-full bg-white/5 border border-white/10 px-4 py-3 text-base text-gray-500 cursor-not-allowed"
              />
            </div>

            {/* Change Password */}
            <div className="pt-3 border-t border-white/5">
              <div className="flex items-center gap-2 mb-4">
                <Lock size={18} className="text-gray-300" />
                <span className="text-base font-medium text-gray-200">Change Password</span>
              </div>
              <div className="space-y-4">
                <div className="relative">
                  <input
                    type={showCurrentPassword ? 'text' : 'password'}
                    value={currentPassword}
                    onChange={(e) => setCurrentPassword(e.target.value)}
                    className="w-full bg-white/5 border border-white/10 px-4 py-3 text-base text-white placeholder-gray-400 focus:outline-none focus:border-white/30 pr-12"
                    placeholder="Current password"
                  />
                  <button
                    type="button"
                    onClick={() => setShowCurrentPassword(!showCurrentPassword)}
                    className="absolute right-4 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-200"
                  >
                    {showCurrentPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                  </button>
                </div>
                <div className="relative">
                  <input
                    type={showNewPassword ? 'text' : 'password'}
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    className="w-full bg-white/5 border border-white/10 px-4 py-3 text-base text-white placeholder-gray-400 focus:outline-none focus:border-white/30 pr-12"
                    placeholder="New password"
                  />
                  <button
                    type="button"
                    onClick={() => setShowNewPassword(!showNewPassword)}
                    className="absolute right-4 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-200"
                  >
                    {showNewPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                  </button>
                </div>
                <input
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  className="w-full bg-white/5 border border-white/10 px-4 py-3 text-base text-white placeholder-gray-400 focus:outline-none focus:border-white/30"
                  placeholder="Confirm new password"
                />
                {passwordError && (
                  <p className="text-sm text-red-400">{passwordError}</p>
                )}
                <button
                  onClick={handlePasswordChange}
                  disabled={!currentPassword || !newPassword || !confirmPassword || passwordStatus === 'saving'}
                  className="flex items-center gap-2 px-5 py-3 text-base text-gray-300 hover:text-white border border-white/10 hover:border-white/20 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  {passwordStatus === 'saving' && <Loader2 size={18} className="animate-spin" />}
                  {passwordStatus === 'saved' && <Check size={18} className="text-green-400" />}
                  Update Password
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Credit Balance Tab */}
        {activeTab === 'credits' && (
          <div className="bg-white/[0.03] border border-white/10 p-7 space-y-6">
            {/* Low Balance Alert */}
            <div>
              <label className="block text-base text-gray-300 mb-2">Low Balance Alert Threshold ($)</label>
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
                className="w-40 bg-white/5 border border-white/10 px-4 py-3 text-base text-white focus:outline-none focus:border-white/30"
              />
              <p className="text-sm text-gray-400 mt-2">Warn when balance drops below this amount</p>
            </div>

            {/* Show Balance in Navbar */}
            <div className="flex items-center justify-between">
              <div>
                <label className="text-base text-gray-200">Show balance in navbar</label>
                <p className="text-sm text-gray-400">Display credit balance next to your profile</p>
              </div>
              <button
                onClick={() => handleCreditsSetting('show_balance_in_navbar', !(settings?.credits.show_balance_in_navbar ?? true))}
                className={`w-14 h-8 rounded-full transition-colors relative ${
                  (settings?.credits.show_balance_in_navbar ?? true) ? 'bg-green-500' : 'bg-white/10'
                }`}
              >
                <span
                  className={`absolute top-1 w-6 h-6 rounded-full bg-white transition-transform ${
                    (settings?.credits.show_balance_in_navbar ?? true) ? 'left-[30px]' : 'left-1'
                  }`}
                />
              </button>
            </div>

            {/* Add Credits */}
            <div>
              <button
                onClick={handleAddCredits}
                className="flex items-center gap-2 px-5 py-3 text-base text-gray-300 hover:text-white border border-white/10 hover:border-white/20 transition-colors"
              >
                <Plus size={18} />
                Add Credits
              </button>
            </div>
          </div>
        )}

        {/* UI Preferences Tab */}
        {activeTab === 'ui' && (
          <div className="bg-white/[0.03] border border-white/10 p-7 space-y-6">
            {/* Background */}
            <div>
              <label className="block text-base text-gray-300 mb-2">Background</label>
              <Dropdown
                value={settings?.ui.background ?? 'glow'}
                onChange={(v) => handleUISetting('background', v)}
                options={[
                  { value: 'dark', label: 'Dark' },
                  { value: 'light', label: 'Light' },
                  { value: 'glow', label: 'Glow' },
                  { value: 'system', label: 'System' },
                ]}
              />
            </div>

            {/* Interface */}
            <div>
              <label className="block text-base text-gray-300 mb-2">Interface</label>
              <Dropdown
                value={settings?.ui.interface ?? 'glass'}
                onChange={(v) => handleUISetting('interface', v)}
                options={[
                  { value: 'default', label: 'Default' },
                  { value: 'glass', label: 'Glass' },
                ]}
              />
            </div>

            {/* Default View Mode */}
            <div>
              <label className="block text-base text-gray-300 mb-2">Default View Mode</label>
              <Dropdown
                value={settings?.ui.default_view_mode ?? 'split'}
                onChange={(v) => handleUISetting('default_view_mode', v)}
                options={[
                  { value: 'split', label: 'Split' },
                  { value: 'single', label: 'Single' },
                ]}
              />
            </div>

            {/* Default Landing Page */}
            <div>
              <label className="block text-base text-gray-300 mb-2">Default Landing Page</label>
              <Dropdown
                value={settings?.ui.default_page ?? '/'}
                onChange={(v) => handleUISetting('default_page', v)}
                options={[
                  { value: '/', label: 'Generate' },
                  { value: '/training', label: 'Custom Models' },
                  { value: '/patching', label: 'Activation Patching' },
                  { value: '/analysis', label: 'Attention Analysis' },
                  { value: '/sae', label: 'SAE Analysis' },
                ]}
              />
            </div>
          </div>
        )}

      </div>
    </div>
  )
}
