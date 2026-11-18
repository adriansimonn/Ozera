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
import { APP_PAGES, landingPath } from '../lib/appPages'
import {
  Eye,
  EyeOff,
  Check,
  AlertCircle } from 'lucide-react'
import { usePurchaseCredits } from '../contexts/PurchaseCreditsContext'
import { Dropdown } from '../components/common/Dropdown'
import { Spinner } from '../components/common/Spinner'

type SettingsTab = 'account' | 'credits' | 'ui'

export default function SettingsPage() {
  const { user, isAuthenticated, refreshUser } = useAuthStore()
  const { settings, updateSettings } = useSettingsStore()
  const { showPurchaseCredits } = usePurchaseCredits()

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
    showPurchaseCredits()
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
      <div className="settings-page">
        <NavBar />
        <div className="st-wrap">
          <h1 className="st-title">Settings.</h1>
          <p className="oz-hint">Please log in to access settings.</p>
        </div>
        <SettingsStyles />
      </div>
    )
  }

  const tabs: { key: SettingsTab; label: string }[] = [
    { key: 'account', label: 'Account' },
    { key: 'credits', label: 'Credit Balance' },
    { key: 'ui', label: 'UI Preferences' },
  ]

  const showBalance = settings?.credits.show_balance_in_navbar ?? true

  return (
    <div className="settings-page">
      <NavBar />

      <div className="st-wrap">
        <h1 className="st-title">Settings.</h1>

        {/* Tab Buttons */}
        <div className="oz-tabs" role="tablist">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              role="tab"
              aria-selected={activeTab === tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={`oz-tab${activeTab === tab.key ? ' active' : ''}`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Account Tab */}
        {activeTab === 'account' && (
          <div className="st-rows">
            {/* Display Name */}
            <div className="st-row">
              <label className="st-name" htmlFor="st-display-name">Display name</label>
              <div className="st-control st-inline">
                <input
                  id="st-display-name"
                  type="text"
                  value={displayName}
                  onChange={(e) => handleDisplayNameChange(e.target.value)}
                  className="oz-input"
                  placeholder="Your name"
                />
                <span className="st-status">
                  {nameSaveStatus === 'saving' && <Spinner />}
                  {nameSaveStatus === 'saved' && <Check className="st-ok" aria-label="Saved" />}
                  {nameSaveStatus === 'error' && <AlertCircle className="st-error" aria-label="Couldn't save" />}
                </span>
              </div>
            </div>

            {/* Email (read-only) */}
            <div className="st-row">
              <span className="st-name">Email</span>
              <div className="st-control st-value">{user.email}</div>
            </div>

            {/* Change Password */}
            <div className="st-row">
              <span className="st-name">Change password</span>
              <div className="st-control st-stack">
                <div className="st-password">
                  <input
                    type={showCurrentPassword ? 'text' : 'password'}
                    value={currentPassword}
                    onChange={(e) => setCurrentPassword(e.target.value)}
                    className="oz-input"
                    placeholder="Current password"
                  />
                  <button
                    type="button"
                    onClick={() => setShowCurrentPassword(!showCurrentPassword)}
                    className="oz-icon-btn"
                    aria-label={showCurrentPassword ? 'Hide password' : 'Show password'}
                  >
                    {showCurrentPassword ? <EyeOff /> : <Eye />}
                  </button>
                </div>
                <div className="st-password">
                  <input
                    type={showNewPassword ? 'text' : 'password'}
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    className="oz-input"
                    placeholder="New password"
                  />
                  <button
                    type="button"
                    onClick={() => setShowNewPassword(!showNewPassword)}
                    className="oz-icon-btn"
                    aria-label={showNewPassword ? 'Hide password' : 'Show password'}
                  >
                    {showNewPassword ? <EyeOff /> : <Eye />}
                  </button>
                </div>
                <input
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  className="oz-input"
                  placeholder="Confirm new password"
                />
                {passwordError && <p className="st-error">{passwordError}</p>}
                <button
                  onClick={handlePasswordChange}
                  disabled={!currentPassword || !newPassword || !confirmPassword || passwordStatus === 'saving'}
                  className={`oz-btn st-submit${passwordStatus === 'saving' ? '' : ' oz-btn--arrow'}`}
                >
                  {passwordStatus === 'saving' && <Spinner />}
                  {passwordStatus === 'saved' && <Check />}
                  Update Password
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Credit Balance Tab */}
        {activeTab === 'credits' && (
          <div className="st-rows">
            {/* Balance */}
            <div className="st-row">
              <span className="st-name">Available credits</span>
              <div className="st-control st-inline st-balance-row">
                <span className="st-balance">${user.available_balance.toFixed(2)}</span>
                <button onClick={handleAddCredits} className="oz-btn oz-btn--arrow">
                  Add Credits
                </button>
              </div>
            </div>

            {/* Low Balance Alert */}
            <div className="st-row">
              <label className="st-name" htmlFor="st-low-balance">
                Low balance alert threshold ($)
                <span className="oz-hint">Warn when balance drops below this amount</span>
              </label>
              <div className="st-control">
                <input
                  id="st-low-balance"
                  type="number"
                  min={0}
                  step={0.5}
                  value={lowBalanceAlert ?? settings?.credits.low_balance_alert ?? 5}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value) || 0
                    setLowBalanceAlert(val)
                    handleCreditsSetting('low_balance_alert', val, true)
                  }}
                  className="oz-input oz-input--number st-number"
                />
              </div>
            </div>

            {/* Show Balance in Navbar */}
            <div className="st-row">
              <label className="st-name" htmlFor="st-show-balance">
                Show balance in navbar
                <span className="oz-hint">Display credit balance next to your name</span>
              </label>
              <div className="st-control">
                <input
                  id="st-show-balance"
                  type="checkbox"
                  checked={showBalance}
                  onChange={() => handleCreditsSetting('show_balance_in_navbar', !showBalance)}
                />
              </div>
            </div>
          </div>
        )}

        {/* UI Preferences Tab */}
        {activeTab === 'ui' && (
          <div className="st-rows">
            {/* Background */}
            <div className="st-row">
              <span className="st-name">Background</span>
              <div className="st-control">
                <Dropdown
                  value={settings?.ui.background ?? 'system'}
                  onChange={(v) => handleUISetting('background', v)}
                  options={[
                    { value: 'dark', label: 'Dark' },
                    { value: 'light', label: 'Light' },
                    { value: 'system', label: 'System' },
                  ]}
                />
              </div>
            </div>

            {/* Default Landing Page */}
            <div className="st-row">
              <span className="st-name">Default landing page</span>
              <div className="st-control">
                <Dropdown
                  value={landingPath(settings)}
                  onChange={(v) => handleUISetting('default_page', v)}
                  options={APP_PAGES.map((p) => ({ value: p.path, label: p.label }))}
                />
              </div>
            </div>
          </div>
        )}
      </div>

      <SettingsStyles />
    </div>
  )
}

/** The page's styles, scoped under .settings-page: a title in the homepage's display type over ruled rows. */
function SettingsStyles() {
  return (
    <style>{`
      .settings-page {
        min-height: 125vh;
      }
      .settings-page .st-wrap {
        max-width: 62rem;
        padding: calc(var(--oz-nav-h) + 4rem) var(--oz-gutter) 6rem;
      }
      .settings-page .st-title {
        margin: 0 0 2.5rem;
        font-size: clamp(3rem, 6vw, 5.5rem);
        font-weight: 500;
        letter-spacing: -0.04em;
        line-height: 1;
      }
      /* A setting to a row: its name on the left, its control on the right */
      .settings-page .st-row {
        display: grid;
        grid-template-columns: minmax(0, 20rem) minmax(0, 28rem);
        align-items: start;
        gap: 1rem 3rem;
        padding: 1.5rem 0;
        border-bottom: 1px solid var(--oz-rule);
      }
      .settings-page .st-name {
        display: flex;
        flex-direction: column;
        gap: 0.2rem;
        padding-top: 0.5rem;
        font-size: 1.0625rem;
        font-weight: 500;
        letter-spacing: -0.02em;
      }
      .settings-page .st-name .oz-hint {
        font-weight: 400;
        letter-spacing: 0;
      }
      .settings-page .st-control {
        min-height: 2.6rem;
        display: flex;
        align-items: center;
      }
      .settings-page .st-control > .ozera-dropdown,
      .settings-page .st-control > .oz-input {
        flex: 1;
      }
      .settings-page .st-inline {
        gap: 0.75rem;
      }
      .settings-page .st-stack {
        flex-direction: column;
        align-items: stretch;
        gap: 6px;
      }
      .settings-page .st-value {
        color: var(--oz-fg-dim);
        font-size: 0.9375rem;
      }
      .settings-page .st-status {
        display: inline-flex;
        justify-content: center;
        width: 1.5rem;
      }
      .settings-page .st-status svg {
        width: 18px;
        height: 18px;
        stroke-width: 2.25;
      }
      .settings-page .st-ok {
        color: var(--oz-ok);
      }
      .settings-page .st-error {
        color: var(--oz-danger);
        font-size: 0.875rem;
      }
      .settings-page .st-password {
        position: relative;
      }
      .settings-page .st-password .oz-input {
        padding-right: 2.75rem;
      }
      .settings-page .st-password .oz-icon-btn {
        position: absolute;
        top: 50%;
        right: 0.3rem;
        transform: translateY(-50%);
      }
      .settings-page .st-submit {
        align-self: flex-start;
        margin-top: 0.75rem;
      }
      .settings-page .st-number {
        width: 9rem;
        flex: none;
      }
      .settings-page .st-balance-row {
        justify-content: space-between;
      }
      .settings-page .st-balance {
        font-size: 2.5rem;
        font-weight: 500;
        letter-spacing: -0.04em;
        line-height: 1;
        font-variant-numeric: tabular-nums;
      }
      @media (max-width: 800px) {
        .settings-page .st-row { grid-template-columns: minmax(0, 1fr); gap: 0.5rem; }
      }
    `}</style>
  )
}
