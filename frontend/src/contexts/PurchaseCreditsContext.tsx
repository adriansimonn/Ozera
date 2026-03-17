import { createContext, useContext, type ReactNode } from 'react'

interface PurchaseCreditsContextType {
  showPurchaseCredits: () => void
}

const PurchaseCreditsContext = createContext<PurchaseCreditsContextType | null>(null)

export function usePurchaseCredits(): PurchaseCreditsContextType {
  const ctx = useContext(PurchaseCreditsContext)
  if (!ctx) {
    throw new Error('usePurchaseCredits must be used within PurchaseCreditsProvider')
  }
  return ctx
}

export function PurchaseCreditsProvider({
  children,
  onShow,
}: {
  children: ReactNode
  onShow: () => void
}) {
  return (
    <PurchaseCreditsContext.Provider value={{ showPurchaseCredits: onShow }}>
      {children}
    </PurchaseCreditsContext.Provider>
  )
}
