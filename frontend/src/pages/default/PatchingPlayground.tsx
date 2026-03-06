import { NavBar } from '../../components/common/NavBar'

interface PatchingPlaygroundProps {
  onShowPurchaseCredits?: () => void
}

export default function DefaultPatchingPlayground({ onShowPurchaseCredits: _onShowPurchaseCredits }: PatchingPlaygroundProps) {
  return (
    <div className="min-h-screen pt-[70px]">
      <NavBar />
      <div className="max-w-7xl mx-auto px-6 py-8">
        <p className="text-gray-400">Default interface — coming soon</p>
      </div>
    </div>
  )
}
