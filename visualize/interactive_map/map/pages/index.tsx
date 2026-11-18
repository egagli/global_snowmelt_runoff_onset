import React, { useEffect } from 'react'
import { Box } from 'theme-ui'
import { Map } from '../components/map'
import { Sidebar, SidebarContent, SheetSubtitle, CompactLegend } from '../components/sidebar'
import {
  FloatingCards,
  GmbaRangeCard,
  PointQueryCard,
  TopRightCard,
  WarningToast,
} from '../components/floating-cards'
import { MobileSheet } from '../components/mobile-sheet'
import { SearchBox } from '../components/search'
import { useStore } from '../lib/store'
import { SHEET_MQ } from '../lib/sheet'
import { syncUrlHash } from '../lib/url-hash'

export default function Home() {
  const sidebarWidth = useStore((s) => s.sidebarWidth)
  const queryKey = useStore((s) => (s.clickInfo ? `${s.clickInfo.lat},${s.clickInfo.lng}` : ''))

  // Shareable view state: apply the URL hash before the map mounts (Map only
  // renders once the sidebar has reported its width, a render later), then
  // keep the hash in sync.
  useEffect(() => syncUrlHash(), [])

  return (
    <Box
      sx={{
        position: 'absolute',
        top: 0,
        bottom: 0,
        left: 0,
        right: 0,
        overflow: 'hidden',
        [SHEET_MQ]: { overflow: 'clip' },
      }}
    >
      <Sidebar />
      {sidebarWidth !== null && (
        <>
          <Map />
          <SearchBox />
          <Box sx={{ display: 'block', [SHEET_MQ]: { display: 'none' } }}>
            <FloatingCards />
          </Box>
          <MobileSheet
            title='Global snowmelt runoff onset'
            subtitle={<SheetSubtitle />}
            legend={<CompactLegend />}
            queryKey={queryKey}
          >
            <PointQueryCard />
            <GmbaRangeCard />
            <SidebarContent extras={<TopRightCard />} />
          </MobileSheet>
          <Box sx={{ display: 'none', [SHEET_MQ]: { display: 'block' } }}>
            <WarningToast variant='banner' />
          </Box>
        </>
      )}
    </Box>
  )
}
