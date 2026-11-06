import React, { useEffect } from 'react'
import { Box } from 'theme-ui'
import { Map } from '../components/map'
import { Sidebar } from '../components/sidebar'
import { FloatingCards } from '../components/floating-cards'
import { SearchBox } from '../components/search'
import { useStore } from '../lib/store'
import { syncUrlHash } from '../lib/url-hash'

export default function Home() {
  const sidebarWidth = useStore((s) => s.sidebarWidth)

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
      }}
    >
      <Sidebar />
      {sidebarWidth !== null && (
        <>
          <Map />
          <SearchBox />
          <Box sx={{ display: ['none', 'none', 'block'] }}>
            <FloatingCards />
          </Box>
        </>
      )}
    </Box>
  )
}
