import React from 'react'
import { Box } from 'theme-ui'
import { Map } from '../components/map'
import { Sidebar } from '../components/sidebar'
import { FloatingCards } from '../components/floating-cards'
import { useStore } from '../lib/store'

export default function Home() {
  const sidebarWidth = useStore((s) => s.sidebarWidth)

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
          <Box sx={{ display: ['none', 'none', 'block'] }}>
            <FloatingCards />
          </Box>
        </>
      )}
    </Box>
  )
}
