import React from 'react'
import Head from 'next/head'
import type { AppProps } from 'next/app'
import { ThemeProvider } from 'theme-ui'
import theme from '@carbonplan/theme'
import '@carbonplan/components/fonts.css'
import '@carbonplan/components/globals.css'
import 'maplibre-gl/dist/maplibre-gl.css'

// Absolute URLs: Open Graph / Twitter scrapers do not resolve relative paths.
const SITE = 'https://egagli.github.io/global_snowmelt_runoff_onset'
const TITLE = 'Global snowmelt runoff onset from Sentinel-1 SAR'
const DESCRIPTION =
  'Interactive map of a global 80 m dataset of snowmelt runoff onset timing from ' +
  'Sentinel-1 SAR, water years 2015-2025. Described in Gagliano et al. (2026), ' +
  'Earth System Science Data.'

const App = ({ Component, pageProps }: AppProps) => (
  <ThemeProvider theme={theme}>
    <Head>
      <title>{TITLE}</title>
      <meta name='viewport' content='width=device-width, initial-scale=1' />
      <meta name='description' content={DESCRIPTION} />
      <meta name='theme-color' content='#1b1e23' />
      <link rel='canonical' href={`${SITE}/`} />

      <meta property='og:type' content='website' />
      <meta property='og:url' content={`${SITE}/`} />
      <meta property='og:title' content={TITLE} />
      <meta property='og:description' content={DESCRIPTION} />
      <meta property='og:image' content={`${SITE}/og.png`} />
      <meta property='og:image:width' content='1200' />
      <meta property='og:image:height' content='630' />
      <meta
        property='og:image:alt'
        content='Northern Hemisphere polar map of 11-year median snowmelt runoff onset date'
      />

      <meta name='twitter:card' content='summary_large_image' />
      <meta name='twitter:title' content={TITLE} />
      <meta name='twitter:description' content={DESCRIPTION} />
      <meta name='twitter:image' content={`${SITE}/og.png`} />
    </Head>
    <Component {...pageProps} />
  </ThemeProvider>
)

export default App
