import { Html, Head, Main, NextScript } from 'next/document'
import { GA_MEASUREMENT_ID } from '../lib/analytics'

// skip the page view when embedded in egagli.github.io/maps/ (the wrapper page already counts it)
const GA_INIT = `
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  window.gtag = gtag;
  gtag('js', new Date());
  var gaEmbedded = window.self !== window.top;
  gtag('set', { embedded: gaEmbedded });
  gtag('config', '${GA_MEASUREMENT_ID}', { send_page_view: !gaEmbedded });
`

export default function Document() {
  return (
    <Html lang='en'>
      <Head>
        <script async src={`https://www.googletagmanager.com/gtag/js?id=${GA_MEASUREMENT_ID}`} />
        <script dangerouslySetInnerHTML={{ __html: GA_INIT }} />
      </Head>
      <body>
        <Main />
        <NextScript />
      </body>
    </Html>
  )
}
