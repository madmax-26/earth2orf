import { createTheme } from '@mui/material/styles'

// Font stacks live in index.css (:root custom properties); these are the
// handles for inline styles. Use HEADING_FONT on titles and section headers
// together with a 600/700 weight.
export const BODY_FONT = 'var(--font-body)'
export const HEADING_FONT = 'var(--font-heading)'

// MUI components (AppBar, Typography, IconButton) ship with Roboto by default —
// point them at the same fonts as everything else.
export const theme = createTheme({
  typography: {
    fontFamily: BODY_FONT,
    h1: { fontFamily: HEADING_FONT, fontWeight: 700 },
    h2: { fontFamily: HEADING_FONT, fontWeight: 700 },
    h3: { fontFamily: HEADING_FONT, fontWeight: 700 },
    h4: { fontFamily: HEADING_FONT, fontWeight: 600 },
    h5: { fontFamily: HEADING_FONT, fontWeight: 600 },
    h6: { fontFamily: HEADING_FONT, fontWeight: 600 },
  },
})
