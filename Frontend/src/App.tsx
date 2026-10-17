import './App.css'
import 'vite/modulepreload-polyfill'
import '@fontsource/open-sans/400.css'
import '@fontsource/open-sans/400-italic.css'
import '@fontsource/open-sans/600.css'
import '@fontsource/open-sans/700.css'
import Scrollbar from './pages/Main/components/Scrollbar/Scrollbar'
import StatusBar from './pages/Main/components/StatusBar/StatusBar'
import NodeView from './pages/NodeView/NodeView'
import { Provider } from 'jotai'
import { ThemeProvider } from '@mui/material/styles'
import { theme } from './theme'
function App() {

  return (
    <Provider>
      <ThemeProvider theme={theme}>
      <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <StatusBar />
        <Scrollbar />
      </div>
      {/* Reference inspector window — renders nothing until a node is clicked */}
      <NodeView />
      </ThemeProvider>
    </Provider>
  )
}

export default App
