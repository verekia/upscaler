import dynamic from 'next/dynamic'

const App = dynamic(() => import('../src/App').then(m => m.App), { ssr: false })

const Page = () => <App />

export default Page
