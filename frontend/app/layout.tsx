import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'FinBot — AI Financial Research',
  description: 'AI-powered Indian & global stock market research platform',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-dark-900 text-white min-h-screen`}>
        {children}
      </body>
    </html>
  )
}
