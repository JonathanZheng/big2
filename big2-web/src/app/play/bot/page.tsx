'use client';

import Link from 'next/link';
import { GameBoard } from '@/components/game/GameBoard';
import { Button } from '@/components/ui/button';

export default function PlayBotPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 to-gray-950 p-4">
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center justify-between mb-4">
          <Link href="/">
            <Button variant="ghost" className="text-white hover:text-gray-300">
              ← Back to Home
            </Button>
          </Link>
          <h1 className="text-2xl font-bold text-white">Play vs Bots</h1>
          <div className="w-24" /> {/* Spacer for centering */}
        </div>

        <GameBoard humanPlayer={0} botDelay={800} />
      </div>
    </div>
  );
}
