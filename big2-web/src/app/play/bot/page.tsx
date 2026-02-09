'use client';

import Link from 'next/link';
import { useState, useCallback } from 'react';
import { GameBoard } from '@/components/game/GameBoard';
import { Button } from '@/components/ui/button';
import { recordGameResult, GameResultResponse } from './actions';
import { PlayerIndex } from '@/lib/game/types';

export default function PlayBotPage() {
  const [lastResult, setLastResult] = useState<{
    isWin: boolean;
    eloChange: number;
  } | null>(null);
  const [isRecording, setIsRecording] = useState(false);

  const handleGameEnd = useCallback(async (winner: PlayerIndex) => {
    setIsRecording(true);
    try {
      const result = await recordGameResult(winner, 0); // humanPlayer is always 0

      if (result.success && result.isWin !== undefined && result.eloChange !== undefined) {
        setLastResult({
          isWin: result.isWin,
          eloChange: result.eloChange,
        });
      }
    } catch (error) {
      console.error('Failed to record game result:', error);
    } finally {
      setIsRecording(false);
    }
  }, []);

  const handleNewGame = useCallback(() => {
    setLastResult(null);
  }, []);

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

        {lastResult && (
          <div
            className={`mb-4 p-3 rounded-lg text-center ${
              lastResult.isWin
                ? 'bg-green-600/20 text-green-300 border border-green-500/30'
                : 'bg-red-600/20 text-red-300 border border-red-500/30'
            }`}
          >
            <span className="font-semibold">
              {lastResult.isWin ? 'Victory!' : 'Defeat'}
            </span>
            <span className="mx-2">|</span>
            <span>
              ELO: {lastResult.eloChange >= 0 ? '+' : ''}
              {lastResult.eloChange}
            </span>
            {isRecording && <span className="ml-2 text-sm">(saving...)</span>}
          </div>
        )}

        <GameBoard
          humanPlayer={0}
          botDelay={800}
          onGameEnd={handleGameEnd}
          onNewGame={handleNewGame}
        />
      </div>
    </div>
  );
}
