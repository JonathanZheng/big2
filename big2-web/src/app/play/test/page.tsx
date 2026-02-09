'use client';

import { useState, useCallback } from 'react';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { CardSelector, HistoryInput, SuggestionDisplay, HistoryMove } from '@/components/test-mode';
import { Card, PlayerIndex, analyzeSituation, MoveSuggestion } from '@/lib/game';

export default function TestModePage() {
  const [hand, setHand] = useState<Card[]>([]);
  const [position, setPosition] = useState<PlayerIndex>(0);
  const [history, setHistory] = useState<HistoryMove[]>([]);
  const [suggestion, setSuggestion] = useState<MoveSuggestion | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleAnalyze = useCallback(() => {
    if (hand.length === 0) {
      setError('Please select at least one card in your hand');
      return;
    }

    setIsLoading(true);
    setError(null);

    // Small delay for UX
    setTimeout(() => {
      try {
        const result = analyzeSituation({
          myHand: hand,
          myPosition: position,
          history: history.map((h) => ({
            player: h.player,
            cards: h.cards,
            isPass: h.isPass,
          })),
        });
        setSuggestion(result);
        setError(null);
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Analysis failed');
        setSuggestion(null);
      } finally {
        setIsLoading(false);
      }
    }, 100);
  }, [hand, position, history]);

  const handleReset = useCallback(() => {
    setHand([]);
    setHistory([]);
    setSuggestion(null);
    setError(null);
  }, []);

  // Calculate which cards are used in history (can't be in user's hand)
  const cardsInHistory = history.flatMap((m) => m.cards);

  return (
    <div className="min-h-screen bg-gradient-to-b from-purple-900 to-purple-950 p-4 md:p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <Link href="/">
            <Button variant="ghost" className="text-white hover:text-gray-300">
              ← Back to Home
            </Button>
          </Link>
          <h1 className="text-2xl font-bold text-white">Test Mode - AI Analysis</h1>
          <Button
            variant="ghost"
            className="text-purple-300 hover:text-white"
            onClick={handleReset}
          >
            Reset All
          </Button>
        </div>

        {/* Instructions */}
        <div className="bg-white/5 rounded-lg p-4 mb-6 text-purple-200 text-sm">
          <p>
            <strong>How to use:</strong> Enter your current hand (up to 13 cards), optionally add
            the game history (cards played by all players), then click "Analyze" to get AI
            suggestions for your best move.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left column - Input */}
          <div className="space-y-6">
            {/* Card selector */}
            <CardSelector
              selectedCards={hand}
              onSelectionChange={setHand}
              maxCards={13}
              disabledCards={cardsInHistory}
              title="Your Hand"
            />

            {/* Position selector */}
            <div className="bg-white/10 rounded-lg p-4">
              <label className="text-white font-medium block mb-2">Your Position</label>
              <select
                value={position}
                onChange={(e) => setPosition(Number(e.target.value) as PlayerIndex)}
                className="w-full p-2 rounded bg-white/10 text-white border border-white/20 focus:border-purple-400 focus:outline-none"
              >
                <option value={0} className="bg-purple-900">
                  Position 0 (First if holding 3♦)
                </option>
                <option value={1} className="bg-purple-900">
                  Position 1
                </option>
                <option value={2} className="bg-purple-900">
                  Position 2
                </option>
                <option value={3} className="bg-purple-900">
                  Position 3
                </option>
              </select>
              <p className="text-purple-300 text-xs mt-2">
                In Big 2, the player holding 3♦ goes first. Select your seating position (0-3).
              </p>
            </div>

            {/* History input */}
            <HistoryInput
              history={history}
              onHistoryChange={setHistory}
              userPosition={position}
              usedCards={hand}
            />
          </div>

          {/* Right column - Analysis */}
          <div className="space-y-6">
            {/* Analyze button */}
            <Button
              onClick={handleAnalyze}
              disabled={hand.length === 0 || isLoading}
              className="w-full py-6 text-lg bg-purple-600 hover:bg-purple-500 disabled:bg-gray-600 disabled:cursor-not-allowed"
            >
              {isLoading ? 'Analyzing...' : 'Analyze & Get Suggestion'}
            </Button>

            {/* Status info */}
            <div className="bg-white/5 rounded-lg p-4 text-sm">
              <div className="grid grid-cols-2 gap-4 text-purple-200">
                <div>
                  <span className="text-purple-400">Cards in hand:</span> {hand.length}/13
                </div>
                <div>
                  <span className="text-purple-400">Moves in history:</span> {history.length}
                </div>
                <div>
                  <span className="text-purple-400">Your position:</span> Player {position}
                </div>
                <div>
                  <span className="text-purple-400">Cards played:</span> {cardsInHistory.length}
                </div>
              </div>
            </div>

            {/* Suggestion display */}
            <SuggestionDisplay
              suggestion={suggestion}
              isLoading={isLoading}
              error={error}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
