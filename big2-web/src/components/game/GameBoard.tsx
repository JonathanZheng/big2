'use client';

import { useState, useCallback, useEffect } from 'react';
import { cn } from '@/lib/utils';
import { PlayerHand } from './PlayerHand';
import { PlayArea, SelectedCardsPreview } from './PlayArea';
import { MoveControls } from './MoveControls';
import {
  Big2Game,
  createGame,
  createMove,
  isPass,
  getLegalMoves,
  selectActionGreedyBot,
  PlayerIndex,
  Move,
} from '@/lib/game';

interface GameBoardProps {
  humanPlayer?: PlayerIndex;
  botDelay?: number;
  onGameEnd?: (winner: PlayerIndex) => void;
}

type GamePhase = 'playing' | 'ended';

export function GameBoard({ humanPlayer = 0, botDelay = 1000, onGameEnd }: GameBoardProps) {
  const [game, setGame] = useState<Big2Game | null>(null);
  const [selectedCards, setSelectedCards] = useState<number[]>([]);
  const [gamePhase, setGamePhase] = useState<GamePhase>('playing');
  const [isProcessingBotTurn, setIsProcessingBotTurn] = useState(false);

  // Initialize game
  useEffect(() => {
    const newGame = createGame();
    setGame(newGame);
  }, []);

  // Handle bot turns
  useEffect(() => {
    if (!game || gamePhase === 'ended' || isProcessingBotTurn) return;

    const currentPlayer = game.getCurrentPlayer();
    if (currentPlayer === humanPlayer) return;

    // Bot turn
    setIsProcessingBotTurn(true);

    const timer = setTimeout(() => {
      const botMove = selectActionGreedyBot(game, currentPlayer);
      game.step(botMove);

      if (game.isDone()) {
        setGamePhase('ended');
        onGameEnd?.(game.getWinner()!);
      }

      setGame(game);
      setIsProcessingBotTurn(false);
    }, botDelay);

    return () => clearTimeout(timer);
  }, [game, humanPlayer, botDelay, gamePhase, isProcessingBotTurn, onGameEnd]);

  const handleCardClick = useCallback((card: number) => {
    setSelectedCards((prev) =>
      prev.includes(card) ? prev.filter((c) => c !== card) : [...prev, card]
    );
  }, []);

  const handlePlay = useCallback(() => {
    if (!game) return;

    const move = createMove(selectedCards, humanPlayer);
    game.step(move);

    if (game.isDone()) {
      setGamePhase('ended');
      onGameEnd?.(game.getWinner()!);
    }

    setSelectedCards([]);
    setGame(game);
  }, [game, selectedCards, humanPlayer, onGameEnd]);

  const handlePass = useCallback(() => {
    if (!game) return;

    const move = createMove([], humanPlayer);
    game.step(move);

    setSelectedCards([]);
    setGame(game);
  }, [game, humanPlayer]);

  const handleClear = useCallback(() => {
    setSelectedCards([]);
  }, []);

  const handleNewGame = useCallback(() => {
    const newGame = createGame();
    setGame(newGame);
    setSelectedCards([]);
    setGamePhase('playing');
    setIsProcessingBotTurn(false);
  }, []);

  if (!game) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-gray-500">Loading game...</div>
      </div>
    );
  }

  const currentPlayer = game.getCurrentPlayer();
  const isHumanTurn = currentPlayer === humanPlayer;
  const lastMove = game.getLastMove();
  const canPass = lastMove !== null && !game.isFirstMove();

  // Get player hands
  const playerHands = [0, 1, 2, 3].map((i) => game.getHand(i as PlayerIndex));

  return (
    <div className="relative w-full h-full min-h-[700px] bg-green-800 rounded-xl p-4 flex flex-col">
      {/* Game end overlay */}
      {gamePhase === 'ended' && (
        <div className="absolute inset-0 bg-black/60 flex items-center justify-center z-20 rounded-xl">
          <div className="bg-white p-8 rounded-xl shadow-2xl text-center">
            <h2 className="text-3xl font-bold mb-4">
              {game.getWinner() === humanPlayer ? '🎉 You Won!' : 'Game Over'}
            </h2>
            <p className="text-gray-600 mb-6">
              {game.getWinner() === humanPlayer
                ? 'Congratulations!'
                : `Player ${game.getWinner()} wins!`}
            </p>
            <button
              onClick={handleNewGame}
              className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
            >
              Play Again
            </button>
          </div>
        </div>
      )}

      {/* Top player (Player 2) */}
      <div className="flex justify-center mb-4">
        <PlayerHand
          cards={playerHands[2]}
          selectedCards={[]}
          position="top"
          playerName="Bot 2"
          showCards={false}
          isCurrentPlayer={currentPlayer === 2}
        />
      </div>

      {/* Middle section */}
      <div className="flex flex-1 items-center justify-between px-4">
        {/* Left player (Player 1) */}
        <div className="flex items-center">
          <PlayerHand
            cards={playerHands[1]}
            selectedCards={[]}
            position="left"
            playerName="Bot 1"
            showCards={false}
            isCurrentPlayer={currentPlayer === 1}
          />
        </div>

        {/* Center - Play Area */}
        <div className="flex-1 flex flex-col items-center justify-center">
          <div className="bg-green-700/50 rounded-xl p-6 min-w-[300px]">
            <PlayArea
              lastMove={lastMove}
              message={game.isFirstMove() ? 'First move - must include 3♦' : undefined}
            />
          </div>

          {/* Turn indicator */}
          <div className="mt-4 text-white/80 text-sm">
            {isProcessingBotTurn ? (
              <span className="flex items-center gap-2">
                <span className="animate-spin">⚙️</span>
                Bot {currentPlayer} is thinking...
              </span>
            ) : isHumanTurn ? (
              <span className="text-yellow-300 font-medium">Your turn!</span>
            ) : (
              <span>Waiting for Player {currentPlayer}...</span>
            )}
          </div>
        </div>

        {/* Right player (Player 3) */}
        <div className="flex items-center">
          <PlayerHand
            cards={playerHands[3]}
            selectedCards={[]}
            position="right"
            playerName="Bot 3"
            showCards={false}
            isCurrentPlayer={currentPlayer === 3}
          />
        </div>
      </div>

      {/* Bottom section - Human player */}
      <div className="mt-4 flex flex-col items-center gap-4">
        {/* Selected cards preview */}
        {isHumanTurn && (
          <div className="bg-green-700/30 rounded-lg px-6 py-3">
            <SelectedCardsPreview cards={selectedCards} />
          </div>
        )}

        {/* Controls */}
        {isHumanTurn && (
          <MoveControls
            selectedCards={selectedCards}
            lastMove={lastMove}
            isFirstMove={game.isFirstMove()}
            canPass={canPass}
            onPlay={handlePlay}
            onPass={handlePass}
            onClear={handleClear}
            disabled={gamePhase === 'ended'}
          />
        )}

        {/* Human player's hand */}
        <PlayerHand
          cards={playerHands[humanPlayer]}
          selectedCards={selectedCards}
          onCardClick={handleCardClick}
          position="bottom"
          playerName="You"
          showCards={true}
          isCurrentPlayer={isHumanTurn}
          disabled={!isHumanTurn || gamePhase === 'ended'}
        />
      </div>
    </div>
  );
}
