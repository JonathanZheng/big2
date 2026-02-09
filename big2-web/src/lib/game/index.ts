/**
 * Big 2 Game Engine - TypeScript implementation
 */

// Types
export * from './types';

// Constants
export * from './constants';

// Game Engine
export { Big2Game, createGame, createMove, isPass, moveToString } from './game-engine';

// Move Detection
export { detectMoveType, getMoveValue, canBeat, getMoveTypeName } from './move-detector';

// Move Generation
export { getLegalMoves } from './move-generator';

// Greedy Bot
export { selectActionGreedyBot } from './greedy-bot';

// Test Mode
export { analyzeSituation } from './test-mode';
export type { TestModeState, TestModeMove, MoveSuggestion } from './test-mode';
