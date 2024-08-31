# Speech-Based Connect 4

This project implements a speech-based Connect 4 game using Hidden Markov Models (HMM) for speech recognition. The game allows a player to play against an AI where the player's moves are predicted based on their speech input.

## Overview

This repository contains the source code for a Connect 4 game enhanced with speech recognition capabilities. The speech recognition is achieved using Hidden Markov Models (HMM), which are trained to recognize player inputs.

## Features

- **Speech Recognition:** Predict player moves using HMM.
- **AI Opponent:** Play against an AI using the Minimax algorithm.
- **Interactive Gameplay:** Enjoy a classic game of Connect 4 with a modern twist.

## Prerequisites

- **C++ Compiler:** Ensure you have a C++ compiler (like `g++`) installed.
- **Libraries:** The project requires standard C++ libraries (`cmath`, `vector`, `string`, `iostream`, `fstream`, `iomanip`, `ctime`).
- **Data Files:** Speech recognition dataset (`234101048_universe.csv`).

## Functionality

### Hidden Markov Model (HMM) Functions

- **Initialization:** Initializes the HMM parameters.
- **Alpha Calculation:** Forward procedure for calculating alpha values.
- **Beta Calculation:** Backward procedure for calculating beta values.
- **Gamma Calculation:** Calculation of gamma values.
- **Re-Estimation:** Re-estimates the HMM parameters.
- **Model Training:** Trains the HMM with provided data.
- **Model Testing:** Tests the HMM with provided data.
- **Codebook Generation:** Generates a codebook from the given universe of speech data.
- **Codebook Loading:** Loads the generated codebook.

### Connect 4 Game Functions

- **Create Board:** Initializes the game board.
- **Drop Piece:** Drops a piece in the specified column.
- **Valid Location Check:** Checks if a column is a valid location for a move.
- **Next Open Row:** Finds the next open row in a column.
- **Winning Move Check:** Checks if the last move was a winning move.
- **Minimax Algorithm:** AI move decision-making using the minimax algorithm.
- **Print Board:** Prints the current state of the game board.

### Main Function

- Initializes the game and HMM parameters.
- Enters the game loop where Player 1's moves are predicted using the HMM and Player 2's moves are decided by the AI.
- Ends the game when a player wins or the board is full.
