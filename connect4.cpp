#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <stdlib.h>     /* srand, rand */
#include <ctime>       /* time */
#include "gamelogic.cpp"

using namespace std;

int main() {
    srand(time(0)); // Initialize random seed
    vector<vector<int>> board = create_board();
    bool game_over = false;
    int turn = 0;

    while (!game_over) {
        if (turn == 0) {
            int col;
            cout << "Player 1 make your selection (0-6): ";
            cin >> col;
            if (is_valid_location(board, col)) {
                int row = get_next_open_row(board, col);
                drop_piece(board, row, col, 1);
                if (winning_move(board, 1)) {
                    cout << "Player 1 wins!" << endl;
                    game_over = true;
                }
                print_board(board);
                cout << endl;
            }
        } else {
            pair<int, int> result = minimax(board, 5, numeric_limits<int>::min(), numeric_limits<int>::max(), true);
            int col = result.first;
            if (is_valid_location(board, col)) {
                int row = get_next_open_row(board, col);
                drop_piece(board, row, col, 2);
                if (winning_move(board, 2)) {
                    cout << "AI wins!" << endl;
                    game_over = true;
                }
                print_board(board);
            }
        }
        turn = 1 - turn;
    }
    return 0;
}
