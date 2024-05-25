#include<bits/stdc++.h>
#include<stdio.h>


using namespace std;

vector<vector<int>> create_board() {
    return vector<vector<int>>(6, vector<int>(7, 0));
}

void drop_piece(vector<vector<int>>& board, int row, int col, int piece) {
    board[row][col] = piece;
}

bool is_valid_location(const vector<vector<int>>& board, int col) {
    return board[6 - 1][col] == 0;
}

int get_next_open_row(const vector<vector<int>>& board, int col) {
    for (int r = 0; r < 6; r++) {
        if (board[r][col] == 0) {
            return r;
        }
    }
    return -1;
}

bool winning_move(const vector<vector<int>>& board, int piece) {
    // Check horizontal locations for win
    for (int c = 0; c < 7 - 3; c++) {
        for (int r = 0; r < 6; r++) {
            if (board[r][c] == piece && board[r][c + 1] == piece &&
                board[r][c + 2] == piece && board[r][c + 3] == piece) {
                return true;
            }
        }
    }
    // Check vertical locations for win
    for (int c = 0; c < 7; c++) {
        for (int r = 0; r < 6 - 3; r++) {
            if (board[r][c] == piece && board[r + 1][c] == piece &&
                board[r + 2][c] == piece && board[r + 3][c] == piece) {
                return true;
            }
        }
    }
    // Check positively sloped diagonals
    for (int c = 0; c < 7 - 3; c++) {
        for (int r = 0; r < 6 - 3; r++) {
            if (board[r][c] == piece && board[r + 1][c + 1] == piece &&
                board[r + 2][c + 2] == piece && board[r + 3][c + 3] == piece) {
                return true;
            }
        }
    }
    // Check negatively sloped diagonals
    for (int c = 0; c < 7 - 3; c++) {
        for (int r = 3; r < 6; r++) {
            if (board[r][c] == piece && board[r - 1][c + 1] == piece &&
                board[r - 2][c + 2] == piece && board[r - 3][c + 3] == piece) {
                return true;
            }
        }
    }
    return false;
}

int evaluate_window(const vector<int>& window, int piece) {
    int score = 0;
    int opponent_piece = (piece == 2) ? 1 : 2;
    int count_piece = count(window.begin(), window.end(), piece);
    int count_0 = count(window.begin(), window.end(), 0);
    int count_opponent = count(window.begin(), window.end(), opponent_piece);

    if (count_piece == 4) {
        score += 100;
    } else if (count_piece == 3 && count_0 == 1) {
        score += 5;
    } else if (count_piece == 2 && count_0 == 2) {
        score += 2;
    }

    if (count_opponent == 3 && count_0 == 1) {
        score -= 4;
    }

    return score;
}

int score_position(const vector<vector<int>>& board, int piece) {
    int score = 0;
    // Score center column higher as it's generally advantageous
    vector<int> center_array(6);
    for (int i = 0; i < 6; ++i) {
        center_array[i] = board[i][7 / 2];
    }
    int center_count = count(center_array.begin(), center_array.end(), piece);
    score += center_count * 3;

    // Score horizontal groups of four pieces
    for (int r = 0; r < 6; ++r) {
        vector<int> row_array(board[r].begin(), board[r].end());
        for (int c = 0; c < 7 - 3; ++c) {
            vector<int> window(row_array.begin() + c, row_array.begin() + c + 4);
            score += evaluate_window(window, piece);
        }
    }

    // Score vertical groups
    for (int c = 0; c < 7; ++c) {
        vector<int> col_array(6);
        for (int r = 0; r < 6; ++r) {
            col_array[r] = board[r][c];
        }
        for (int r = 0; r < 6 - 3; ++r) {
            vector<int> window(col_array.begin() + r, col_array.begin() + r + 4);
            score += evaluate_window(window, piece);
        }
    }

    // Score positive diagonal groups
    for (int r = 0; r < 6 - 3; ++r) {
        for (int c = 0; c < 7 - 3; ++c) {
            vector<int> window(4);
            for (int i = 0; i < 4; ++i) {
                window[i] = board[r + i][c + i];
            }
            score += evaluate_window(window, piece);
        }
    }

    // Score negative diagonal groups
    for (int r = 0; r < 6 - 3; ++r) {
        for (int c = 0; c < 7 - 3; ++c) {
            vector<int> window(4);
            for (int i = 0; i < 4; ++i) {
                window[i] = board[r + 3 - i][c + i];
            }
            score += evaluate_window(window, piece);
        }
    }

    return score;
}

vector<int> get_valid_locations(const vector<vector<int>>& board) {
    vector<int> valid_locations;
    for (int col = 0; col < 7; ++col) {
        if (is_valid_location(board, col)) {
            valid_locations.push_back(col);
        }
    }
    return valid_locations;
}

bool is_terminal_node(const vector<vector<int>>& board) {
    return winning_move(board, 1) || winning_move(board, 2) || get_valid_locations(board).empty();
}

pair<int, int> minimax(vector<vector<int>>& board, int depth, int alpha, int beta, bool maximizing_player) {
    vector<int> valid_locations = get_valid_locations(board);
    bool is_terminal = is_terminal_node(board);
    if (depth == 0 || is_terminal) {
        if (is_terminal) {
            if (winning_move(board, 2)) {
                return make_pair(-1, numeric_limits<int>::max());
            } else if (winning_move(board, 1)) {
                return make_pair(-1, numeric_limits<int>::min());
            } else {
                return make_pair(-1, 0);
            }
        } else {
            return make_pair(-1, score_position(board, 2));
        }
    }

    if (maximizing_player) {
        int value = numeric_limits<int>::min();
        int column = valid_locations[rand() % valid_locations.size()];
        for (int col : valid_locations) {
            int row = get_next_open_row(board, col);
            vector<vector<int>> board_copy = board;
            drop_piece(board_copy, row, col, 2);
            pair<int, int> new_score = minimax(board_copy, depth - 1, alpha, beta, false);
            if (new_score.second > value) {
                value = new_score.second;
                column = col;
            }
            alpha = max(alpha, value);
            if (alpha >= beta) {
                break;
            }
        }
        return make_pair(column, value);
    } else {
        int value = numeric_limits<int>::max();
        int column = valid_locations[rand() % valid_locations.size()];
        for (int col : valid_locations) {
            int row = get_next_open_row(board, col);
            vector<vector<int>> board_copy = board;
            drop_piece(board_copy, row, col, 1);
            pair<int, int> new_score = minimax(board_copy, depth - 1, alpha, beta, true);
            if (new_score.second < value) {
                value = new_score.second;
                column = col;
            }
            beta = min(beta, value);
            if (alpha >= beta) {
                break;
            }
        }
        return make_pair(column, value);
    }
}



void print_board(const vector<vector<int>>& board) {
    cout << "  ";
    for (int i = 0; i < 7; ++i) {
        cout << i << "  ";
    }
    cout << endl;
    cout << "------------------------" << endl;
    for (int r = 6 - 1; r >= 0; --r) {
        cout << "| ";
        for (int c = 0; c < 7; ++c) {
            if (board[r][c] == 0) {
                cout << " . ";
            } else if (board[r][c] == 1) {
                cout << " X ";
            } else {
                cout << " O ";
            }
        }
        cout << "|" << endl;
    }
    cout << "------------------------" << endl;
}