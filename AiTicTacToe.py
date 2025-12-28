"""
Tic-Tac-Toe AI Game with Minimax Algorithm and Alpha-Beta Pruning

This module implements an interactive Tic-Tac-Toe game featuring an unbeatable AI
opponent using the minimax algorithm with alpha-beta pruning optimization.

Author: Omar Camara
"""

import numpy as np


def get_initial_state():
    """
    Create an empty 3x3 Tic-Tac-Toe board.
    
    Returns:
        np.ndarray: 3x3 array filled with "_" representing empty spaces
    """
    return np.full((3, 3), "_")


def state_string(state):
    """
    Convert board state to human-readable string format.
    
    Args:
        state (np.ndarray): Current 3x3 board state
    
    Returns:
        str: Formatted board string with rows separated by newlines
    
    Example:
        X O _
        _ X O
        O _ X
    """
    return "\n".join([" ".join(row) for row in state])


def get_score(state):
    """
    Evaluate the current board state and return its utility score.
    
    Scoring system rewards faster wins:
    - X wins: positive score (magnitude = empty_spaces + 1)
    - O wins: negative score (-magnitude)
    - Draw/ongoing: 0
    
    Args:
        state (np.ndarray): Current 3x3 board state
    
    Returns:
        int: Utility score for the state
            Positive if X wins, negative if O wins, 0 otherwise
    """
    # Count the number of empty spaces
    empty_spaces = np.count_nonzero(state == "_")
    
    # Reward faster wins: earlier wins (with more empty spaces) = higher score
    magnitude = empty_spaces + 1  # +1 to avoid multiplying by 0
    
    # Check all winning conditions for both players
    for player, value in (("X", magnitude), ("O", -magnitude)):
        if (state == player).all(axis=0).any(): return value  # Column win
        if (state == player).all(axis=1).any(): return value  # Row win
        if (np.diag(state) == player).all(): return value  # Main diagonal win
        if (np.diag(np.rot90(state)) == player).all(): return value  # Anti-diagonal win
    
    return 0


def get_player(state):
    """
    Determine whose turn it is based on current board state.
    
    X always goes first. Player is determined by counting pieces on board.
    
    Args:
        state (np.ndarray): Current 3x3 board state
    
    Returns:
        str: "X" or "O" indicating current player
    """
    # If O's count < X's count, it's O's turn; otherwise it's X's turn
    return "XO"[np.count_nonzero(state == "O") < np.count_nonzero(state == "X")]


def valid_actions(state):
    """
    Get all valid moves (empty positions) on the board.
    
    Args:
        state (np.ndarray): Current 3x3 board state
    
    Returns:
        list: List of tuples (row, col) representing empty positions
        
    Example:
        [(0, 0), (0, 2), (1, 1), (2, 2)]
    """
    return list(zip(*np.nonzero(state == "_")))


def perform_action(state, action):
    """
    Execute a move on the board for the current player.
    
    Creates a copy of the state to avoid modifying the original.
    
    Args:
        state (np.ndarray): Current 3x3 board state
        action (tuple): (row, col) position to place the piece
    
    Returns:
        np.ndarray: New board state after the move
    """
    state = state.copy()
    state[action] = get_player(state)
    return state


def children_of(state):
    """
    Generate all possible next states from the current state.
    
    Args:
        state (np.ndarray): Current 3x3 board state
    
    Returns:
        list: List of np.ndarray representing all possible next states
    """
    symbol = get_player(state)
    children = []
    for r in range(state.shape[0]):
        for c in range(state.shape[1]):
            if state[r, c] == "_":
                child = state.copy()
                child[r, c] = symbol
                children.append(child)
    return children


def minimax(state, alpha=-np.inf, beta=np.inf):
    """
    Minimax algorithm with alpha-beta pruning for optimal move selection.
    
    Recursively explores the game tree to find the best move for the current player.
    Alpha-beta pruning eliminates branches that cannot influence the final decision,
    reducing computation time by ~97% without affecting move quality.
    
    Args:
        state (np.ndarray): Current 3x3 board state
        alpha (float): Best value the maximizer (X) can guarantee so far
        beta (float): Best value the minimizer (O) can guarantee so far
    
    Returns:
        tuple: (best_action, utility_value)
            best_action (tuple): (row, col) of optimal move, or None if game over
            utility_value (int): Expected utility of the best move
    
    Algorithm:
        - Maximizing player (X): Choose move with highest utility
        - Minimizing player (O): Choose move with lowest utility
        - Prune branches when beta <= alpha (no better move possible)
    """
    score = get_score(state)
    actions = valid_actions(state)
    
    # Base case: no valid actions or game over
    if len(actions) == 0 or score != 0:
        return None, score
    
    player = get_player(state)
    
    if player == "X":  # Maximizing player
        max_utility = -np.inf
        best_action = None
        for action in actions:
            child = perform_action(state, action)
            _, utility = minimax(child, alpha, beta)
            if utility > max_utility:
                max_utility = utility
                best_action = action
            alpha = max(alpha, utility)  # Update alpha
            if beta <= alpha:  # Prune remaining branches
                break
        return best_action, max_utility
    
    else:  # Minimizing player (O)
        min_utility = np.inf
        best_action = None
        for action in actions:
            child = perform_action(state, action)
            _, utility = minimax(child, alpha, beta)
            if utility < min_utility:
                min_utility = utility
                best_action = action
            beta = min(beta, utility)  # Update beta
            if beta <= alpha:  # Prune remaining branches
                break
        return best_action, min_utility


def playAgainstAi(state):
    """
    Interactive game mode: Human (X) vs AI (O).
    
    Human player inputs moves in format 'row,col' (e.g., '0,1').
    AI uses minimax with alpha-beta pruning for optimal play.
    
    Args:
        state (np.ndarray): Initial board state (usually empty)
    """
    while True:
        score = get_score(state)
        player = get_player(state)
        actions = valid_actions(state)
        
        # Check for game over conditions
        if len(actions) == 0: break
        if score != 0: break
        
        print("\nCurrent state:")
        print(state_string(state))
        print("Current player:", player)
        print("Valid actions:", actions)
        
        if player == "X":
            # Human player's turn
            choice = input("Choose action in format 'r,c': ")
            try:
                r, c = map(int, choice.split(","))
                assert (0 <= r < 3) and (0 <= c < 3)
            except:
                print("Invalid choice", choice)
                continue
        else:
            # AI's turn
            (r, c), utility = minimax(state)
            print(f"Minimax chose {(r, c)} with utility {utility}")
        
        state = perform_action(state, (r, c))
    
    # Display final result
    if score < 0:
        message = "You lost, try again!"
    elif score > 0:
        message = "You won!!"
    else:
        message = "Draw"
    
    print(message)
    print("Game over, score =", score)
    print(state_string(state))


def AiAgainstAi(state):
    """
    Automated game mode: AI (X) vs AI (O).
    
    Both players use minimax with alpha-beta pruning.
    Demonstrates perfect play resulting in a draw.
    
    Args:
        state (np.ndarray): Initial board state (usually empty)
    """
    while True:
        score = get_score(state)
        player = get_player(state)
        actions = valid_actions(state)
        
        # Check for game over conditions
        if len(actions) == 0: break
        if score != 0: break
        
        print("\nCurrent state:")
        print(state_string(state))
        print("Current player:", player)
        print("Valid actions:", actions)
        
        # AI makes move
        (r, c), utility = minimax(state)
        print(f"Minimax chose {(r, c)} with utility {utility}")
        
        state = perform_action(state, (r, c))
    
    print("Game over, score =", score)
    print(state_string(state))


# Main execution
if __name__ == "__main__":
    state = get_initial_state()
    
    answer = int(input("Type 1 if you want to try and beat the AI or 2 if you want to watch AI play itself\n"))
    if answer == 1:
        playAgainstAi(state)
    elif answer == 2:
        AiAgainstAi(state)
    else:
        print("Invalid option")
