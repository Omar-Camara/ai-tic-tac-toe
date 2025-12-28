"""
Performance Benchmarking: Alpha-Beta Pruning vs Regular Minimax

This module compares the execution time of minimax with alpha-beta pruning
against standard minimax without pruning. Demonstrates ~97% performance
improvement with pruning while maintaining identical decision quality.

Author: Omar Camara
"""

import numpy as np
import time


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
    """
    empty_spaces = np.count_nonzero(state == "_")
    magnitude = empty_spaces + 1
    
    for player, value in (("X", magnitude), ("O", -magnitude)):
        if (state == player).all(axis=0).any(): return value
        if (state == player).all(axis=1).any(): return value
        if (np.diag(state) == player).all(): return value
        if (np.diag(np.rot90(state)) == player).all(): return value
    return 0


def get_player(state):
    """
    Determine whose turn it is based on current board state.
    
    Args:
        state (np.ndarray): Current 3x3 board state
    
    Returns:
        str: "X" or "O" indicating current player
    """
    return "XO"[np.count_nonzero(state == "O") < np.count_nonzero(state == "X")]


def valid_actions(state):
    """
    Get all valid moves (empty positions) on the board.
    
    Args:
        state (np.ndarray): Current 3x3 board state
    
    Returns:
        list: List of tuples (row, col) representing empty positions
    """
    return list(zip(*np.nonzero(state == "_")))


def perform_action(state, action):
    """
    Execute a move on the board for the current player.
    
    Args:
        state (np.ndarray): Current 3x3 board state
        action (tuple): (row, col) position to place the piece
    
    Returns:
        np.ndarray: New board state after the move
    """
    state = state.copy()
    state[action] = get_player(state)
    return state


def minimax_with_pruning(state, alpha=-np.inf, beta=np.inf):
    """
    Minimax algorithm WITH alpha-beta pruning optimization.
    
    Recursively explores the game tree while pruning branches that cannot
    affect the final decision. Significantly faster than regular minimax.
    
    Args:
        state (np.ndarray): Current 3x3 board state
        alpha (float): Best value maximizer can guarantee (pruning bound)
        beta (float): Best value minimizer can guarantee (pruning bound)
    
    Returns:
        tuple: (best_action, utility_value)
            best_action (tuple): (row, col) of optimal move
            utility_value (int): Expected utility of best move
    
    Time Complexity: O(b^(d/2)) best case, where b=branching factor, d=depth
    """
    score = get_score(state)
    actions = valid_actions(state)
    
    if len(actions) == 0 or score != 0:  # Base case
        return None, score
    
    player = get_player(state)
    
    if player == "X":  # Maximizing player
        max_utility = -np.inf
        best_action = None
        for action in actions:
            child = perform_action(state, action)
            _, utility = minimax_with_pruning(child, alpha, beta)
            if utility > max_utility:
                max_utility = utility
                best_action = action
            alpha = max(alpha, utility)
            if beta <= alpha:  # Prune
                break
        return best_action, max_utility
    
    else:  # Minimizing player
        min_utility = np.inf
        best_action = None
        for action in actions:
            child = perform_action(state, action)
            _, utility = minimax_with_pruning(child, alpha, beta)
            if utility < min_utility:
                min_utility = utility
                best_action = action
            beta = min(beta, utility)
            if beta <= alpha:  # Prune
                break
        return best_action, min_utility


def minimax_no_pruning(state):
    """
    Regular minimax algorithm WITHOUT alpha-beta pruning.
    
    Explores all branches of the game tree without optimization.
    Used as baseline for performance comparison.
    
    Args:
        state (np.ndarray): Current 3x3 board state
    
    Returns:
        tuple: (best_action, utility_value)
            best_action (tuple): (row, col) of optimal move
            utility_value (int): Expected utility of best move
    
    Time Complexity: O(b^d), where b=branching factor, d=depth
    Note: Significantly slower than pruned version (~35x slower)
    """
    score = get_score(state)
    actions = valid_actions(state)
    
    if len(actions) == 0 or score != 0:  # Base case
        return None, score
    
    player = get_player(state)
    utilities = []
    
    # Evaluate all children without pruning
    for action in actions:
        child = perform_action(state, action)
        _, child_utility = minimax_no_pruning(child)
        utilities.append(child_utility)
    
    # Select best move based on player
    if player == "X":  # Maximizing
        idx = np.argmax(utilities)
    else:  # Minimizing
        idx = np.argmin(utilities)
    
    return actions[idx], utilities[idx]


def compare_minimax(state):
    """
    Benchmark both algorithms on the same board state.
    
    Runs both pruned and non-pruned minimax from identical state
    and measures execution time for fair comparison.
    
    Args:
        state (np.ndarray): Current board state to evaluate
    
    Returns:
        tuple: (pruning_time, no_pruning_time)
            pruning_time (float): Seconds for alpha-beta version
            no_pruning_time (float): Seconds for regular version
    """
    # Measure Alpha-Beta Pruning performance
    start_time = time.time()
    (r_prune, c_prune), utility_prune = minimax_with_pruning(state)
    pruning_time = time.time() - start_time
    
    # Measure Regular Minimax performance
    start_time = time.time()
    (r_no_prune, c_no_prune), utility_no_prune = minimax_no_pruning(state)
    no_pruning_time = time.time() - start_time
    
    return pruning_time, no_pruning_time


def run_automated_games(num_games):
    """
    Run multiple AI vs AI games and calculate average performance metrics.
    
    Simulates complete games while benchmarking both algorithms at each
    decision point. Accumulates timing data across all games to compute
    average performance improvement.
    
    Args:
        num_games (int): Number of complete games to simulate
    
    Output:
        Prints performance statistics including:
        - Average execution time for pruned minimax
        - Average execution time for regular minimax
        - Percentage speedup from pruning
    
    Example Output:
        Average Alpha-Beta Pruning Time: 1.01 seconds
        Average Regular Minimax Time: 34.78 seconds
        Alpha-Beta Pruning is 97.11% faster
    """
    total_pruning_time = 0
    total_no_pruning_time = 0
    
    for game in range(num_games):
        print(f"\nStarting game {game + 1}")
        state = get_initial_state()
        
        while True:
            score = get_score(state)
            player = get_player(state)
            actions = valid_actions(state)
            
            if len(actions) == 0 or score != 0:  # Game over
                break
            
            # Benchmark both algorithms at each move
            if player == "X":  # X uses pruning
                pruning_time, no_pruning_time = compare_minimax(state)
                total_pruning_time += pruning_time
                total_no_pruning_time += no_pruning_time
                r, c = minimax_with_pruning(state)[0]
            else:  # O uses no pruning
                pruning_time, no_pruning_time = compare_minimax(state)
                total_pruning_time += pruning_time
                total_no_pruning_time += no_pruning_time
                r, c = minimax_no_pruning(state)[0]
            
            state = perform_action(state, (r, c))
    
    # Calculate statistics
    avg_pruning_time = total_pruning_time / num_games
    avg_no_pruning_time = total_no_pruning_time / num_games
    speedup_percentage = ((avg_no_pruning_time - avg_pruning_time) / avg_no_pruning_time) * 100
    
    # Display results
    print(f"\n{'='*70}")
    print(f"PERFORMANCE RESULTS ({num_games} games)")
    print(f"{'='*70}")
    print(f"Average Alpha-Beta Pruning Time: {avg_pruning_time:.6f} seconds")
    print(f"Average Regular Minimax Time: {avg_no_pruning_time:.6f} seconds")
    print(f"Alpha-Beta Pruning is {speedup_percentage:.2f}% faster than Regular Minimax")
    print(f"{'='*70}")


# Main execution
if __name__ == "__main__":
    run_automated_games(10)
