import numpy as np
import time

def get_initial_state():
    return np.full((3, 3), "_")  # Create tic-tac-toe board filled with underscores

def state_string(state):
    return "\n".join([" ".join(row) for row in state])  # Convert current 2D array into readable string format

def get_score(state):
    empty_spaces = np.count_nonzero(state == "_")  # Count the number of empty spaces
    magnitude = empty_spaces + 1  # Reward faster wins (earlier wins = higher score)

    for player, value in (("X", magnitude), ("O", -magnitude)):
        if (state == player).all(axis=0).any(): return value
        if (state == player).all(axis=1).any(): return value
        if (np.diag(state) == player).all(): return value
        if (np.diag(np.rot90(state)) == player).all(): return value
    return 0

def get_player(state):
    return "XO"[
        np.count_nonzero(state == "O") < np.count_nonzero(state == "X")]

def valid_actions(state):
    return list(zip(*np.nonzero(state == "_")))

def perform_action(state, action):
    state = state.copy()
    state[action] = get_player(state)
    return state

# Minimax with Alpha-Beta Pruning
def minimax_with_pruning(state, alpha=-np.inf, beta=np.inf):
    score = get_score(state)
    actions = valid_actions(state)

    if len(actions) == 0 or score != 0:  # Base case: no actions or game over
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
            alpha = max(alpha, utility)  # Update alpha
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
            beta = min(beta, utility)  # Update beta
            if beta <= alpha:  # Prune
                break
        return best_action, min_utility

# Regular Minimax (without Alpha-Beta Pruning)
def minimax_no_pruning(state):
    score = get_score(state)
    actions = valid_actions(state)

    if len(actions) == 0 or score != 0:  # Base case: no actions or game over
        return None, score

    player = get_player(state)
    utilities = []

    for action in actions:
        child = perform_action(state, action)
        _, child_utility = minimax_no_pruning(child) # Ignore first value(actions) since it's unnecessary
        utilities.append(child_utility)

    if player == "X":  # Maximizing player
        idx = np.argmax(utilities)
    else:  # Minimizing player
        idx = np.argmin(utilities)

    return actions[idx], utilities[idx]

# To compare Alpha-Beta Pruning vs. Regular Minimax
def compare_minimax(state):
    # Measure time for Alpha-Beta Pruning
    start_time = time.time()
    (r_prune, c_prune), utility_prune = minimax_with_pruning(state)
    pruning_time = time.time() - start_time

    # Measure time for regular Minimax (no pruning)
    start_time = time.time()
    (r_no_prune, c_no_prune), utility_no_prune = minimax_no_pruning(state)
    no_pruning_time = time.time() - start_time

    return pruning_time, no_pruning_time

# Function to run AI vs AI for a specified number of games and calculate speedup percentage
def run_automated_games(num_games):
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

            if player == "X":  # X is AI with Alpha-Beta Pruning
                pruning_time, no_pruning_time = compare_minimax(state)
                total_pruning_time += pruning_time
                total_no_pruning_time += no_pruning_time
                r, c = minimax_with_pruning(state)[0]
            else:  # O is AI without Alpha-Beta Pruning
                pruning_time, no_pruning_time = compare_minimax(state)
                total_pruning_time += pruning_time
                total_no_pruning_time += no_pruning_time
                r, c = minimax_no_pruning(state)[0]

            state = perform_action(state, (r, c))

    # Calculate average times
    avg_pruning_time = total_pruning_time / num_games
    avg_no_pruning_time = total_no_pruning_time / num_games

    # Calculate speedup percentage
    speedup_percentage = ((avg_no_pruning_time - avg_pruning_time) / avg_no_pruning_time) * 100

    print(f"\nAverage Alpha-Beta Pruning Time over {num_games} games: {avg_pruning_time:.6f} seconds")
    print(f"Average Regular Minimax Time over {num_games} games: {avg_no_pruning_time:.6f} seconds")
    print(f"Alpha-Beta Pruning is {speedup_percentage:.2f}% faster than Regular Minimax on average over {num_games} games.")

# Run 10 games
run_automated_games(10)
