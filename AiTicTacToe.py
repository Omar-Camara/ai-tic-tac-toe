# state[r,c] is the content at position (r,c) of the board: "_", "X", or "O"
import numpy as np

def get_initial_state():
    return np.full((3,3), "_") # create tic-tac-toe board filled with underscores to represent empty spaces

def state_string(state):
    return "\n".join([" ".join(row) for row in state]) #convert current 2D NumPy array into readable string format

def get_score(state):
    # Count the number of empty spaces
    empty_spaces = np.count_nonzero(state == "_")

    # Reward faster wins: earlier wins (with more empty spaces) will have a higher score
    magnitude = empty_spaces + 1  # +1 to avoid multiplying by 0

    for player, value in (("X", magnitude), ("O", -magnitude)):
        if (state == player).all(axis=0).any(): return value
        if (state == player).all(axis=1).any(): return value
        if (np.diag(state) == player).all(): return value
        if (np.diag(np.rot90(state)) == player).all(): return value
    return 0

def get_player(state):
    # If O's < X's, it's player O's turn, otherwise it's X's turn
    return "XO"[
        np.count_nonzero(state == "O") < np.count_nonzero(state == "X")]

def valid_actions(state):
    # return list of positions with '_' (empty)
  return list(zip(*np.nonzero(state == "_")))

def perform_action(state, action):
    # Update the board with the move and location of the current player
  state = state.copy()
  state[action] = get_player(state)
  return state

def children_of(state):
    # Gets all possible next states of current state
    symbol = get_player(state)
    children = []
    for r in range(state.shape[0]):
        for c in range(state.shape[1]):
            if state[r,c] == "_":
                child = state.copy()
                child[r,c] = symbol
                children.append(child)
    return children
    
# Minimax with Alpha-Beta Pruning
def minimax(state, alpha=-np.inf, beta=np.inf):
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
            if beta <= alpha:  # Prune
                break
        return best_action, max_utility

    else:  # Minimizing player
        min_utility = np.inf
        best_action = None
        for action in actions:
            child = perform_action(state, action)
            _, utility = minimax(child, alpha, beta)
            if utility < min_utility:
                min_utility = utility
                best_action = action
            beta = min(beta, utility)  # Update beta
            if beta <= alpha:  # Prune
                break
        return best_action, min_utility
    
state = get_initial_state()
def playAgainstAi(state):
    while True:
    
      score = get_score(state)
      player = get_player(state)
      actions = valid_actions(state)
    
      if len(actions) == 0: break
      if score != 0: break
    
      print("\nCurrent state:")
      print(state_string(state))
      print("Current player:", player)
      print("Valid actions:", actions)
    
      if player == "X":
    
          choice = input("Choose action in format 'r,c': ")
          try:
            r, c = map(int, choice.split(","))
            assert (0 <= r < 3) and (0 <= c < 3)
          except:
            print("Invalid choice", choice)
            continue
    
      else:
    
          (r,c), utility = minimax(state)
          print(f"Minimax chose {(r,c)} with utility {utility}")
    
      state = perform_action(state, (r,c))
    
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
    while True:
    
      score = get_score(state)
      player = get_player(state)
      actions = valid_actions(state)
    
      if len(actions) == 0: break
      if score != 0: break
    
      print("\nCurrent state:")
      print(state_string(state))
      print("Current player:", player)
      print("Valid actions:", actions)
    
      (r,c), utility = minimax(state)
      print(f"Minimax chose {(r,c)} with utility {utility}")
    
      state = perform_action(state, (r,c))
    
    print("Game over, score =", score)
    print(state_string(state))

answer = int(input("Type 1 if you want to try and beat the AI or 2 if you want to watch AI play itself\n"))
if answer == 1:
    playAgainstAi(state)
elif answer == 2: 
    AiAgainstAi(state)
else:
    print("Invalid option")
