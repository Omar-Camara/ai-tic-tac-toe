# Tic-Tac-Toe AI with Minimax & Alpha-Beta Pruning

An intelligent Tic-Tac-Toe game implementation featuring minimax algorithm with alpha-beta pruning optimization. Includes performance benchmarking showing **97% speed improvement** with pruning.

## 🎮 Features

- **Unbeatable AI** using minimax algorithm
- **Alpha-Beta Pruning** optimization for faster decision-making
- **Interactive gameplay** - Play against the AI or watch AI vs AI
- **Performance comparison** - Benchmark pruning vs non-pruning algorithms
- **Perfect play guarantee** - AI never loses

## 📊 Performance Results
```
Average Alpha-Beta Pruning Time: 1.01 seconds
Average Regular Minimax Time: 34.78 seconds
Speedup: 97.11% faster with Alpha-Beta Pruning
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy
```

### Run Interactive Game
```bash
python AiTicTacToe.py
```

Choose your mode:
- `1` - Play against the AI
- `2` - Watch AI play against itself

### Run Performance Benchmark
```bash
python AiTest.py
```

## 🎯 How It Works

### Minimax Algorithm
The AI uses the minimax algorithm to explore all possible game states and choose the optimal move. Each board state is scored based on:
- **Win for X**: Positive score (higher for faster wins)
- **Win for O**: Negative score (lower for faster wins)
- **Draw**: Score of 0

### Alpha-Beta Pruning
Optimization technique that eliminates branches in the game tree that don't need to be explored, reducing computation time by ~97% without affecting decision quality.

### Scoring System
```python
magnitude = empty_spaces + 1
# X wins: +magnitude (rewards faster wins)
# O wins: -magnitude (rewards faster wins)
# Draw: 0
```

## 📁 Project Structure
```
├── AiTicTacToe.py          # Interactive game with AI opponent
├── AiTest.py               # Performance benchmarking script
└── README.md               # Project documentation
```

## 🎲 Example Gameplay
```
Current state:
_ _ _
_ X _
_ _ _
Current player: O
Valid actions: [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
Minimax chose (0, 0) with utility 0
```

## 🧠 Key Concepts

- **Game Tree Search**: Exhaustive exploration of all possible game states
- **Utility Function**: Evaluates desirability of board positions
- **Pruning**: Eliminates unnecessary branches for efficiency
- **Perfect Information Game**: Complete knowledge of game state enables optimal play

## 📈 Complexity Analysis

- **Without Pruning**: O(b^d) where b=branching factor, d=depth
- **With Pruning**: O(b^(d/2)) in best case
- **Tic-Tac-Toe**: ~60,000 positions without pruning → ~5,000 with pruning

## 🔧 Customization

Modify the scoring function in `get_score()` to adjust AI behavior:
```python
magnitude = empty_spaces + 1  # Current: rewards faster wins
# magnitude = 1  # Alternative: all wins valued equally
```

## 📝 Technical Details

**Language**: Python 3.x  
**Dependencies**: NumPy  
**Algorithm**: Minimax with Alpha-Beta Pruning  
**Game Type**: Two-player, zero-sum, perfect information  

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- GUI implementation
- Difficulty levels (non-optimal play)
- Larger board sizes (4x4, 5x5)
- Opening book optimization

## 📄 License

MIT License - feel free to use for learning and projects

## 🎓 Educational Value

Perfect for learning:
- Game theory fundamentals
- Search algorithms
- Optimization techniques
- Recursion and tree traversal
- Python and NumPy

---

**Author**: Omar Camara
```
