import numpy as np

class Board:
    def __init__(self):
        # сетка 8х8 заполненная нулями
        self.grid = np.zeros((8, 8), dtype=int)
        self.reset()

    def reset(self):
        # шашки в начальное положение
        self.grid.fill(0) 
        
        # черные шашки сверху
        for row in range(3):
            start_col = 1 if row % 2 == 0 else 0
            self.grid[row, start_col::2] = 2
            
        # белые шашки снизу
        for row in range(5, 8):
            start_col = 1 if row % 2 == 0 else 0
            self.grid[row, start_col::2] = 1

    def get_piece(self, row, col):
        if 0 <= row < 8 and 0 <= col < 8:
            return self.grid[row, col]
        return 0

    def set_piece(self, row, col, value):
        if 0 <= row < 8 and 0 <= col < 8:
            self.grid[row, col] = value

    def copy(self):
        new_board = Board()
        new_board.grid = self.grid.copy()
        return new_board