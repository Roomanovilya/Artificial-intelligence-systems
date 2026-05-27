class GameLogic:
    def __init__(self):
        self.current_turn = 1 
        self.capture_chain_pos = None 
        self.selected_piece = None

    def get_opponent(self, player):
        if player == 1:
            return 2
        return 1

    def is_my_piece(self, piece, player):
        if player == 1:
            return piece in [1, 3] 
        else:
            return piece in [2, 4] 

    def is_king(self, piece):
        return piece in [3, 4]

    def get_all_legal_moves(self, board, player, chain_position=None):
        captures = []

        if chain_position is not None:
            cells_to_check = [chain_position]
        else:
            cells_to_check = self.get_all_my_cells(board, player)
            
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)] 

        # Ищем все возможные взятия
        for row, col in cells_to_check:
            captures.extend(self.find_captures(board, row, col, player, directions))

        enemy_captures = []
        self_captures = []
        for cap in captures:
            if cap['is_own_piece']:
                self_captures.append(cap)
            else:
                enemy_captures.append(cap)

        if len(enemy_captures) > 0:
            return enemy_captures

        quiet_moves = []
        if chain_position is None:
            for row, col in self.get_all_my_cells(board, player):
                quiet_moves.extend(self.find_quiet_moves(board, row, col, player, directions))

        if len(self_captures) > 0:
            if chain_position is not None:
                self_captures.append({'from': chain_position, 'to': chain_position, 
                                      'captured': [], 'is_own_piece': False, 'end_turn': True})
            return self_captures + quiet_moves

        return quiet_moves

    def get_all_my_cells(self, board, player):
        cells = []
        for row in range(8):
            for col in range(8):
                piece = board.get_piece(row, col)
                if piece != 0 and self.is_my_piece(piece, player):
                    cells.append((row, col))
        return cells

    def find_captures(self, board, row, col, player, directions):
        captures = []
        piece = board.get_piece(row, col)
        
        for dir_row, dir_col in directions:
            if self.is_king(piece):
                captures.extend(self.get_king_captures(board, row, col, dir_row, dir_col, player))
            else:
                capture = self.get_man_capture(board, row, col, dir_row, dir_col, player)
                if capture is not None:
                    captures.append(capture)
        return captures

    def get_man_capture(self, board, row, col, dir_row, dir_col, player):
        mid_row = row + dir_row
        mid_col = col + dir_col
        target_row = row + 2 * dir_row
        target_col = col + 2 * dir_col
        
        if 0 <= target_row < 8 and 0 <= target_col < 8:
            mid_piece = board.get_piece(mid_row, mid_col)
            target_piece = board.get_piece(target_row, target_col)

            if mid_piece != 0 and target_piece == 0:
                return {
                    'from': (row, col), 
                    'to': (target_row, target_col), 
                    'captured': [(mid_row, mid_col)], 
                    'is_own_piece': self.is_my_piece(mid_piece, player)
                }
        return None

    def get_king_captures(self, board, row, col, dir_row, dir_col, player):
        captures = []
        current_row = row + dir_row
        current_col = col + dir_col
        victim_piece = None
        victim_pos = None

        while 0 <= current_row < 8 and 0 <= current_col < 8:
            piece_in_path = board.get_piece(current_row, current_col)
            
            if piece_in_path != 0:
                if victim_piece is not None:
                    break 
                victim_piece = piece_in_path
                victim_pos = (current_row, current_col)
            elif victim_piece is not None:
                captures.append({
                    'from': (row, col), 
                    'to': (current_row, current_col),
                    'captured': [victim_pos], 
                    'is_own_piece': self.is_my_piece(victim_piece, player)
                })
                
            current_row += dir_row
            current_col += dir_col
            
        return captures

    def find_quiet_moves(self, board, row, col, player, directions):
        moves = []
        piece = board.get_piece(row, col)
        
        if self.is_king(piece):
            allowed_dirs = directions
        else:
            if player == 1:
                allowed_dirs = [(-1, -1), (-1, 1)] 
            else:
                allowed_dirs = [(1, -1), (1, 1)]  

        for dir_row, dir_col in allowed_dirs:
            target_row = row + dir_row
            target_col = col + dir_col
            
            if self.is_king(piece):
                while 0 <= target_row < 8 and 0 <= target_col < 8:
                    if board.get_piece(target_row, target_col) == 0:
                        moves.append({'from': (row, col), 'to': (target_row, target_col),
                                      'captured': [], 'is_own_piece': False})
                    else:
                        break 
                    target_row += dir_row
                    target_col += dir_col
            else:
                if 0 <= target_row < 8 and 0 <= target_col < 8:
                    if board.get_piece(target_row, target_col) == 0:
                        moves.append({'from': (row, col), 'to': (target_row, target_col),
                                      'captured': [], 'is_own_piece': False})
        return moves

    def make_move(self, board, move, player):
        if move.get('end_turn'):
            self.switch_turn(player)
            return False

        from_row, from_col = move['from']
        to_row, to_col = move['to']
        moving_piece = board.get_piece(from_row, from_col)

        # Перемещаем шашку
        board.set_piece(from_row, from_col, 0)
        board.set_piece(to_row, to_col, moving_piece)

        # Убираем съеденные шашки
        for cap_row, cap_col in move['captured']:
            board.set_piece(cap_row, cap_col, 0)

        # Превращение в дамку
        if player == 1 and to_row == 0 and moving_piece == 1:
            board.set_piece(to_row, to_col, 3)
        elif player == 2 and to_row == 7 and moving_piece == 2:
            board.set_piece(to_row, to_col, 4)

        # Проверка на цепочку взятий
        if len(move['captured']) > 0:
            next_moves = self.get_all_legal_moves(board, player, chain_position=(to_row, to_col))

            has_next_captures = False
            for next_move in next_moves:
                if len(next_move.get('captured', [])) > 0 and not next_move.get('end_turn'):
                    has_next_captures = True
                    break
            
            if has_next_captures:
                self.capture_chain_pos = (to_row, to_col)
                self.selected_piece = (to_row, to_col)
                return True

        self.switch_turn(player)
        return False

    def switch_turn(self, player):
        self.capture_chain_pos = None
        self.selected_piece = None
        self.current_turn = self.get_opponent(player)