import copy

class Minimax:
    def __init__(self, player_num=1, search_depth=3):
        self.player_num = player_num
        self.search_depth = search_depth

    def evaluate_board(self, board):
        score = 0
        for row in range(8):
            for col in range(8):
                piece = board.get_piece(row, col)
                
                if piece == 1:    
                    score += 10 + (7 - row)  
                elif piece == 3:  
                    score += 35
                elif piece == 2: 
                    score -= 10 + row        
                elif piece == 4: 
                    score -= 35
        
        if self.player_num == 2:
            score = -score
            
        return score

    def act(self, board, game_logic):
        moves = game_logic.get_all_legal_moves(board, self.player_num, game_logic.capture_chain_pos)
        if len(moves) == 0:
            return None

        best_score = -999999
        best_move = moves[0]

        for move in moves:
            new_board = board.copy()
            new_logic = copy.deepcopy(game_logic)
            new_logic.make_move(new_board, move, self.player_num)
            
            score = self.minimax(new_board, new_logic, self.search_depth - 1, -999999, 999999, False)
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move

    def minimax(self, board, game_logic, depth, alpha, beta, is_maximizing):
        current_player = game_logic.current_turn
        moves = game_logic.get_all_legal_moves(board, current_player, game_logic.capture_chain_pos)

        if depth == 0 or len(moves) == 0:
            return self.evaluate_board(board)
        
        if is_maximizing:
            max_score = -999999
            for move in moves:
                new_board = board.copy()
                new_logic = copy.deepcopy(game_logic)
                new_logic.make_move(new_board, move, current_player)
                
                eval_score = self.minimax(new_board, new_logic, depth - 1, alpha, beta, False)
                max_score = max(max_score, eval_score)
                alpha = max(alpha, eval_score)
                
                if beta <= alpha:
                    break 
            return max_score
        else:
            min_score = 999999
            for move in moves:
                new_board = board.copy()
                new_logic = copy.deepcopy(game_logic)
                new_logic.make_move(new_board, move, current_player)
                
                eval_score = self.minimax(new_board, new_logic, depth - 1, alpha, beta, True)
                min_score = min(min_score, eval_score)
                beta = min(beta, eval_score)
                
                if beta <= alpha:
                    break
            return min_score