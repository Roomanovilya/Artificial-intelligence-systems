import torch
import torch.nn as nn
import numpy as np
import random
import os

class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 256),  
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4096)  
        )

    def forward(self, state):
        return self.net(state)

class Bot:
    def __init__(self, player=2):
        self.player = player
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QNetwork().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_function = nn.MSELoss()
        
        self.memory = []
        self.gamma = 0.99      
        self.epsilon = 1.0     
        self.eps_min = 0.10     
        self.eps_decay = 0.9995  
        self.train_step = 0
        self.sacrifice_probability = 0.3

    def get_state(self, board):
        flat_board = board.grid.flatten().astype(np.float32)
        return torch.FloatTensor(flat_board).to(self.device)

    def encode_move(self, move):
        if not move: 
            return 0
        from_row, from_col = move['from']
        to_row, to_col = move.get('to', (from_row, from_col))
        from_index = from_row * 8 + from_col
        to_index = to_row * 8 + to_col
        return from_index * 64 + to_index

    def is_sacrifice_for_capture(self, move, board, possible_moves):
        if not move.get('captured') or not move.get('is_own_piece'):
            return False
        
        try:
            simulated_board = board.copy() if hasattr(board, 'copy') else board

            from_row, from_col = move['from']
            to_row, to_col = move['to']

            simulated_board.grid[to_row][to_col] = simulated_board.grid[from_row][from_col]
            simulated_board.grid[from_row][from_col] = 0
            

            if hasattr(simulated_board, 'get_possible_moves'):
                next_moves = simulated_board.get_possible_moves(self.player)
            else:
                return False
            
            for next_move in next_moves:
                if next_move.get('captured') and not next_move.get('is_own_piece'):
                    return True
                    
        except Exception as e:
            return False
        
        return False

    def filter_sacrifice_moves(self, moves, board):

        enemy_captures = []     
        sacrifice_then_capture = [] 
        self_captures = []      
        quiet_moves = []    
        end_moves = []         
        
        for m in moves:
            if m.get('end'):
                end_moves.append(m)
            elif m.get('captured'):
                if m.get('is_own_piece'):
                    if self.is_sacrifice_for_capture(m, board, moves):
                        sacrifice_then_capture.append(m)
                    else:
                        self_captures.append(m)
                else:
                    enemy_captures.append(m)
            else:
                quiet_moves.append(m)
        
        return enemy_captures, sacrifice_then_capture, self_captures, quiet_moves, end_moves

    def act(self, board, possible_moves):
        if len(possible_moves) == 0: 
            return None

        enemy_captures, sacrifice_moves, self_captures, quiet_moves, end_moves = \
            self.filter_sacrifice_moves(possible_moves, board)

        good_moves = []
        
        good_moves.extend(enemy_captures)

        if sacrifice_moves:
            if random.random() < self.sacrifice_probability:
                good_moves.extend(sacrifice_moves)
        
        good_moves.extend(quiet_moves)
        
        good_moves.extend(end_moves)
        
        if not good_moves:
            good_moves.extend(self_captures)
            good_moves.extend(end_moves)
        
        if not good_moves:
            return possible_moves[0]

        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
        
        if random.random() < self.epsilon:
            return random.choice(good_moves)

        state = self.get_state(board).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state).squeeze(0)
        self.model.train()

        mask = torch.full((4096,), -1e9, device=self.device)
        for move in good_moves:
            mask[self.encode_move(move)] = 0

        best_move_index = (q_values + mask).argmax().item()
        
        for move in good_moves:
            if self.encode_move(move) == best_move_index:
                return move
        return good_moves[0]

    def remember(self, state, action, reward, next_state, is_done):
        self.memory.append((state, action, reward, next_state, is_done))
        if len(self.memory) > 100000:
            self.memory.pop(0) 

    def train(self, iterations=3):
        for _ in range(iterations):
            self._train_once()
    
    def _train_once(self):
        batch_size = 64
        if len(self.memory) < batch_size: 
            return
        
        batch = random.sample(self.memory, batch_size)
        
        states_np = np.array([item[0] for item in batch], dtype=np.float32)
        actions_np = np.array([item[1] for item in batch], dtype=np.int64)
        rewards_np = np.array([item[2] for item in batch], dtype=np.float32)
        next_states_np = np.array([item[3] for item in batch], dtype=np.float32)
        dones_np = np.array([item[4] for item in batch], dtype=np.float32)
        
        states = torch.from_numpy(states_np).to(self.device)
        actions = torch.from_numpy(actions_np).to(self.device)
        rewards = torch.from_numpy(rewards_np).to(self.device)
        next_states = torch.from_numpy(next_states_np).to(self.device)
        dones = torch.from_numpy(dones_np).to(self.device)
        
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        with torch.no_grad():
            next_max_q = self.model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_max_q
        
        loss = self.loss_function(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.train_step += 1

    def save(self, path="bot_model.pth"):
        torch.save({
            'model': self.model.state_dict(), 
            'epsilon': self.epsilon,
            'train_step': self.train_step,
            'sacrifice_probability': self.sacrifice_probability
        }, path)
        print(f"Модель сохранена: {path}")

    def load(self, path="bot_model.pth"):
        if not os.path.exists(path): 
            return False
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model'])
            self.epsilon = checkpoint.get('epsilon', 0.1)
            self.train_step = checkpoint.get('train_step', 0)
            self.sacrifice_probability = checkpoint.get('sacrifice_probability', 0.3)
            print(f"Модель загружена: {path}, epsilon: {self.epsilon:.4f}")
            return True
        except:
            return False