import pygame
import sys
from board import Board
from game_logic import GameLogic
from bot import Bot          
from minimax import Minimax 
from ui import UI
import auth

STATE_MENU = 0
STATE_AUTH = 1
STATE_MODE = 2
STATE_GAME = 3

def main():
    ui = UI()
    game_board = Board()
    logic = GameLogic()
    
    bot = Bot(player=2)
    bot.load("bot_model.pth")
    bot.epsilon = 0.3
    bot.eps_min = 0.05
    bot.eps_decay = 0.9995
    
    minimax = Minimax(player_num=1, search_depth=1)
    
    current_state = STATE_MENU
    current_user = None
    
    login_str = ""
    password_str = ""
    active_input_field = "login"
    auth_message = ""
    
    game_mode = 1  
    winner = None
    clock = pygame.time.Clock()

    train_episode = 0
    train_total_episodes = 50000
    train_wins = 0
    train_losses = 0
    train_draws = 0
    best_win_rate = 0
    
    pending_state = None
    pending_action = None
    ate_own_last_turn = False

    while True:
        
        # МЕНЮ
        if current_state == STATE_MENU:
            btn_play, btn_auth, btn_exit = ui.show_menu(current_user)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    quit_game()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if btn_exit.collidepoint(event.pos): 
                        quit_game()
                    if btn_auth.collidepoint(event.pos): 
                        current_state = STATE_AUTH
                        auth_message = ""
                    if btn_play.collidepoint(event.pos) and current_user is not None: 
                        current_state = STATE_MODE

        # АВТОРИЗАЦИЯ
        elif current_state == STATE_AUTH:
            rect_log, rect_pass, btn_log, btn_reg, btn_back = ui.show_auth_screen(
                login_str, password_str, active_input_field, auth_message)
                
            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    quit_game()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if rect_log.collidepoint(event.pos): 
                        active_input_field = "login"
                    elif rect_pass.collidepoint(event.pos): 
                        active_input_field = "pass"
                    elif btn_back.collidepoint(event.pos): 
                        current_state = STATE_MENU
                    elif btn_log.collidepoint(event.pos):
                        success, auth_message = auth.login(login_str, password_str)
                        if success: 
                            current_user = login_str
                            current_state = STATE_MENU
                    elif btn_reg.collidepoint(event.pos):
                        success, auth_message = auth.register(login_str, password_str)
                        
                if event.type == pygame.KEYDOWN:
                    if active_input_field == "login":
                        if event.key == pygame.K_BACKSPACE:
                            login_str = login_str[:-1]
                        elif event.unicode.isalnum():
                            login_str += event.unicode
                    else:
                        if event.key == pygame.K_BACKSPACE:
                            password_str = password_str[:-1]
                        elif event.unicode.isalnum():
                            password_str += event.unicode

        # ВЫБОР РЕЖИМА
        elif current_state == STATE_MODE:
            btn_pve, btn_bots, btn_train, btn_back = ui.show_mode_select()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    quit_game()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if btn_back.collidepoint(event.pos): 
                        current_state = STATE_MENU
                    
                    if btn_pve.collidepoint(event.pos):
                        game_mode, current_state = 1, STATE_GAME
                        game_board.reset()
                        logic = GameLogic()
                        winner = None
                        bot.load("bot_model.pth")
                        bot.epsilon = 0.05
                        ate_own_last_turn = False
                        
                    if btn_bots.collidepoint(event.pos):
                        game_mode, current_state = 2, STATE_GAME
                        game_board.reset()
                        logic = GameLogic()
                        winner = None
                        bot.load("bot_model.pth")
                        bot.epsilon = 0.05
                        ate_own_last_turn = False
                        
                    if btn_train.collidepoint(event.pos):
                        game_mode, current_state = 3, STATE_GAME
                        game_board.reset()
                        logic = GameLogic()
                        winner = None
                        train_episode = 0
                        train_wins = train_losses = train_draws = 0
                        pending_state = None
                        pending_action = None
                        ate_own_last_turn = False
                        print(">>> Обучение начато: против depth=1")

        # ИГРА
        elif current_state == STATE_GAME:
            stats = None
            if game_mode == 3:
                stats = {
                    'episode': train_episode,
                    'total': train_total_episodes,
                    'wins': train_wins,
                    'losses': train_losses,
                    'draws': train_draws,
                    'epsilon': bot.epsilon
                }
            
            btn_menu = ui.draw_game_board(game_board, logic.current_turn, game_mode, logic.selected_piece, winner, stats)
            possible_moves = logic.get_all_legal_moves(game_board, logic.current_turn, logic.capture_chain_pos)
            
            # КОНЕЦ ИГРЫ
            if len(possible_moves) == 0 and winner is None:
                if logic.current_turn == 2:
                    winner = "БЕЛЫЕ"
                else:
                    winner = "ЧЁРНЫЕ"
                
                if pending_state is not None and pending_action is not None:
                    final_reward = 1000 if winner == "ЧЁРНЫЕ" else -1000
                    state_after = bot.get_state(game_board).cpu().numpy()
                    bot.remember(pending_state, pending_action, final_reward, state_after, True)
                    for _ in range(5):
                        bot.train()
                    pending_state = None
                    pending_action = None
                
                if game_mode == 3:
                    if winner == "ЧЁРНЫЕ":
                        train_wins += 1
                    elif winner == "БЕЛЫЕ":
                        train_losses += 1
                    else:
                        train_draws += 1
                    
                    train_episode += 1
                    
                    total_played = train_wins + train_losses
                    if total_played > 0:
                        win_rate = train_wins / total_played
                        if win_rate > best_win_rate:
                            best_win_rate = win_rate
                            bot.save("bot_model.pth")
                    
                    if train_episode % 100 == 0:
                        bot.save("bot_model.pth")
                        total = train_wins + train_losses
                        wr = train_wins / total * 100 if total > 0 else 0
                        print(f"Эпизод {train_episode}: W={train_wins} L={train_losses} WR={wr:.1f}% ε={bot.epsilon:.3f}")
                    
                    if train_episode < train_total_episodes:
                        game_board.reset()
                        logic = GameLogic()
                        winner = None
                        pending_state = None
                        pending_action = None
                        ate_own_last_turn = False
                    else:
                        bot.save("bot_model.pth")
                        winner = "ОБУЧЕНИЕ ЗАВЕРШЕНО"
                        print(f"\nФинальный WinRate: {best_win_rate:.1%}")

            # ХОДЫ ИИ
            if winner is None:
                if game_mode == 1 and logic.current_turn == 2:  
                    pygame.time.wait(150)
                    ate_own_last_turn = perform_bot_move(bot, game_board, logic, possible_moves, 2, ate_own_last_turn)
                    
                elif game_mode == 2 and logic.current_turn == 1:  
                    pygame.time.wait(250)
                    best_move = minimax.act(game_board, logic)
                    if best_move: 
                        logic.make_move(game_board, best_move, 1)
                    
                elif game_mode == 2 and logic.current_turn == 2: 
                    pygame.time.wait(250)
                    ate_own_last_turn = perform_bot_move(bot, game_board, logic, possible_moves, 2, ate_own_last_turn)
                    
                elif game_mode == 3: 
                    if logic.current_turn == 1:
                        best_move = minimax.act(game_board, logic)
                        if best_move: 
                            logic.make_move(game_board, best_move, 1)
                        
                        next_moves = logic.get_all_legal_moves(game_board, logic.current_turn, logic.capture_chain_pos)
                        if len(next_moves) == 0:
                            if pending_state is not None and pending_action is not None:
                                if logic.current_turn == 2:
                                    state_after = bot.get_state(game_board).cpu().numpy()
                                    bot.remember(pending_state, pending_action, -1000, state_after, True)
                                    for _ in range(5):
                                        bot.train()
                                    pending_state = None
                                    pending_action = None
                    else:
                        state_before = bot.get_state(game_board).cpu().numpy()
                        chosen_move = bot.act(game_board, possible_moves)
                        
                        if chosen_move:
                            action_encoded = bot.encode_move(chosen_move)
                            logic.make_move(game_board, chosen_move, 2)
                            
                            # ВЫЗЫВАЕМ ОБЩУЮ ФУНКЦИЮ НАГРАД
                            reward, ate_own_last_turn = calculate_reward(
                                chosen_move, possible_moves, game_board, logic, ate_own_last_turn
                            )
                            
                            next_moves = logic.get_all_legal_moves(game_board, logic.current_turn, logic.capture_chain_pos)
                            is_done = len(next_moves) == 0
                            
                            state_after = bot.get_state(game_board).cpu().numpy()
                            bot.remember(state_before, action_encoded, reward, state_after, is_done)
                            for _ in range(5):
                                bot.train()
                            
                            if is_done:
                                pending_state = None
                                pending_action = None
                            else:
                                pending_state = state_after
                                pending_action = action_encoded

            # КЛИКИ
            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    quit_game()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if btn_menu.collidepoint(event.pos):
                        current_state = STATE_MENU
                        if game_mode == 3:
                            bot.save("bot_model.pth")
                    
                    if game_mode == 1 and logic.current_turn == 1 and winner is None:
                        mouse_x, mouse_y = event.pos
                        if mouse_x < 600 and mouse_y < 600:
                            square_size = 600 // 8
                            clicked_col = mouse_x // square_size
                            clicked_row = mouse_y // square_size
                            
                            if logic.selected_piece:
                                matched_move = find_move_in_list(possible_moves, logic.selected_piece, clicked_row, clicked_col)
                                if matched_move:
                                    logic.make_move(game_board, matched_move, 1)
                                    ate_own_last_turn = False
                                elif logic.capture_chain_pos is None and logic.is_my_piece(game_board.get_piece(clicked_row, clicked_col), 1):
                                    logic.selected_piece = (clicked_row, clicked_col)
                            else:
                                if logic.is_my_piece(game_board.get_piece(clicked_row, clicked_col), 1):
                                    can_move = False
                                    for m in possible_moves:
                                        if m['from'] == (clicked_row, clicked_col):
                                            can_move = True
                                            break
                                    if can_move:
                                        logic.selected_piece = (clicked_row, clicked_col)


def calculate_reward(move, moves, board, logic, ate_own_last_turn):
    reward = 10
    ate_own_this_turn = False
    
    if len(move.get('captured', [])) > 0:
        if not move.get('is_own_piece'): 
            reward += 100  

            if ate_own_last_turn:
                reward += 250
        else:
            forced = True
            for m in moves:
                if len(m.get('captured', [])) > 0 and not m.get('is_own_piece', False):
                    forced = False
            reward -= 10 if forced else 200
            ate_own_this_turn = True
    
    to_row, to_col = move['to']
    if board.get_piece(to_row, to_col) in [3, 4]: 
        reward += 80
    
    if len(move.get('captured', [])) > 1:
        reward += 20
    
    next_moves = logic.get_all_legal_moves(board, logic.current_turn, logic.capture_chain_pos)
    is_done = len(next_moves) == 0
    
    if is_done:
        winner = "ЧЁРНЫЕ" if logic.current_turn == 1 else "БЕЛЫЕ"
        reward += 1000 if winner == "ЧЁРНЫЕ" else -1000
    
    return reward, ate_own_this_turn


def perform_bot_move(agent, board, logic, moves, player, ate_own_last_turn=False):
    state_before = agent.get_state(board).cpu().numpy()
    chosen_move = agent.act(board, moves)
    
    if not chosen_move: 
        return ate_own_last_turn
    
    action_encoded = agent.encode_move(chosen_move)
    logic.make_move(board, chosen_move, player)

    reward, ate_own_this_turn = calculate_reward(
        chosen_move, moves, board, logic, ate_own_last_turn
    )
    
    next_moves = logic.get_all_legal_moves(board, logic.current_turn, logic.capture_chain_pos)
    is_done = len(next_moves) == 0
    
    state_after = agent.get_state(board).cpu().numpy()
    agent.remember(state_before, action_encoded, reward, state_after, is_done)
    for _ in range(5):
        agent.train()
    
    return ate_own_this_turn


def find_move_in_list(moves, from_position, to_row, to_col):
    for move in moves:
        if move.get('end') and (to_row, to_col) == from_position: 
            return move
        if move['from'] == from_position and move['to'] == (to_row, to_col): 
            return move
    return None

def quit_game():
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()