import pygame

COLOR_TEXT_CREAM = (245, 240, 225) 
COLOR_TEXT_GOLD = (240, 195, 20)  
COLOR_TEXT_HEADER = (210, 60, 60)  
COLOR_BTN_BG = (100, 20, 20)       
COLOR_BTN_TEXT = (240, 195, 20)  
COLOR_BTN_BORDER = (165, 120, 85)  
COLOR_INPUT_BG = (50, 45, 40)      
COLOR_INPUT_TEXT = (245, 240, 225) 
COLOR_WHITE_PIECE = (245, 245, 245)
COLOR_BLACK_PIECE = (20, 20, 20)
COLOR_GOLD_PIECE = (240, 195, 20) 
COLOR_PANEL_BG = (50, 45, 40)      
COLOR_PANEL_TEXT = (245, 240, 225) 
COLOR_GOLD = (240, 195, 20)

class UI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Шашки – Самоеды")

        self.font_normal = pygame.font.SysFont("Georgia", 22)
        self.font_big = pygame.font.SysFont("Georgia", 36)

        self.bg_image = pygame.image.load("gg.jpg").convert()
        self.bg_image = pygame.transform.scale(self.bg_image, (800, 600))

        self.board_image = pygame.image.load("gg2.jpg").convert()
        self.board_image = pygame.transform.scale(self.board_image, (600, 600))

    def draw_button(self, text, x, y, width=200, height=50):
        rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, COLOR_BTN_BG, rect)
        pygame.draw.rect(self.screen, COLOR_BTN_BORDER, rect, 2)
        
        text_surface = self.font_normal.render(text, True, COLOR_BTN_TEXT)
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)
        return rect

    def draw_text(self, text, x, y, color=COLOR_TEXT_CREAM, center=False, is_big=False):
        font = self.font_big if is_big else self.font_normal
        text_surface = font.render(text, True, color)
        
        if center:
            text_rect = text_surface.get_rect(center=(x, y))
            self.screen.blit(text_surface, text_rect)
        else:
            self.screen.blit(text_surface, (x, y))

    def show_menu(self, username=None):
        self.screen.blit(self.bg_image, (0, 0))
        self.draw_text("Шашки – Самоеды", 400, 100, color=COLOR_TEXT_HEADER, is_big=True, center=True)
        
        if username:
            self.draw_text(f"Игрок: {username}", 400, 160, COLOR_TEXT_HEADER, center=True)
        else:
            self.draw_text("Войдите в аккаунт", 400, 160, COLOR_TEXT_CREAM, center=True)
        
        btn_play = self.draw_button("Играть", 300, 240)
        btn_auth = self.draw_button("Вход/Регистрация", 300, 310)
        btn_exit = self.draw_button("Выход", 300, 380)
        
        pygame.display.flip()
        return btn_play, btn_auth, btn_exit

    def show_auth_screen(self, login_text, password_text, active_field, message=""):
        self.screen.blit(self.bg_image, (0, 0))
        self.draw_text("Вход / Регистрация", 400, 80, color=COLOR_TEXT_HEADER, is_big=True, center=True)
        self.draw_text("Логин:", 200, 200, color=COLOR_TEXT_CREAM)
        self.draw_text("Пароль:", 200, 270, color=COLOR_TEXT_CREAM)
        
        rect_login = pygame.Rect(320, 195, 250, 40)
        pygame.draw.rect(self.screen, COLOR_INPUT_BG, rect_login)
        border_color = COLOR_TEXT_GOLD if active_field == "login" else COLOR_TEXT_CREAM
        pygame.draw.rect(self.screen, border_color, rect_login, 2)
        self.screen.blit(self.font_normal.render(login_text, True, COLOR_INPUT_TEXT), (330, 205))
        
        rect_password = pygame.Rect(320, 265, 250, 40)
        pygame.draw.rect(self.screen, COLOR_INPUT_BG, rect_password)
        border_color = COLOR_TEXT_GOLD if active_field == "pass" else COLOR_TEXT_CREAM
        pygame.draw.rect(self.screen, border_color, rect_password, 2)
        hidden_password = "*" * len(password_text)
        self.screen.blit(self.font_normal.render(hidden_password, True, COLOR_INPUT_TEXT), (330, 275))
        
        btn_login = self.draw_button("Войти", 230, 350, 160, 45)
        btn_register = self.draw_button("Регистрация", 410, 350, 160, 45)
        btn_back = self.draw_button("Назад", 320, 420, 160, 45)
        
        if message:
            self.draw_text(message, 400, 510, COLOR_TEXT_HEADER, center=True)
            
        pygame.display.flip()
        return rect_login, rect_password, btn_login, btn_register, btn_back

    def show_mode_select(self):
        self.screen.blit(self.bg_image, (0, 0))
        self.draw_text("Режим игры", 400, 120, color=COLOR_TEXT_HEADER, is_big=True, center=True)
        
        btn_pve = self.draw_button("Человек vs ИИ", 250, 200, 300, 50)
        btn_bots = self.draw_button("ИИ vs Минимакс", 250, 270, 300, 50)
        btn_train = self.draw_button("Обучение ИИ", 250, 340, 300, 50)
        btn_back = self.draw_button("Назад", 320, 430, 160, 50)
        
        pygame.display.flip()
        return btn_pve, btn_bots, btn_train, btn_back

    def draw_game_board(self, board, current_turn, mode, selected_piece=None, winner=None, stats=None):
        self.screen.fill((40, 35, 30))
        
        if self.board_image:
            self.screen.blit(self.board_image, (0, 0))
            
        square_size = 600 // 8

        for row in range(8):
            for col in range(8):
                rect = pygame.Rect(col * square_size, row * square_size, square_size, square_size)

                if selected_piece == (row, col):
                    pygame.draw.rect(self.screen, COLOR_TEXT_GOLD, rect, 3)

                piece = board.get_piece(row, col)
                if piece != 0:
                    piece_color = COLOR_WHITE_PIECE if piece in [1, 3] else COLOR_BLACK_PIECE
                    center_x = col * square_size + square_size // 2
                    center_y = row * square_size + square_size // 2
                    radius = square_size // 2 - 8
                    
                    pygame.draw.circle(self.screen, piece_color, (center_x, center_y), radius)
                    border_color = (60, 60, 60) if piece_color == COLOR_WHITE_PIECE else COLOR_WHITE_PIECE
                    pygame.draw.circle(self.screen, border_color, (center_x, center_y), radius, 1)

                    if piece in [3, 4]:
                        pygame.draw.circle(self.screen, COLOR_GOLD_PIECE, (center_x, center_y), radius // 2, 3)
 
        panel_x = 615
        panel_rect = pygame.Rect(600, 0, 200, 600)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, panel_rect)
        
        self.draw_text("Ход:", panel_x, 40, color=COLOR_PANEL_TEXT)
        turn_text = "БЕЛЫЕ" if current_turn == 1 else "ЧЁРНЫЕ"
        self.draw_text(turn_text, panel_x, 70, color=COLOR_GOLD)
        
        mode_names = {1: "PvE", 2: "ИИ vs MM", 3: "Обучение"}
        self.draw_text(mode_names[mode], panel_x, 110, color=COLOR_PANEL_TEXT)
        
        if stats:
            y = 160
            self.draw_text(f"Эпизод: {stats['episode']}/{stats['total']}", panel_x, y, color=COLOR_PANEL_TEXT)
            y += 35
            self.draw_text(f"Побед: {stats['wins']}", panel_x, y, color=COLOR_PANEL_TEXT)
            y += 25
            self.draw_text(f"Пораж: {stats['losses']}", panel_x, y, color=COLOR_PANEL_TEXT)
            y += 25
            self.draw_text(f"Ничьи: {stats['draws']}", panel_x, y, color=COLOR_PANEL_TEXT)
            y += 35
            total_games = max(1, stats['wins'] + stats['losses'])
            win_rate = (stats['wins'] / total_games) * 100
            self.draw_text(f"WinRate: {win_rate:.1f}%", panel_x, y, color=COLOR_GOLD)
        
        if winner:
            self.draw_text("ИГРА ОКОНЧЕНА", panel_x, 400, color=COLOR_TEXT_HEADER)
            self.draw_text(f"Победили: {winner}", panel_x, 430, color=COLOR_GOLD)

        btn_menu = self.draw_button("В меню", 620, 530, 160, 40)
        pygame.display.flip()
        
        return btn_menu