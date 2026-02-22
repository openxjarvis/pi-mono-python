
import random

BOARD_WIDTH = 20
BOARD_HEIGHT = 10

class SnakeGame:
    def __init__(self):
        self.snake = [(5, 5)]  # 蛇的初始位置
        self.food = self.create_food()
        self.board = [[' ' for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]
        self.direction = 'RIGHT' # 初始方向
        self.game_over = False

    def create_food(self):
        while True:
            food = (random.randint(0, BOARD_HEIGHT - 1), random.randint(0, BOARD_WIDTH - 1))
            if food not in self.snake:
                return food

    def update_board(self):
        self.board = [[' ' for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]
        for row, col in self.snake:
            self.board[row][col] = 'S'
        self.board[self.food[0]][self.food[1]] = 'F'

    def move(self):
        head_row, head_col = self.snake[0]
        if self.direction == 'RIGHT':
            new_head = (head_row, head_col + 1)
        elif self.direction == 'LEFT':
            new_head = (head_row, head_col - 1)
        elif self.direction == 'UP':
            new_head = (head_row - 1, head_col)
        elif self.direction == 'DOWN':
            new_head = (head_row + 1, head_col)

        if (new_head[0] < 0 or new_head[0] >= BOARD_HEIGHT or
            new_head[1] < 0 or new_head[1] >= BOARD_WIDTH or
            new_head in self.snake):
            self.game_over = True
            return

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.food = self.create_food()
        else:
            self.snake.pop()

    def change_direction(self, new_direction):
        if new_direction == 'RIGHT' and self.direction != 'LEFT':
            self.direction = 'RIGHT'
        elif new_direction == 'LEFT' and self.direction != 'RIGHT':
            self.direction = 'LEFT'
        elif new_direction == 'UP' and self.direction != 'DOWN':
            self.direction = 'UP'
        elif new_direction == 'DOWN' and self.direction != 'UP':
            self.direction = 'DOWN'

    def display_board(self):
        for row in self.board:
            print(''.join(row))
        print("-" * BOARD_WIDTH)

    def play(self):
        while not self.game_over:
            self.update_board()
            self.display_board()

            move = input("Enter move (UP, DOWN, LEFT, RIGHT, or QUIT): ").upper()
            if move == "QUIT":
                break
            if move in ("UP", "DOWN", "LEFT", "RIGHT"):
                self.change_direction(move)

            self.move()

        if self.game_over:
            print("Game Over!")
        else:
            print("Quitting game.")

if __name__ == '__main__':
    game = SnakeGame()
    game.play()
