
import random

class SnakeGame:
    def __init__(self, width=20, height=10):
        self.width = width
        self.height = height
        self.snake = [(width // 2, height // 2)]
        self.food = self.create_food()
        self.direction = "right"
        self.game_over = False

    def create_food(self):
        while True:
            food = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            if food not in self.snake:
                return food

    def move(self):
        head_x, head_y = self.snake[0]
        if self.direction == "right":
            new_head = (head_x + 1, head_y)
        elif self.direction == "left":
            new_head = (head_x - 1, head_y)
        elif self.direction == "up":
            new_head = (head_x, head_y - 1)
        elif self.direction == "down":
            new_head = (head_x, head_y + 1)

        if (
            new_head[0] < 0
            or new_head[0] >= self.width
            or new_head[1] < 0
            or new_head[1] >= self.height
            or new_head in self.snake
        ):
            self.game_over = True
            return

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.food = self.create_food()
        else:
            self.snake.pop()

    def change_direction(self, direction):
        if direction == "right" and self.direction != "left":
            self.direction = "right"
        elif direction == "left" and self.direction != "right":
            self.direction = "left"
        elif direction == "up" and self.direction != "down":
            self.direction = "up"
        elif direction == "down" and self.direction != "up":
            self.direction = "down"

    def display(self):
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) == self.food:
                    print("F", end="")
                elif (x, y) in self.snake:
                    print("S", end="")
                else:
                    print(".", end="")
            print()
        print("-" * self.width)  # Separator line
        print(f"Direction: {self.direction}, Score: {len(self.snake)}")

    def play(self):
        while not self.game_over:
            self.display()
            direction = input("Enter direction (up, down, left, right, or quit): ").lower()
            if direction == "quit":
                break
            self.change_direction(direction)
            self.move()

        print("Game Over!")

if __name__ == "__main__":
    game = SnakeGame()
    game.play()
