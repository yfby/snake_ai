import random
import time
import copy
from pynput import keyboard

direction = 'up'

class Snake:
    def __init__(self, x, y, direction):
        self.parts = [[3, 3, 0]]
        self.direction = direction

    def update(self, direction):
        last_direction = self.direction
        self.direction = direction

        last_parts = copy.deepcopy(self.parts)

        for i in range(len(self.parts)):
            if i != 0:
                self.parts[i][0] = last_parts[i-1][0]
                self.parts[i][1] = last_parts[i-1][1]

            if i == 0:
                if self.direction == "up":
                    if last_direction == "down":
                        self.direction = last_direction
                        self.parts[i][1] = self.parts[i][1] + 1
                    else:
                        self.parts[i][1] = self.parts[i][1] - 1
                elif self.direction == "down":
                    if last_direction == "up":
                        self.direction = last_direction
                        self.parts[i][1] = self.parts[i][1] - 1
                    else:
                        self.parts[i][1] = self.parts[i][1] + 1
                elif self.direction == "left":
                    if last_direction == "right":
                        self.direction = last_direction
                        self.parts[i][0] = self.parts[i][0] + 1
                    else:
                        self.parts[i][0] = self.parts[i][0] - 1
                elif self.direction == "right":
                    if last_direction == "left":
                        self.direction = last_direction
                        self.parts[i][0] = self.parts[i][0] - 1
                    else:
                        self.parts[i][0] = self.parts[i][0] + 1

    def grow(self):
        self.parts.append(copy.deepcopy(self.parts[-1]))

class Game:
    def __init__(self, snake, x_width, y_height):
        self.diemension = [x_width, y_height]
        self.snake = snake
        self.food = self.generate_food()
        self.alive = True

    def update(self, direction):
        if self.alive == False:
            return

        self.snake.update(direction)
        self.check_collision()

    def check_collision(self): # Check Snake Collisions
        #Check if hitted wall
        if self.snake.parts[0][0] >= self.diemension[0] or self.snake.parts[0][0] < 0 or self.snake.parts[0][1] >= self.diemension[1] or self.snake.parts[0][1] < 0:
            self.alive = False

        # Check if eaten food
        if self.snake.parts[0][0] == self.food[0] and self.snake.parts[0][1] == self.food[1]:
            self.snake.grow()

            if len(self.snake.parts) == (self.diemension[0]*self.diemension[1]):
                self.alive = False

            self.food = self.generate_food()

        # Check if snake collided with it self
        for snake_position in range(len(self.snake.parts)):
            if self.snake.parts[0][0] == self.snake.parts[snake_position][0] and self.snake.parts[0][1] == self.snake.parts[snake_position][1] and snake_position > 1:
                self.alive = False

    def generate_food(self): # Generate Food
        food_position = [random.randrange(self.diemension[0]), random.randrange(self.diemension[1])]
        for snake_position in self.snake.parts:
            if snake_position[0] == food_position[0] and snake_position[1] == food_position[1]:
                return self.generate_food()
        return food_position

    def display(self):
        if self.alive == False:
            print("Game Over!")
            return

        grid = []
        # Filling in the blank grid with '.'
        for i in range(self.diemension[0]*self.diemension[1]):
            grid.append('.')
        
        # Placing snake in the grid
        for snake_place in self.snake.parts:
            grid[(snake_place[1]*self.diemension[1])+snake_place[0]] = "#"

        # Place Food
        grid[(self.food[1]*self.diemension[1])+self.food[0]] = "@"

        # Print the grids
        print("\n" * 32)
        for grid_operation in range(int(len(grid)/self.diemension[1])):
            print(''.join(grid[(grid_operation*self.diemension[0]):(grid_operation+1)*self.diemension[0]]))


snake = Snake(4, 4, direction)
game = Game(snake, 6, 6)

def on_press(key):
    global direction
    try:
        if key == keyboard.Key.up: 
            direction = 'up'
        if key == keyboard.Key.down:
            direction = 'down'
        if key == keyboard.Key.left:
            direction = 'left'
        if key == keyboard.Key.right:
            direction = 'right'
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=on_press)
listener.start()

while game.alive:
    time.sleep(0.2)
    game.update(direction)
    game.display()
