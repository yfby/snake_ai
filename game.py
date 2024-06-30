from random import randrange 
from copy import deepcopy

class Game:
    def __init__(self, x_width, y_height):
        self.diemension = [x_width, y_height]
        self.new_game()

    def new_game(self):
        self.snake = [[int(self.diemension[0]/2), int(self.diemension[1]/2), 0]]
        self.direction = 1
        self.food = self.generate_food()
        self.alive = True
        self.score = 0

    def generate_food(self): # Generate Food
        food_position = [randrange(self.diemension[0]), randrange(self.diemension[1])]
        for snake_position in self.snake:
            if snake_position[0] == food_position[0] and snake_position[1] == food_position[1]:
                return self.generate_food()
        return food_position


    def move(self, direction):
        last_direction = self.direction
        self.direction = direction

        last_parts = deepcopy(self.snake)

        for i in range(len(self.snake)):
            if i != 0:
                self.snake[i][0] = last_parts[i-1][0]
                self.snake[i][1] = last_parts[i-1][1]
            
            # 0 is the head
            if i == 0:
                '''
                direction
                1 up
                2 down
                3 left
                4 right
                '''             
                if self.direction == 1:
                    if last_direction == 2:
                        self.direction = last_direction
                        self.snake[i][1] = self.snake[i][1] + 1
                    else:
                        self.snake[i][1] = self.snake[i][1] - 1
                elif self.direction == 2:
                    if last_direction == 1:
                        self.direction = last_direction
                        self.snake[i][1] = self.snake[i][1] - 1
                    else:
                        self.snake[i][1] = self.snake[i][1] + 1
                elif self.direction == 3:
                    if last_direction == 4:
                        self.direction = last_direction
                        self.snake[i][0] = self.snake[i][0] + 1
                    else:
                        self.snake[i][0] = self.snake[i][0] - 1
                elif self.direction == 4:
                    if last_direction == 3:
                        self.direction = last_direction
                        self.snake[i][0] = self.snake[i][0] - 1
                    else:
                        self.snake[i][0] = self.snake[i][0] + 1

    def grow(self):
        self.snake.append(deepcopy(self.snake[-1]))

    def update(self, direction):
        reward = 0
        if self.alive == False:
            return

        # Move snake
        self.move(direction)

        #Check for collisions
        #Check if hitted wall
        if self.snake[0][0] >= self.diemension[0] or self.snake[0][0] < 0 or self.snake[0][1] >= self.diemension[1] or self.snake[0][1] < 0:
            self.alive = False
            reward = -10

        # Check if eaten food
        if self.snake[0][0] == self.food[0] and self.snake[0][1] == self.food[1]:
            # Add block to snake
            self.grow()
            reward = 10
            self.score += 1

            # Check if won the game
            if len(self.snake) == (self.diemension[0]*self.diemension[1]):
                self.alive = False
                reward = 100

            # Make food
            self.food = self.generate_food()

        # Check if snake collided with it self
        for snake_position in range(len(self.snake)):
            if self.snake[0][0] == self.snake[snake_position][0] and self.snake[0][1] == self.snake[snake_position][1] and snake_position > 1:
                self.alive = False
                reward = -10

        print(reward)
        return reward, self.score, not self.alive


    def display(self):
        if self.alive == False:
            print("Game Over!")
            return

        grid = []
        # Filling in the blank grid with '.'
        for i in range(self.diemension[0]*self.diemension[1]):
            grid.append('.')
        
        # Placing snake in the grid
        for snake_place in self.snake:
            grid[(snake_place[1]*self.diemension[1])+snake_place[0]] = "#"

        # Place Food
        grid[(self.food[1]*self.diemension[1])+self.food[0]] = "@"

        # Print the grids
        print("\n" * 32)
        for grid_operation in range(int(len(grid)/self.diemension[1])):
            print(''.join(grid[(grid_operation*self.diemension[0]):(grid_operation+1)*self.diemension[0]]))
        print(self.score)
