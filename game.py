from random import randrange
from copy import deepcopy
from enum import Enum
from typing import List, Tuple


class Direction(Enum):
    """Direction enum to replace magic numbers."""
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class Game:
    """Class representing the Snake game logic and state."""

    def __init__(self, x_width: int, y_height: int) -> None:
        """Initialize the Game with given dimensions.

        Args:
            x_width: Width of the game grid
            y_height: Height of the game grid
        """
        self.dimension = [x_width, y_height]
        self.snake = [[int(self.dimension[0]/2), int(self.dimension[1]/2), 0]]
        self.direction = Direction.UP
        self.food = self.generate_food()
        self.alive = True
        self.score = 0

    def new_game(self) -> None:
        """Reset the game to its initial state."""
        self.snake = [[int(self.dimension[0]/2), int(self.dimension[1]/2), 0]]
        self.direction = Direction.UP
        self.food = self.generate_food()
        self.alive = True
        self.score = 0

    def generate_food(self) -> List[int]:
        """Generate food at a random position not occupied by the snake.

        Uses an iterative approach to avoid stack overflow with large snakes.

        Returns:
            List of [x, y] coordinates for the food
        """
        occupied = {(s[0], s[1]) for s in self.snake}
        while True:
            food_position = [randrange(self.dimension[0]), randrange(self.dimension[1])]
            if tuple(food_position) not in occupied:
                return food_position


    def move(self, direction: Direction) -> None:
        """Move the snake in the given direction.

        Prevents the snake from reversing into itself by ignoring opposite directions.

        Args:
            direction: Direction enum value for movement
        """
        last_direction = self.direction
        self.direction = direction

        last_parts = deepcopy(self.snake)

        for i in range(len(self.snake)):
            if i != 0:
                self.snake[i][0] = last_parts[i-1][0]
                self.snake[i][1] = last_parts[i-1][1]

            # Move the head (index 0)
            if i == 0:
                if direction == Direction.UP:
                    # Prevent reversing into opposite direction
                    if last_direction != Direction.DOWN:
                        self.snake[i][1] = self.snake[i][1] - 1
                    else:
                        self.direction = last_direction
                        self.snake[i][1] = self.snake[i][1] + 1
                elif direction == Direction.DOWN:
                    if last_direction != Direction.UP:
                        self.snake[i][1] = self.snake[i][1] + 1
                    else:
                        self.direction = last_direction
                        self.snake[i][1] = self.snake[i][1] - 1
                elif direction == Direction.LEFT:
                    if last_direction != Direction.RIGHT:
                        self.snake[i][0] = self.snake[i][0] - 1
                    else:
                        self.direction = last_direction
                        self.snake[i][0] = self.snake[i][0] + 1
                elif direction == Direction.RIGHT:
                    if last_direction != Direction.LEFT:
                        self.snake[i][0] = self.snake[i][0] + 1
                    else:
                        self.direction = last_direction
                        self.snake[i][0] = self.snake[i][0] - 1

    def grow(self) -> None:
        """Add a new segment to the end of the snake."""
        self.snake.append(deepcopy(self.snake[-1]))

    def update(self, direction: Direction) -> Tuple[int, int, bool]:
        """Update the game state for one tick.

        Moves the snake, checks for collisions, handles food consumption, and self-collision.

        Args:
            direction: Direction enum for the snake to move

        Returns:
            Tuple of (reward, score, game_over)
        """
        reward = 0
        if not self.alive:
            return reward, self.score, not self.alive

        # Move snake
        self.move(direction)

        # Check for wall collisions
        if (self.snake[0][0] >= self.dimension[0] or self.snake[0][0] < 0 or
            self.snake[0][1] >= self.dimension[1] or self.snake[0][1] < 0):
            self.alive = False
            reward = -10

        # Check if eaten food
        if self.snake[0][0] == self.food[0] and self.snake[0][1] == self.food[1]:
            self.grow()
            reward = 10
            self.score += 1

            # Check if won the game (filled entire grid)
            if len(self.snake) == (self.dimension[0] * self.dimension[1]):
                self.alive = False
                reward = 100

            # Generate new food
            self.food = self.generate_food()

        # Check if snake collided with itself
        for snake_position in range(len(self.snake)):
            if (self.snake[0][0] == self.snake[snake_position][0] and 
                self.snake[0][1] == self.snake[snake_position][1] and 
                snake_position > 1):
                self.alive = False
                reward = -10

        return reward, self.score, not self.alive


    def display(self) -> None:
        """Display the current game state in the terminal.
 
        Uses a 2D grid representation for clarity and correctness.
        """
        if not self.alive:
            print("Game Over!")
            return

        # Create 2D grid
        grid = [['.' for _ in range(self.dimension[0])] for _ in range(self.dimension[1])]

        # Place snake segments
        for segment in self.snake:
            x, y = segment[0], segment[1]
            if 0 <= x < self.dimension[0] and 0 <= y < self.dimension[1]:
                grid[y][x] = "#"

        # Place food
        fx, fy = self.food
        if 0 <= fx < self.dimension[0] and 0 <= fy < self.dimension[1]:
            grid[fy][fx] = "@"

        # Clear screen and print grid
        print("\n" * 32)
        for row in grid:
            print(''.join(row))
        print(f"Score: {self.score}")
