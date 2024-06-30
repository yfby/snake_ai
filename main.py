from game import Game
import time
from pynput import keyboard

snake_game = Game(4, 4)

direction = 1
def on_press(key):
    global direction
    try:
        if key == keyboard.Key.up: 
            direction = 1
        if key == keyboard.Key.down:
            direction = 2
        if key == keyboard.Key.left:
            direction = 3
        if key == keyboard.Key.right:
            direction = 4
        if key == keyboard.Key.esc:
            snake_game.new_game()
    except AttributeError:
        pass
    snake_game.update(direction)
    snake_game.display()

listener = keyboard.Listener(on_press=on_press)
listener.start()

while True:
    time.sleep(0.2)

'''
direction
1 up
2 down
3 left
4 right
'''
