import torch

from game import Direction, Snake

# 1. Check for Nvidia GPU (CUDA)
if torch.cuda.is_available():
    print(f"CUDA is available! 🎉")
    print(f"Total GPU Count: {torch.cuda.device_count()}")
    print(f"Current Device ID: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")


def main():
    game = Snake(16, 16)
    direction = Direction.UP

    while game.alive:
        game.display()
        key = input("Move (w/a/s/d, q=quit): ").strip().lower()

        if key == "w":
            direction = Direction.UP
        elif key == "s":
            direction = Direction.DOWN
        elif key == "a":
            direction = Direction.LEFT
        elif key == "d":
            direction = Direction.RIGHT
        elif key == "q":
            break
        else:
            continue

        game.tick(direction)

    game.display()
    print(f"Final Score: {game.score}")


if __name__ == "__main__":
    main()
