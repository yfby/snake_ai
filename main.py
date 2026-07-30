from game import Direction, Snake


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
