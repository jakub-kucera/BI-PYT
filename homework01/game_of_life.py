"""
Homework 01 - Game of life.

Your task is to implement a kind of cellular automaton called "Game of life".
The automaton is a 2D simulation where each cell on the grid is either dead
or alive.

The state of each cell is updated in every iteration based state of neighbouring cells.
Cell neighbours are cells that are horizontally, vertically, or diagonally adjacent.

Rules for the update are as follows:

1. Any live cell with fewer than two live neighbours dies, as if by underpopulation.
2. Any live cell with two or three live neighbours lives on to the next generation.
3. Any live cell with more than three live neighbours dies, as if by overpopulation.
4. Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.


Our implementation uses the coordinate system with grid coordinates starting
from (0, 0) - upper left corner. The first coordinate is a row, and the second
is a column.

Do not use wrap-around (toroid) when reaching the edge of the board.

For more details about Game of Life, see Wikipedia:
https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life
"""


def update(alive: set, size: (int, int), iter_n: int) -> set:
    """
    Perform iter_n iterations.

    Args
    ----
        alive (set):
            A set of cell coordinates marked as alive, can be empty.
        size (int, int):
            The size of simulation grid as a tuple of two ints.
        iter_n (int):
            A number of iterations to perform.

    Returns
    -------
        _  (set):
            A set of coordinates of alive cells after iter_n iterations.
    """
    # TODO: Implement update rules.

    for _ in range(iter_n):

        new_alive = set()

        # count neighbors
        for y in range(size[0]):
            for x in range(size[1]):
                alive_neighbors = 0
                if (y + 1, x) in alive:
                    alive_neighbors += 1
                if (y - 1, x) in alive:
                    alive_neighbors += 1
                if (y, x + 1) in alive:
                    alive_neighbors += 1
                if (y, x - 1) in alive:
                    alive_neighbors += 1
                if (y + 1, x + 1) in alive:
                    alive_neighbors += 1
                if (y + 1, x - 1) in alive:
                    alive_neighbors += 1
                if (y - 1, x + 1) in alive:
                    alive_neighbors += 1
                if (y - 1, x - 1) in alive:
                    alive_neighbors += 1

                if alive_neighbors == 3 or (alive_neighbors == 2 and (y, x) in alive):
                    new_alive.add((y, x))

        alive = new_alive #todo check if copy


    return alive


def draw(alive: set, size: (int, int)) -> str:
    """
    Draw a game board.

    Args
    ----
        alive (set):
            A set of cell coordinates marked as alive, can be empty.
        size (int, int):
            The size of simulation grid as a tuple of two ints.

    Returns
    -------
        _  (string):
           A string showing the board state with alive cells marked with X.
    """
    # TODO: implement board drawing logic and return it as output
    # Don't call print in this method, just return board string as output.
    # Example of 3x3 board with 1 alive cell at coordinates (0, 2):
    # +---+
    # |  X|
    # |   |
    # |   |
    # +---+
    drawn_map = "+" + size[1]*"-" + "+\n"

    for y in range(size[0]):
        drawn_map += "|"
        for x in range(size[1]):
            if (y, x) in alive:
                drawn_map += "X"
            else:
                drawn_map += " "
        drawn_map += "|\n"

    drawn_map += "+" + size[1]*"-" + "+"


    return drawn_map
