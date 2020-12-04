import imageio


def read_from_file(file_object):
    img = imageio.imread(file_object, pilmode="RGB")

    return img


def solve(board):
    """
    A method to solve the sudoku puzzle using the other functions defined.
    We use a simple recursion and backtracking method.

    Arguments:
    board - a list of nine sub lists with 9 numbers in each sub list

    Output:
    Returns True once the puzzle is successfully solved else False
    """
    find = findEmpty(board)

    if not find:
        return True
    else:
        row, col = find

    for i in range(1, 10):
        if valid(board, i, find):
            board[row][col] = i

            if solve(board):
                # print_board(board)
                return True

            board[row][col] = 0
    return False


# If this happens, the current board is the solution to the original Sudoku
def all_board_non_zero(matrix):
    for i in range(9):
        for j in range(9):
            if matrix[i][j] == 0:
                return False
    return True


def valid(bo, num, pos):
    # Check row
    for i in range(len(bo[0])):
        if bo[pos[0]][i] == num and pos[1] != i:
            return False
    # Check column
    for i in range(len(bo)):
        if bo[i][pos[1]] == num and pos[0] != i:
            return False
    # Check box
    box_x = pos[1] // 3
    box_y = pos[0] // 3
    for i in range(box_y * 3, box_y * 3 + 3):
        for j in range(box_x * 3, box_x * 3 + 3):
            if bo[i][j] == num and (i, j) != pos:
                return False
    return True


def findEmpty(board):
    """
    A method to find the next empty cell of the puzzle.
    Iterates from left to right and top to bottom

    Arguments:
    board - a list of nine sub lists with 9 numbers in each sub list

    Output:
    A tuple (i, j) which is index of row, column
    """
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 0:
                return (i, j)
