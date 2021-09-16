import numpy as np


# Создаем рандомную матрицу, заполненную 0 и 1 по заданному количеству строк и столбцов
def create_matrix(row, column):
    return np.random.randint(0, 2, (row, column))


# Поменять местами две заданные строчки матрицы
def swap_rows(matrix, row1, row2):
    matrix[[row1, row2]] = matrix[[row2, row1]]


# Суммирование двух строчек матрицы, сумма сохраняется в строчку row
def sum_rows(matrix, row, rowsum):
    matrix[row] = np.array([((a + k) % 2) for a, k in zip(matrix[row], matrix[rowsum])])


# Количество строк матрицы
def row_size(matrix):
    return matrix.shape[0]


# Кличество столбцов матрицы
def column_size(matrix):
    return matrix.shape[1]


# Ступенчатый вид
def ref(matrix):
    column = 0  # Первый столбец
    leadrow = 0  # Ведущая строка
    while column < column_size(matrix):  # Идём по столбцам матрицы
        for r in range(leadrow + 1, row_size(matrix)):  # Идём от следующей от ведущей строки до последней строки
            if matrix[r][column] > matrix[leadrow][column]:  # Если элемент больше
                currentrow = r  # Запоминаем строчку
                swap_rows(matrix, currentrow, leadrow)  # Меняем строчку, чтобы была новая ведущая
            if matrix[r][column] == matrix[leadrow][column]:  # Если элементы равны
                sum_rows(matrix, r, leadrow)  # Суммируем строчки чтобы занулить элементы
        if np.all(matrix[leadrow:, column] == 0):  # Если все элементы столбца ниже ведуще строки = 0
            leadrow = leadrow  # Ведущая строка не меняется
        else:
            leadrow += 1  # Ведущая строка становится следующей
        column += 1  # Переходим к следующему столбцу
    return matrix  # Возвращаем полученную матрицу

def rref(originalMatrix): 
    matrix = copy.deepcopy(ref(originalMatrix))
    m = matrix.shape[0]
    n = matrix.shape[1]
    step = 0
    for i in range(m):
        for j in range(step, n):
            if matrix[i][j] == 1: # убираем значения выше ведущих элементов
                for k in range(0, i):
                    matrix[k][j] = 0
                if i != m-1:
                    i += 1
                    step = i
                else:
                    return matrix
            if j == n-1 and matrix[i][j] == 0: # если дошли до нулевой строки делаем обрезание
                return matrix[1:i-1, :]
    return matrix

# Убираем нулевые строчки
def no_zero_matrix(matrix):
    r = 0  # Первая строчка
    while r < row_size(matrix):  # Пока не пройдем все строчки
        if np.all(matrix[r, :] == 0):  # Если все элементы строчки = 0
            matrix = np.delete(matrix, r, 0)  # Удаляем эту строчку
            r = r  # Не сдвигаем номер строчки
        else:
            r += 1 # Если ничего не удаляли, то переходим к новой строчке
    return matrix


if __name__ == '__main__':
    rand_matrix = create_matrix(6, 11)  # Создаем матрицу, размер пишем в ()
    print('Исходная матрица:')
    print(rand_matrix)
    step_matrix = ref(rand_matrix)
    print('Ступенчатая матрица:')
    print(step_matrix)
    no_zero = no_zero_matrix(step_matrix)
    print('Ступенчатая матрица без нулей:')
    print(no_zero)
