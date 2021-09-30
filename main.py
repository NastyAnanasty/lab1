import copy

import numpy as np


# Создаем рандомную матрицу, заполненную 0 и 1 по заданному количеству строк и столбцов
def create_matrix(row, column):
    matrix = np.random.randint(0, 9, (row, column))
    for r in range(0, row):
        matrix[r] = matrix[r] % 2
    return matrix


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


def rref(originalMatrix): #приведённый ступенчатый вид
    matrix = copy.deepcopy(no_zero_matrix(originalMatrix))
    k = matrix.shape[0]
    n = matrix.shape[1]
    curRow = 0 #текущая ступенька
    for j in range(n): #проходим по всем столбцам
        for i in range(curRow + 1, k): #по строкам кроме уже построенных ступенек
            if matrix[i][j] == 1: #в этой строчке на данном столбце 1 быть не может
                if matrix[curRow][j] == 0: #если ступенька не построена
                    matrix[curRow] = (matrix[i] + matrix[curRow]) % 2 #строим ступеньку
                    matrix[i] = (matrix[i] + matrix[curRow]) % 2 #обнуляем 1 в данной строчке
                else: #если уже построена
                    matrix[i] = (matrix[i] + matrix[curRow]) % 2 #обнуляем 1 в данной строчке
        if matrix[curRow][j] == 1: #если в даном столбце построили новую ступеньку
            for i in range(curRow): #по уже построенным ступенькам
                if matrix[i][j] == 1: #в этой строчке на данном столбце 1 быть не может
                    matrix[i] = (matrix[i] + matrix[curRow]) % 2 #складываем строки
            curRow += 1 #начинаем строить следующую ступеньку
            if curRow == k: #если все ступеньки построены, выходим
                break
    return matrix[0:curRow]

# Убираем нулевые строчки
def no_zero_matrix(matrix):
    r = 0  # Первая строчка
    while r < row_size(matrix):  # Пока не пройдем все строчки
        if np.all(matrix[r, :] == 0):  # Если все элементы строчки = 0
            matrix = np.delete(matrix, r, 0)  # Удаляем эту строчку
            r = r  # Не сдвигаем номер строчки
        else:
            r += 1  # Если ничего не удаляли, то переходим к новой строчке
    return matrix

def all_perm(matrix):
    matr = rref(matrix)
    k = np.shape(matr)[0]
    k2 = 2 ** k
    new_matrix = np.array([[0] * k])
    for i in range(k2):
        inline_array = np.zeros(k, dtype = int)
        bin_number = f"{i:b}"
        j = len(bin_number) - 1
        z = 0
        while j >= 0:
            inline_array[k - z - 1] = bin_number[j]
            z+=1
            j-=1
        new_matrix = np.append(new_matrix, [inline_array], axis = 0)
    return new_matrix

def mult_k_g(matrix):
    return np.matmul(all_perm(matrix), rref(matrix))

def distance(matrix):
    all_words = mult_k_g(matrix)
    min_count = np.shape(all_words)[1]
    for i in range(len(all_words)):
        for j in range(i + 1, len(all_words)):
            temp = 0
            for k in range(np.shape(all_words)[1]):
                if all_words[i][k] != all_words[j][k]:
                    temp += 1
            if temp < min_count: min_count = temp
    return min_count

def error_check(matrix):
    def t = distance(matrix) - 1
    def g = rref(matrix)
    def count = np.shape(g)[1]
    def need_to_brake = false
    def i = 0

    for i in range(len(g)):
        if (need_to_brake):
            break
        def temp = 0
        for j in range(countElem):
            if g[i][j] == 0:
                temp += 1
            if temp == t:
                need_to_brake = true
    
    def temp_row = g[i].slice()
    for k in range(countElem):
        if temp_row[k] == 0:
            temp_row[k] = 1
            j++
        if j == t:
            break
    print('Массив с добавленной ошибкой - ' + temp_row + '\n Номер строки в матрице g - ' + i)
    def h = h()
    def arr = []    
    arr.append(temp_row)
    def mult_t_row_h = np.matmul(arr, h)
    print(mult_t_row_h)
    
def final_task(matrix):
    def d = distance(matrix)
    def c = mult_k_g(matrix)
    def row = 0
    def k = 0
    for i in range(len(c)):
        k = 0
        for j in range(len(c[i])):
            if c[i][j] == true:
                k++
            if k == d
                row = i
                break
    
    for i in range(len(c)):
        if c[row][i] == true:
            c[0][i] = 1
    
    def r = c[0]
    def arr = []
    arr.append(r)
    return np.matmul(arr, h())

class LinearMatrix(object):

    def __init__(self, matrix):
        self.matrix = matrix
        self.n = 0  # число столбцов
        self.k = 0  # число строк
        self.lead_columns = []  # лидирующие столбцы ступенчатой матрицы

    def rref(self):
        new_matrix = rref(self.matrix)
        self.n = column_size(new_matrix)
        self.k = row_size(new_matrix)
        return new_matrix

    def get_leading(self):
        lead = []

        for i in range(self.n - 1):
            for j in range(len(lead) - 1, self.k - 1):
                if self.matrix[j][i] == 1:
                    lead.append(i)
                    break

        return lead

    def get_short_matrix(self):
        lead_columns = self.get_leading
        self.lead_columns = lead_columns
        print(lead_columns)
        short_matrix = np.delete(self.matrix, lead_columns, 1)
        return short_matrix

    def getH(self):
        self.matrix = self.rref()

        short_matrix = self.get_short_matrix()
        identity_matrix = np.identity(self.k)

        h_matrix_rows = self.k + row_size(short_matrix)
        h_matrix = create_matrix(h_matrix_rows, self.k)

        j = 0
        for i in range(0, h_matrix_rows - 1):
            if (self.lead_columns.count(i) > 0):
                index = self.lead_columns.index(i)
                h_matrix[i] = short_matrix[index]
            else:
                h_matrix[i] = identity_matrix[j]
                j += 1

        return h_matrix

if __name__ == '__main__':
    matrix = create_matrix(10, 5)  # Создаем матрицу, размер пишем в ()
    print('Исходная матрица: ')
    print(matrix)
    matrix1 = all_perm(matrix)
    print('Матрица всех возможных комбинаций длины n: ')
    print(matrix1)
    matrix2 = mult_k_g(matrix)
    print('Произведения матрицы всех возможных комбинаций длины n и матрицы  приведённого ступенчатого вида: ')
    print(matrix2)
    min_distance = distance(matrix)
    print('Минимальное расстояние между кодами: ')
    print(min_distance)
    print(error_check(matrix))
    print(final_task(matrix))

    rand_matrix = create_matrix(5, 10)  # Создаем матрицу, размер пишем в ()
    linear_matrix = LinearMatrix(rand_matrix)
    print('Матрица Н: \n' + linear_matrix.getH())