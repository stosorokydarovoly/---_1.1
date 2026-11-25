import numpy as np
import os
import time
from scipy.sparse import random as sparse_random
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt

#task 6
def task6():
    #sub-task 6.1
    print(f"6.1")
    arr = np.arange(20)

    res_1 = arr[1::3]
    print(f"каждый третий элемент, начиная с индекса 1: {res_1}")

    res_2 = arr[5:15][::-1]
    print(f"элементы с 5-го по 15-й (не включая 15) в обратном порядке: {res_2}")

    res_3 = arr[-5:][::-1]
    print(f"последние 5 элементов в обратном порядке: {res_3}\n")

    #sub-task 6.2
    print(f"6.2")
    matrix = np.random.randint(0, 10, size=(6, 6))
    print(matrix)

    submatrix = matrix[1:4, 0:3]
    print(f"\nПодматрица:\n{submatrix}")

    diag = np.diagonal(matrix)
    print(f"\nДиагональ: {diag}")

    rows, cols = np.indices(matrix.shape)
    chess_mask = (rows + cols) % 2 == 0
    chess = np.where(chess_mask, matrix, 0)
    print(f"\nУзор:\n {chess}\n")

    #sub-task 6.3
    print(f"6.3")
    a = np.array([10, 20, 30, 40, 50, 60])
    m = np.arange(12).reshape(3, 4)

    res_a = a[[0, 2, 4]]
    print(f"Элементы a по индексам [0, 2, 4]: {res_a}")

    res_m = m[[0, 1, 2], [1, 3, 0]]
    print(f"Элементы (0,1), (1,3), (2,0): {res_m}")

    res_m_2 = m[np.ix_([0, 2], [1, 3])]
    print(f"Блок из строк [0, 2] и столбцов [1, 3] в виде матрицы 2×2\n{res_m_2}\n")

    #sub-task 6.4
    print(f"6.4")
    x = np.random.randn(1000)

    count_x = np.sum(x > 1)
    print(f"Количество элементов, больших 1: {count_x}")

    x[x < 0] = 0
    print(f"Замена всех отрицательных элементов на 0:\n{x}\n")

    signs = np.sign(x)
    mask = np.zeros(len(x), dtype=bool)
    mask[1:-1] = (signs[:-2] != signs[2:]) 
    x[mask] = 0

    print(f"Массив после обнуления элементов со сменой знака у соседей: {x[:20]}\n")

#task 7
def task7():
    #sub-task 7.1
    print(f"7.1")
    arr = np.random.rand(50000, 200).astype(np.float32)
    arr.tofile('big.dat')

    expected_size = 50000 * 200 * 4
    actual_size = os.path.getsize('big.dat')
    print(f"Размер / ожидаемый размер: {actual_size, expected_size}\n")

    #sub-task 7.2
    print(f"7.2")
    total_sum = 0.0
    total_count = 0

    with open('big.dat', 'rb') as f:
        for start in range(0, 50000, 1000 ):
            n = min(1000 , 50000 - start)
            # Читаем блок
            data = np.fromfile(f, dtype=np.float32, count=n * 200)
            data = data.reshape(n, 200)

            global_min = min(np.inf, data.min())
            global_max = max(-np.inf, data.max())
            total_sum += data.sum()
            total_count += data.size

    print(f"Min: {global_min}")
    print(f"Max: {global_max}")
    print(f"Среднее:{total_sum / total_count}\n")

    #sub-task 7.3
    print(f"7.3")
    arr_c = np.random.rand(3000, 3000)

    start = time.time()
    row_sums_c = []
    for i in range(3000):
        s = 0.0
        for j in range(3000):
            s += arr_c[i, j]
        row_sums_c.append(s)
    time_rows_c = time.time() - start

    start = time.time()
    col_sums_c = []
    for j in range(3000):
        s = 0.0
        for i in range(3000):
            s += arr_c[i, j]
        col_sums_c.append(s)
    time_cols_c = time.time() - start

    print(f"Сумма по строкам (C-order): {time_rows_c}")
    print(f"Сумма по столбцам (C-order): {time_cols_c}\n")

    arr_f = np.asfortranarray(arr_c)

    start = time.time()
    row_sums_f = []
    for i in range(3000):
        s = 0.0
        for j in range(3000):
            s += arr_f[i, j]
        row_sums_f.append(s)
    time_rows_f = time.time() - start

    start = time.time()
    col_sums_f = []
    for j in range(3000):
        s = 0.0
        for i in range(3000):
            s += arr_f[i, j]
        col_sums_f.append(s)
    time_cols_f = time.time() - start

    print(f"Сумма по строкам (F-order): {time_rows_f}")
    print(f"Сумма по столбцам (F-order): {time_cols_f}\n")

    #sub-task 7.4
    print(f"7.4")
    a = np.random.rand(10000000)
    b = np.random.rand(10000000)

    start = time.time()
    res_py = []
    for i in range(10000000):
        res_py.append((a[i] + b[i]) / (a[i] - b[i] + 1))
    time_1 = time.time() - start

    start = time.time()
    res_numpy = (a + b) / (a - b + 1)
    time_2 = time.time() - start

    buf_1 = np.empty_like(a)
    buf_2 = np.empty_like(a)

    start = time.time()
    np.add(a, b, out=buf_1)           
    np.subtract(a, b, out=buf_2)      
    np.add(buf_2, 1, out=buf_2)        
    np.divide(buf_1, buf_2, out=buf_1) 
    time_3 = time.time() - start

    print(f"Python-цикл: {time_1}")
    print(f"NumPy в одну строку: {time_2}")
    print(f"NumPy с out=: {time_3}\n")

#task 8
def task8():
    #sub-task 8.1
    print(f"8.1")
    np.random.seed(42)
    row = np.random.randint(0, 5000, size=10000)
    col = np.random.randint(0, 5000, size=10000)
    data = np.random.rand(10000)


    coo = coo_matrix((data, (row, col)), shape=(5000, 5000))
    print(f"COO: {coo.nnz}")

    csr = coo.tocsr()
    print(f"CSR: {csr.nnz}")

    print(f"Совпадает ли nnz: {coo.nnz == csr.nnz}\n") 

    #sub-task 8.2
    print(f"8.2")

    A_sparse = sparse_random(4000, 4000, density=0.005, format='csr', dtype=np.float64)

    A_dense = A_sparse.toarray()

    mem_dense = 4000 * 4000 * 8
    mem_sparse = A_sparse.data.nbytes + A_sparse.indices.nbytes + A_sparse.indptr.nbytes

    print(f"Память dense (float64): {mem_dense / 1024**2}")
    print(f"Память CSR: {mem_sparse / 1024**2}")
    print(f"Экономия: {mem_dense / mem_sparse}\n")

    x = np.random.rand(4000)

    start = time.time()
    y_sparse = A_sparse @ x
    time_sparse = time.time() - start

    start = time.time()
    y_dense = A_dense @ x
    time_dense = time.time() - start

    print(f"Время умножения (sparse): {time_sparse}")
    print(f"Время умножения (dense):  {time_dense}")
    print(f"Ускорение sparse: {time_dense / time_sparse}\n")

    #sub-task 8.3
    print(f"8.3")

    densities = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    times = []

    x = np.random.rand(2000)

    for d in densities:
        A = sparse_random(2000, 2000, density=d, format='csr', dtype=np.float64)
        mat = A @ x

        t_total = 0
        for mat in range(3):
            start = time.perf_counter()
            _mat = A @ x
            t_total += time.perf_counter() - start
        times.append(t_total / 3)

    plt.figure(figsize=(8, 5))
    plt.plot(densities, times, 'o-', label='CSR @ vector')
    plt.xscale('log')
    plt.xlabel('Плотность (log scale)')
    plt.ylabel('Время умножения, с')
    plt.title(f'Время умножения A @ x для n = 2000')
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.tight_layout()
    plt.show()

task6()
input("Нажмите для продолжения...")
task7()
input("Нажмите для продолжения...")
task8()