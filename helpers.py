import numpy as np

def givens_rotation(A):
    
    def get_givens_matrix(a, b):
        r = np.sqrt(a**2+b**2)
        c = a/r
        s = -b/r

        G = np.identity(n)
        G[[j, i], [j, i]] = c
        G[i, j] = s
        G[j, i] = -s
        return G

    [n, n] = np.shape(A)
    Q = np.identity(n)
    R = np.array(A)

    (rows, cols) = np.tril_indices(n, -1, n)
    for (i, j) in zip(rows, cols):
            G = get_givens_matrix(R[j, j], R[i, j])
            R = np.matmul(G, R)
            Q = np.matmul(Q, G.T)

    return (Q, R)

def get_eigenvalue(A):
    #gets the eigenvalue and deletes the n column and n row
    (n, n) = np.shape(A)
    eigenvector = A[n-1,n-1]
    A = np.delete(A,n-1,0)
    A= np.delete(A,n-1,1)
    return eigenvector, A

def wilkinson_heuristic(A, n):
    def sgn(d):
        val = 1
        if d<0: val=-1
        return val

    def module(a,b):
        return np.sqrt(a**2+b**2)

    # if A shapes nxn, having n=3:
    # A = [ [ x       x         x    ] 
    #       [ x    alfa_n_1     x    ] 
    #       [ x     b_n_1     alfa_n ]  ]
    index = n-1
    alfa_n   = A[index,index]
    alfa_n_1 = A[index-1,index-1]
    b_n_1    = A[index,index-1]
    d = (alfa_n_1 - alfa_n)/2

    return alfa_n + d - sgn(d)*module(d,b_n_1) 

def printmatrix(A):
    for index, cell in np.ndenumerate(A):
        A[index]= round(A[index], 2)
    np.set_printoptions(suppress=True)
    print(A)

def qr(A, eigenvectors, add_shift):
    (n,n) = np.shape(A)

    def get_shift(AK, n):
        shift = 0
        if add_shift:
            next_mi = wilkinson_heuristic(AK, n)
            shift = next_mi
        s_matrix = np.identity(n)
        diag = np.diag_indices(n)
        s_matrix[diag] = shift
        return s_matrix
        
    AK = A
    value = 0
    shift = get_shift(AK, n)
    Q,R = givens_rotation(AK - shift)
    AK = np.matmul(R, Q) + shift

    eigenvectors = np.matmul(eigenvectors, Q)
    diag = np.diag_indices(n)
    values = AK[diag]

    return AK, values, eigenvectors
