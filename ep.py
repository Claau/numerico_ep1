import numpy as np
from numpy.lib import diag_indices
from helpers import *
from numpy.linalg import eig

def printmatrix(A):
    for index, cell in np.ndenumerate(A):
        A[index]= round(A[index], 2)
    np.set_printoptions(suppress=True)
    print(A)

error = 0.000001
A = np.array([
    [2, -1,  0,  0],
    [-1, 2, -1,  0],
    [ 0,-1,  2, -1],
    [ 0, 0, -1,  2],
])
(n,n) = np.shape(A)
diag = np.diag_indices(n)


w,v=eig(A)
print('\nEXPECTED RESULTS\n')
print('E-value:')
printmatrix(w)
print('\nE-vector:')
printmatrix(v)

## ANALITIC VALUES
analitic_eigenval = np.zeros(shape=(n,n))
analitic_eigenvec = np.zeros(shape=(n,n))
eigenvalues = np.zeros(shape=(n,n))
eigenvectors = np.identity(n)

for (i,j), x in np.ndenumerate(A):
    analitic_eigenval[i,i] = 2*(1-np.cos((i+1)*np.pi/(n+1)))
    analitic_eigenvec[i,j] = 1-np.sin((i+1)*(j+1)*np.pi/(n+1))
  
print('\n\n\nANALITIC RESULTS\n')
print("analitic_eigenvec")
printmatrix(analitic_eigenvec)


## QR
true_val = analitic_eigenval[diag]
eigenvalues = eigenvalues[diag]
with_shif = True
ite = 0

for index,eigval in enumerate(true_val):
    value = true_val[index]
    
    while (value-eigenvalues[index]) > error:
        ite += 1
        A,eigenvalues, eigenvectors = qr(A, eigenvectors, with_shif)
        eigenvalues = np.flip(eigenvalues)
        err = value - eigenvalues[index]
        
print("\n\n\nRESULTS")
if with_shif:
    print("\nITE with shitf",ite)
else:
    print("\nITE without shitf",ite)

print("\neigenvalues")
printmatrix(eigenvalues)
print("\neigenvectors")
printmatrix(eigenvectors)

        

    
