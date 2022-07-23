from scipy.linalg import sqrtm
from numpy.linalg import inv
import numpy
from numpy import dot, eye, zeros, outer
from numpy.linalg import inv

A1 = numpy.array([[-0.989992, -0.14112,  0.000, 0],
                 [0.141120 , -0.989992, 0.000, 0],
                 [0.000000 ,  0.00000, 1.000, 0],
                 [0        ,        0,     0, 1]])

B1 = numpy.array([[-0.989992, -0.138307, 0.028036, -26.9559],
                 [0.138307 , -0.911449, 0.387470, -96.1332],
                 [-0.028036 ,  0.387470, 0.921456, 19.4872],
                 [0        ,        0,     0, 1]])

A2 = numpy.array([[0.07073, 0.000000, 0.997495, -400.000],
                [0.000000, 1.000000, 0.000000, 0.000000],
                [-0.997495, 0.000000, 0.070737, 400.000],
                [0, 0, 0,1]])

B2 = numpy.array([[ 0.070737, 0.198172, 0.997612, -309.543],
                [-0.198172, 0.963323, -0.180936, 59.0244],
                [-0.977612, -0.180936, 0.107415, 291.177],
                [0, 0, 0, 1]])


def logR(T):
    R = T[0:3, 0:3]
    theta = numpy.arccos((numpy.trace(R) - 1)/2)
    logr = numpy.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]]) * theta / (2*numpy.sin(theta))
    return logr

def Calibrate(A, B):
    n_data = len(A)
    M = numpy.zeros((3,3))
    C = numpy.zeros((3*n_data, 3))
    d = numpy.zeros((3*n_data, 1))
    A_ = numpy.array([])
    
    for i in range(1):
        alpha = logR(A[i])
        beta = logR(B[i])
        alpha2 = logR(A[i+1])
        beta2 = logR(B[i+1])
        alpha3 = numpy.cross(alpha, alpha2)
        beta3  = numpy.cross(beta, beta2) 
        
        M1 = numpy.dot(beta.reshape(3,1),alpha.reshape(3,1).T)
        M2 = numpy.dot(beta2.reshape(3,1),alpha2.reshape(3,1).T)
        M3 = numpy.dot(beta3.reshape(3,1),alpha3.reshape(3,1).T)
        M = M1+M2+M3
    
    theta = numpy.dot(sqrtm(inv(numpy.dot(M.T, M))), M.T)

    for i in range(n_data):
        rot_a = A[i][0:3, 0:3]
        rot_b = B[i][0:3, 0:3]
        trans_a = A[i][0:3, 3]
        trans_b = B[i][0:3, 3]
        
        C[3*i:3*i+3, :] = numpy.eye(3) - rot_a
        d[3*i:3*i+3, 0] = trans_a - numpy.dot(theta, trans_b)
        
    b_x  = numpy.dot(inv(numpy.dot(C.T, C)), numpy.dot(C.T, d))
    return theta, b_x

X = numpy.eye(4)
A = [A1, A2]
B = [B1, B2]
theta, b_x = Calibrate(A, B)
X[0:3, 0:3] = theta
X[0:3, -1] = b_x.flatten()

print("X: ")
print(X)
 