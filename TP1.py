import fonctions

points,segments,milieux,longueurs,normales = maillage_segments(N, a, forme="cercle")
def G(x,y):
    return 0

quad, milieux = []
p =[]
def A(x, N): #quad est une quadrature (tableau de taille n_q) contenant les tableaux poid,point
    A = np.array(N,len(quad))
    for i in range(N):
        for j in range(len(quad)):
            A[i][j] = quad[0][j] * G(x, 0.5* ((points[segments[0][i]] + points[segments[1][i]]) + quad[1][j]*(points[segments[1][i]] - points[segments[0][i]]))) * longueurs[i]/2
def u(x,p):
    return A(x,n)*p
    
