import numpy as np

def maillage_segments(N, a):
    angles = [np.pi*k/N for k in range(0, N)]
    points = a * np.array([np.sin(angles), np.cos(angles)])
    
    # Création directe de la liste des segments avec le modulo N
    segments = [[i, (i + 1) % N] for i in range(N)]
    
    return points, segments

# --- Test ---
N = 5
a = 1.0
points, segments = maillage_segments(N, a)

print("Liste des segments :")
print(segments)


import matplotlib.pyplot as plt

X = points[0]
Y = points[1]

plt.figure(figsize=(6, 6))

# Tracé des arêtes directement à partir de la liste des segments
for noeud1, noeud2 in segments:
    plt.plot([X[noeud1], X[noeud2]], [Y[noeud1], Y[noeud2]], color='blue')

# Tracé des points
plt.scatter(X, Y, color='red', s=50, zorder=2)

plt.axis('equal')
plt.show()


