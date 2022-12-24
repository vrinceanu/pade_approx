#
# calculates poles and zeros of the Pade approximant for a signal
#

import numpy as np, fractions as fr
import matplotlib.pyplot as pl

def poles_zeros(S):
    P = len(S)//2
    r = P
    q,e = {},{}
    q[-2], q[-1], e[-2], e[-1] = [0],[1],1,1
    for k in range(2*P):
        e[k] = 0
        for a,b in zip(q[k-1],reversed(S[:k+1])):
            e[k] += a*b
        if e[k] == 0:
            r = k//2; break
        qm1 = q[k-1].copy(); qm2 = q[k-2].copy();
        qm2.insert(0,0)
        if k%2 == 1: qm1.insert(len(qm1)+1, 0)
        q[k] = [(e[k-1]*q1 - e[k]*q2)//e[k-2] for (q1,q2) in zip(qm1,qm2)]
    J = np.eye(r, k = 1)
    for k in range(r):
        if k == 0:
            J[k,k] = float(fr.Fraction(e[2*k-2], e[2*k-1])*fr.Fraction(e[2*k+1],e[2*k]))
        else:
            J[k,k] = float(fr.Fraction(e[2*k], e[2*k-1])*fr.Fraction(e[2*k-3],e[2*k-2])) + \
                     float(fr.Fraction(e[2*k-2], e[2*k-1])*fr.Fraction(e[2*k+1],e[2*k]))
            J[k,k-1] = float(fr.Fraction(e[2*k-4], e[2*k-2])*fr.Fraction(e[2*k],e[2*k-2]))
    λ,V = np.linalg.eig(J)
    V1 = np.linalg.inv(V)
    ρ = S[0]*V[0]*V1[:,0]
    zo = np.flip(np.argsort(ρ))
    return((λ[zo],ρ[zo]))


def plot_poles(L):
    fig,ax=pl.subplots()
    ax.scatter(np.real(L), np.imag(L), marker=".", s=2)
    ax.set_aspect(1)
