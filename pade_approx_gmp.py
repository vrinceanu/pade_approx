#
# calculates poles and zeros of the Pade approximant for a signal
# this variant uses gmp library 
#
import gmpy2 as gmp, numpy as np, matplotlib.pyplot

def poles_zeros(L):
    P = len(L)//2
    r = P
    scale = 10**8
    S = [gmp.mpz(x*scale) for x in L]
    q,e = {},{}
    q[-2], q[-1], e[-2], e[-1] = [gmp.mpz(0)],[gmp.mpz(1)],gmp.mpz(1),gmp.mpz(1)
    for k in range(2*P):
        e[k] = gmp.mpz(0)
        for a,b in zip(q[k-1],reversed(S[:k+1])):
            e[k] += a*b
        if e[k] == 0:
            r = k//2; break
        qm1 = q[k-1].copy(); qm2 = q[k-2].copy();
        qm2.insert(0,gmp.mpz(0))
        if k%2 == 1: qm1.insert(len(qm1)+1, gmp.mpz(0))
        q[k] = [gmp.divexact(e[k-1]*q1 - e[k]*q2, e[k-2]) for (q1,q2) in zip(qm1,qm2)]
    d1 = [float(gmp.mpq(e[2*k], e[2*k-1])*gmp.mpq(e[2*k-3],e[2*k-2])) + \
          float(gmp.mpq(e[2*k-2], e[2*k-1])*gmp.mpq(e[2*k+1],e[2*k])) for k in range(1,r)]
    d1.insert(0,float(gmp.mpq(e[-2], e[-1])*gmp.mpq(e[1],e[0])))
    d2 = [float(gmp.mpq(e[2*k-4], e[2*k-2])*gmp.mpq(e[2*k],e[2*k-2])) for k in range(1,r)]
    J = np.eye(r, k = 1) + np.diag(d1) + np.diag(d2, k = -1)
    λ,V = np.linalg.eig(J)
    V1 = np.linalg.inv(V)
    ρ = L[0]*V[0]*V1[:,0]
    zo = np.flip(np.argsort(np.abs(ρ)))
    return((λ[zo],ρ[zo]))
