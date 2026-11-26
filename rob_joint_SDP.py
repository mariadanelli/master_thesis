import numpy as np
import cvxpy as cp
import cvxpy as cp
import numpy as np
from sympy.physics.matrices import msigma
from qiskit.quantum_info import Pauli, SparsePauliOp, Clifford
import math
from functools import reduce


def get_pauli_qudit(d):
    w = np.exp((2j * np.pi) / d)
    
    X = np.zeros((d, d), dtype=complex)
    for j in range(d):
        X[(j + 1) % d, j] = 1.0


    
    diagonale_z = [w**j for j in range(d)]
    Z = np.diag(diagonale_z)
    
    return X, Z





def generalized_pauli(array, n, d):
    X = get_pauli_qudit(d)[0]
    Z = get_pauli_qudit(d)[1]
    L= None
    for i in range(n):
        idx_1 = 2 * i
        idx_2 = (2 * i) + 1
    
        
        Z_pow = np.linalg.matrix_power(Z, array[idx_1])
        X_pow = np.linalg.matrix_power(X, array[idx_2])
        inv_2 = (d + 1) // 2  #DIVISIONE PER DUE MODULARE
        if d == 2:
            
            fase = 1j**(array[idx_1] * array[idx_2])
        else:
           
            w = np.exp((2j * np.pi )/ d)
            fase = w**(-(array[idx_1] * array[idx_2]) * inv_2)
        

    
        T = fase * (Z_pow @ X_pow)
        
        if (i==0):
            L= T 
        else:
            L= np.kron(L,T)
            
    return L



def kron_all(ops):
    return reduce(np.kron, ops)

def genera_generatori_clifford_n_qudit(d, n):
    omega = np.exp(2j * np.pi / d)
    Id = np.eye(d, dtype=complex)


    X = np.zeros((d, d), dtype=complex)
    for k in range(d):
        X[(k + 1) % d, k] = 1.0

   
    j_idx, k_idx = np.meshgrid(np.arange(d), np.arange(d), indexing='ij')
    F = (1.0 / np.sqrt(d)) * (omega ** (j_idx * k_idx))


    z = np.arange(d)
    f_z = (z ** 2) / 2.0 if d % 2 == 0 else (z ** 2 - z) / 2.0
    S = np.diag(omega ** f_z)

    generatori = []

    
    for j in range(n):
        for gate in (X, F, S):
            ops = [Id] * n
            ops[j] = gate
            generatori.append(kron_all(ops))

    
    if n >= 2:
        
        X_powers = [np.linalg.matrix_power(X, x) for x in range(d)]

        
        coppie = [(j, j + 1) for j in range(n - 1)] + [(j + 1, j) for j in range(n - 1)]

        for c, t in coppie:
            SUM_ct = np.zeros((d**n, d**n), dtype=complex)
            for x in range(d):
                proj_x = np.zeros((d, d), dtype=complex)
                proj_x[x, x] = 1.0

                ops = [Id] * n
                ops[c] = proj_x
                ops[t] = X_powers[x]
                SUM_ct += kron_all(ops)

            generatori.append(SUM_ct)

    return generatori


d= 2
n=2




Z = Pauli('Z').to_matrix()
X = Pauli('X').to_matrix()
Y = Pauli('Y').to_matrix()
I = np.eye(2, dtype=complex)

single_paulis = [I, X, Y, Z]
P_list = [np.kron(p1, p2) for p1 in single_paulis for p2 in single_paulis]
P = np.array(P_list) 





clifford=genera_generatori_clifford_n_qudit(d,n)

state0 = np.array([[1], [0]], dtype=complex)
state1 = np.array([[0], [1]],  dtype=complex)

state00 = np.kron(state0, state0)
state11 = np.kron(state1, state1)
psi = (1 / np.sqrt(2)) * (state00 + np.exp(1j * np.pi / 4) * state11)



stateh= (1 / np.sqrt(2)) *(state0+ np.exp(1j* np.pi/4)* state1)
stateh=np.kron(stateh,stateh)
#psi=np.kron(stateh, stateh)


beta = 0.5 * np.arccos(1 / np.sqrt(3))
stateT = np.cos(beta) * state0 + np.exp(1j * np.pi / 4) * np.sin(beta) * state1


#psi = np.kron(stateT, stateT)  # stato magico ma non ent -> rob= 0
#psi = (1 / np.sqrt(2)) * (state00 + state11) #stato ent ma non magico -> rob= 0
#psi= state00

phipiu= (1 / np.sqrt(2)) * (state00 + state11)
phimeno= (1 / np.sqrt(2)) * (state00 - state11)
stateT= np.kron(stateT, stateT)

p= 0.2
rho = (p* phipiu@phipiu.conj().T +(1-p)*stateT@ stateT.conj().T)  #stato che ha sia magic sia ent ma è libero!

#rho= psi@psi.T.conj()
#rho= p* rho+(1 - p) / 4 *np.eye(4, dtype=complex)


def normalize_phase(vec, tol=1e-10):
    idx = np.argmax(np.abs(vec) > tol)
    return vec * (np.abs(vec[idx]) / vec[idx])



N = (d**n) * math.prod(d**(n - k) + 1 for k in range(n))

v0 = normalize_phase(state00)
v = [v0]
coda = [v0]


while coda:
    curr = coda.pop(0)
    
    for U in clifford:
        v1 = normalize_phase(U @ curr)
        if not any(np.allclose(v1, x, atol=1e-10) for x in v):
            v.append(v1)
            coda.append(v1)

sigma = [phi @ phi.conj().T for phi in v]
from fractions import Fraction

def format_coeff(c, tol=1e-9):
    """Converte un numero complesso in una stringa pulita con frazioni/radici."""
    re, im = np.real(c), np.imag(c)
    if abs(re) < tol: re = 0.0
    if abs(im) < tol: im = 0.0
    if re == 0.0 and im == 0.0:
        return "0"

    def num_str(val):
        sign = "-" if val < 0 else ""
        a = abs(val)
        # Controlla se è un razionale semplice o un multiplo di 1/sqrt(2) o 1/2
        for denom_label, scale in [("", 1.0), ("/√2", np.sqrt(2))]:
            frac = Fraction(a * scale).limit_denominator(16)
            if abs(a * scale - float(frac)) < 1e-7:
                if denom_label == "":
                    return f"{sign}{frac}"
                else:
                    num = "" if frac.numerator == 1 else f"{frac.numerator}"
                    den = f"/{frac.denominator}" if frac.denominator != 1 else ""
                    return f"{sign}{num}{denom_label}{den}"
        return f"{val:.3f}"

    if im == 0.0:
        return num_str(re)
    if re == 0.0:
        s = num_str(im)
        return ("i" if s == "1" else "-i" if s == "-1" else f"{s}i")
    
    im_s = num_str(im)
    op = "+" if im > 0 else "-"
    im_abs = im_s.lstrip("-")
    im_part = "i" if im_abs == "1" else f"{im_abs}i"
    return f"({num_str(re)} {op} {im_part})"

def print_stabilizers_dirac(v_list):
    """Stampa la lista dei vettori di stato |phi_i> in notazione di Dirac."""
    basis = ["|00⟩", "|01⟩", "|10⟩", "|11⟩"]
    for idx, vec in enumerate(v_list):
        vec = vec.flatten()
        terms = []
        for amp, ket in zip(vec, basis):
            s = format_coeff(amp)
            if s != "0":
                if s == "1":
                    terms.append(ket)
                elif s == "-1":
                    terms.append(f"-{ket}")
                else:
                    terms.append(f"{s}{ket}")
        expr = " + ".join(terms).replace("+ -", "- ")
        print(f"σ[{idx:2d}]:  |{expr}⟩" if False else f"σ[{idx:2d}]:  {expr}")

# Esempio d'uso sulla tua lista v:
print_stabilizers_dirac(v)



def partial_transpose_b_cvxpy(mat):
    blocks = []
    for i in range(2):
        row_blocks = []
        for j in range(2):
            block_ij = mat[i*2:(i+1)*2, j*2:(j+1)*2]
            row_blocks.append(block_ij.T)
        blocks.append(cp.hstack(row_blocks))
    return cp.vstack(blocks)

x = cp.Variable(N, nonneg=True)
x_prime = cp.Variable(N) #metti nonneg= True per una robustess non generalized

tau = cp.Variable((4, 4), hermitian=True)
tau_prime = cp.Variable((4, 4), hermitian=True)

A = np.zeros((16, N), dtype=float)
b = np.zeros(16, dtype=float)
for i in range(16):
    b[i] = np.real(np.trace(rho @ P[i]))
    for j in range(N):
        A[i, j] = np.real(np.trace(P[i] @ sigma[j]))

c = cp.hstack([cp.real(cp.trace(tau @ P[i])) for i in range(16)])
c_prime = cp.hstack([cp.real(cp.trace(tau_prime @ P[i])) for i in range(16)])

s = cp.real(cp.trace(tau))
s_prime = cp.real(cp.trace(tau_prime))



constraints = [
    tau >> 0,
    partial_transpose_b_cvxpy(tau) >> 0,
    tau_prime >> 0,
   
    #partial_transpose_b_cvxpy(tau_prime) >> 0, decommenta per robustness non generalizz
    A @ (x - x_prime) + c - c_prime == b,
]


objective =  cp.Minimize(cp.norm(x_prime, 1)+ s_prime)
             
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.CLARABEL)

print(f"Stato del solver: {problem.status}")
print(f"Robustness (lambda): {problem.value:.6f}")