import numpy as np
import cvxpy as cp
from sympy.physics.matrices import msigma
from qiskit.quantum_info import Pauli, SparsePauliOp, Clifford
import math
from functools import reduce
from fractions import Fraction
from scipy.linalg import null_space






Z = Pauli('Z').to_matrix()
X = Pauli('X').to_matrix()
Y = Pauli('Y').to_matrix()
I = np.eye(2, dtype=complex)

single_paulis = [I, X, Y, Z]
P_list = [np.kron(p1, p2) for p1 in single_paulis for p2 in single_paulis]
P = np.array(P_list) 


def format_coeff(c, tol=1e-9):
    re, im = np.real(c), np.imag(c)
    if abs(re) < tol: re = 0.0
    if abs(im) < tol: im = 0.0
    if re == 0.0 and im == 0.0:
        return "0"

    def num_str(val):
        sign = "-" if val < 0 else ""
        a = abs(val)
        
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


I4 = np.eye(4, dtype=complex)
v = []

def normalize_phase(vec, tol=1e-10):
    idx = np.argmax(np.abs(vec) > tol)
    return vec * (np.abs(vec[idx]) / vec[idx])

for i in range(1, 16):
    for j in range(i + 1, 16): #siccome S è composto da soli due elementi, perchè siano indipendenti, mi basta che siano diversi 
        if np.allclose(P[i] @ P[j], P[j] @ P[i], atol=1e-10):
            for s1 in (1, -1):
                for s2 in (1, -1):
                    sistema = np.vstack([s1 * P[i] - I4, s2 * P[j] - I4])
                    ns = null_space(sistema, rcond=1e-10)
                    if ns.shape[1] == 1: #nel caso di due (n=2) qubit con dim S=2  (l=2) dovrebbe essere sempre non degenre (k=0)
                        psi = normalize_phase(ns[:, 0])
                        if not any(np.allclose(psi, existing, atol=1e-10) for existing in v):
                            v.append(psi)

print(f"Totale stati stabilizzatori unici trovati: {len(v)}")
print_stabilizers_dirac(v)

sigma = np.array([np.outer(psi, psi.conj()) for psi in v])

print(len(sigma))

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

#t=0.9999999
p= 0.0000001
rho = (p* phipiu@phipiu.conj().T +(1-p)*stateT@ stateT.conj().T)  #stato che ha sia magic sia ent ma è libero!

#rho= psi@psi.T.conj()
#rho= p* rho+(1 - p) / 4 *np.eye(4, dtype=complex)

#rho= p* rho+(1 - p)* (t* phipiu@phipiu.conj().T +(1-t)*stateT@ stateT.conj().T)



 

def partial_transpose_b_cvxpy(mat):
    blocks = []
    for i in range(2):
        row_blocks = []
        for j in range(2):
            block_ij = mat[i*2:(i+1)*2, j*2:(j+1)*2]
            row_blocks.append(block_ij.T)
        blocks.append(cp.hstack(row_blocks))
    return cp.vstack(blocks)

x = cp.Variable(60, nonneg=True)
x_prime = cp.Variable(60, nonneg=True)

tau = cp.Variable((4, 4), hermitian=True)
tau_prime = cp.Variable((4, 4), hermitian=True)

A = np.zeros((16, 60), dtype=float)
b = np.zeros(16, dtype=float)
for i in range(16):
    b[i] = np.real(np.trace(rho @ P[i]))
    for j in range(60):
        A[i, j] = np.real(np.trace(P[i] @ sigma[j]))

c = cp.hstack([cp.real(cp.trace(tau @ P[i])) for i in range(16)])
c_prime = cp.hstack([cp.real(cp.trace(tau_prime @ P[i])) for i in range(16)])




constraints = [
    tau >> 0,
    partial_transpose_b_cvxpy(tau) >> 0,
    tau_prime >> 0,
    partial_transpose_b_cvxpy(tau_prime) >> 0, 
    A @ (x - x_prime) + c - c_prime == b,
]

s = cp.trace(tau_prime) + cp.sum(x_prime)

objective =  cp.Minimize(s)
         
problem = cp.Problem(objective, constraints)

problem.solve(solver=cp.CLARABEL)

print(f"Stato del solver: {problem.status}")
print(f"Robustness (lambda): {problem.value:.6f}")