import time

import numpy as np
import json
import pickle

from pyscf import gto, scf, fci
import openfermion
import openfermionpyscf 

from symmer.operators import PauliwordOp, QuantumState
from symmer.projection import QubitTapering
from symmer.projection import ContextualSubspace

from evolution import VQD


geometry = [('Li', (0, 0, 0.3868)), ('H', (0, 0, -1.3))]
basis = 'sto-3g'
multiplicity = 1
charge = 0
mol = gto.M(
    atom = geometry,
    basis = basis,
    symmetry = True,
)

mf = scf.RHF(mol)
mf.kernel()
cisolver = fci.FCI(mol, mf.mo_coeff)
nroots = 3  
e, fcivec = cisolver.kernel(nroots=nroots)
for i in range(nroots):
    print(f'state {i} energy: ', e[i])

filename = '../../hamiltonian_data/LiH_STO-3G_SINGLET_JW.json'
with open(filename, 'r') as infile:
    data_dict = json.load(infile)
hf_state   = QuantumState(np.asarray(data_dict['data']['hf_array'])) # Hartree-Fock state

hamiltonian = openfermionpyscf.generate_molecular_hamiltonian(
    geometry, basis, multiplicity, charge
)
hamiltonian_ferm_op = openfermion.get_fermion_operator(hamiltonian)
qubit_hamiltonian = openfermion.jordan_wigner(hamiltonian_ferm_op)
# qubit_hamiltonian.compress()

hamiltonian_pauli = PauliwordOp.from_openfermion(qubit_hamiltonian)

QT = QubitTapering(hamiltonian_pauli)
H_taper   = QT.taper_it(ref_state=hf_state)
cs_vqe = ContextualSubspace(H_taper, noncontextual_strategy='diag', unitary_partitioning_method='LCU')


UCC_q = PauliwordOp.from_dictionary(data_dict['data']['auxiliary_operators']['UCCSD_operator'])
UCC_taper = QT.taper_it(aux_operator=UCC_q)#taper consistently with the identified symmetry of the Hamiltonian

max_qubits = 8
# max_qubits = 3

error_fci = {i: [0, 0, 0] for i in range(1, max_qubits + 1)}
his_data = {i: [0, 0, 0] for i in range(1, max_qubits + 1)}

print('The Hamiltonian before contextualized has {} qubits'.format(H_taper.n_qubits))

n_excited=2

def run_vqd(num_qubit):
    cs_vqe.update_stabilizers(n_qubits=num_qubit, strategy='aux_preserving', aux_operator=UCC_taper)
    H_cs = cs_vqe.project_onto_subspace()
    UCC_cs = cs_vqe.project_onto_subspace(UCC_taper)
    hf_cs = cs_vqe.project_state(QT.tapered_ref_state)
    vqd = VQD(observable=H_cs, ref_state=hf_cs, excitation_ops=UCC_cs, excited_order=n_excited, beta_list=[10, 10])
    vqd.verbose = 1
    return vqd.run()

for num_qubit in range(1, max_qubits+1):
    print(f'Number of qubits: {num_qubit}')
    vqd_result, interim_data = run_vqd(num_qubit)
    his_data[num_qubit] = interim_data

    for i in range(3):
        vqd_energy = vqd_result[i]['energy'] if i > 0 else vqd_result[i]['fun']
        error_fci[num_qubit][i] = abs(vqd_energy - e[i])
        print(f'Converged VQD e{i} energy = {vqd_energy} with FCI error {error_fci[num_qubit][i]}')

his_data_dict = {'e0': {i: his_data[i][0] for i in range(1, max_qubits + 1)},
                 'e1': {i: his_data[i][1] for i in range(1, max_qubits + 1)},
                 'e2': {i: his_data[i][2] for i in range(1, max_qubits + 1)}}