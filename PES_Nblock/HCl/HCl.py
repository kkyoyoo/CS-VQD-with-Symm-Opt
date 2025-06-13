import time

import numpy as np
import json

from pyscf import gto, scf, fci
import openfermion
import openfermionpyscf 

from symmer.operators import PauliwordOp, QuantumState
from symmer.projection import QubitTapering
from symmer.projection import ContextualSubspace

from evolution import VQD

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

def N_block(qc,q_1,q_2,thetax,thetay,thetaz):
    qc.rz(np.pi / 2, q_2)

    qc.cx(q_2, q_1)

    qc.rz(2*thetaz-np.pi/2, q_1)
    qc.ry(np.pi/2-2*thetax, q_2)

    qc.cx(q_1, q_2)

    qc.ry(2*thetay-np.pi/2, q_2)

    qc.cx(q_2, q_1)

    qc.ry(-np.pi/2, q_1)

    return qc

def N_repeat_theta_xyz_phase_shift(qc, thetas,repeats=1):
    n_qubits = qc.num_qubits
    param_index = 0
    for _ in range(repeats):
        qc.barrier()
        for i in range(n_qubits//2):
            q_1 = 2*i
            q_2 = 2*i+1
            qc=N_block(qc, q_1, q_2, thetas[param_index], thetas[param_index+1], thetas[param_index+2])
            param_index += 3
        for i in range(n_qubits//2-1):
            qc=N_block(qc, 2*i+1, 2*i+2, thetas[param_index], thetas[param_index+1], thetas[param_index+2])
            param_index += 3

        for i in range(n_qubits):
            qc.rz(thetas[param_index], i)
            param_index += 1
    
    return qc

n_qubits_used = 6
print(f'Number of qubits used: {n_qubits_used}')

#generate Hartree-Fock state from Symmer
filename = '../../hamiltonian_data/HCl_STO-3G_SINGLET_JW.json'
with open(filename, 'r') as infile:
    data_dict = json.load(infile)
hf_state   = QuantumState(np.asarray(data_dict['data']['hf_array']))

basis = 'sto-3g'; multiplicity = 1; charge = 0
number_of_points=51
his_data=[]

def generate_parameters(prefix, count):
    return [Parameter(f'{prefix}_{i}') for i in range(count)]

def generate_H_pauliword(geometry, basis, multiplicity, charge):
    # generate Hamiltonian from openfermion -> symmer
    hamiltonian = openfermionpyscf.generate_molecular_hamiltonian(geometry, basis, multiplicity, charge)
    hamiltonian_ferm_op = openfermion.get_fermion_operator(hamiltonian)
    qubit_hamiltonian = openfermion.jordan_wigner(hamiltonian_ferm_op)
    hamiltonian_pauli = PauliwordOp.from_openfermion(qubit_hamiltonian)
    return hamiltonian_pauli

def generate_pyscf_data(geometry, basis, nroots):
    mol=gto.M(atom=geometry, basis=basis, symmetry=True)
    mf = scf.RHF(mol)
    mf.kernel()
    cisolver = fci.FCI(mol, mf.mo_coeff)

    e, fcivec = cisolver.kernel(nroots=nroots)

    for root_n in range(nroots):
        print(f'state {root_n} energy: ', e[root_n])
    return e, fcivec

opt_params_list=None
start_time = time.time()
print(f'Procedure begins at : {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))}')
for i in range(number_of_points):
    r = i*0.03+0.5; print('\nBondlength: ', r)#r = i*0.005-0.5, r begins from -0.5 don't work;

    geometry = [('Cl', (0, 0, r)), ('H', (0, 0, 0))]

    e, fcivec = generate_pyscf_data(geometry, basis, nroots=3)
    
    hamiltonian_pauli = generate_H_pauliword(geometry, basis, multiplicity, charge)

    _start_time = time.time(); print(f'Single point begins')
    
    QT = QubitTapering(hamiltonian_pauli)
    H_taper   = QT.taper_it(ref_state=hf_state)

    # cs_vqe = ContextualSubspace(H_taper, noncontextual_strategy='SingleSweep_magnitude', unitary_partitioning_method='LCU')
    cs_vqe = ContextualSubspace(H_taper, noncontextual_strategy='diag', unitary_partitioning_method='LCU')

    UCC_q = PauliwordOp.from_dictionary(data_dict['data']['auxiliary_operators']['UCCSD_operator'])
    UCC_taper = QT.taper_it(aux_operator=UCC_q)#taper consistently with the identified symmetry of the Hamiltonian

    print('The Hamiltonian before contextualized has {} qubits\n'.format(H_taper.n_qubits))

    cs_vqe.update_stabilizers(n_qubits = n_qubits_used, strategy='aux_preserving', aux_operator=UCC_taper)

    H_cs = cs_vqe.project_onto_subspace()

    UCC_cs = cs_vqe.project_onto_subspace(UCC_taper)

    qc=QuantumCircuit(n_qubits_used)
    if i == 0:
        hf_cs = cs_vqe.project_state(QT.tapered_ref_state)# Hartree-Fock in contextual subspace
        count_ones = np.sum(hf_cs.state_matrix[0] == 1)

        for occ in range(count_ones):
            qc.x(n_qubits_used-occ-1)#excite the same number of particles as the Hartree-Fock state
        num_gates = 1000
        thetas = generate_parameters('θ', num_gates)
        
        symm_preserve_ansatz=N_repeat_theta_xyz_phase_shift(qc, thetas, repeats=5)

        symm_preserve_ansatz=qc.copy()
        initial_circuit = symm_preserve_ansatz.copy()
    else:
        symm_preserve_ansatz=initial_circuit.copy()

    print('number of parameters: ',symm_preserve_ansatz.num_parameters)

    #Generate the all zero state on N qubits
    zero_state = QuantumState.zero(n_qubits_used)

    # vqd = VQD(observable=H_cs,ref_state=hf_cs,excitation_ops=UCC_cs,excited_order=1, beta_list=[10], method='BFGS')
    vqd = VQD(observable=H_cs,ref_state=zero_state,ansatz_circuit=symm_preserve_ansatz,
              excited_order=1, beta_list=[10], method='BFGS',initial_para='random')

    vqd.verbose = 1
    vqd_result, interim_data,opt_params_list = vqd.run()
    his_data_e0=interim_data[0]
    his_data_e1=interim_data[1]
    
    vqd_e0 = vqd_result[0]['fun']
    vqd_e1 = vqd_result[1]['energy']
    
    error_fci_e0=abs(vqd_e0-(e[0]))
    error_fci_e1=abs(vqd_e1-(e[1]))

    print(f'Converged VQD e0 energy = {vqd_e0} with FCI error {error_fci_e0}')
    print(f'Converged VQD e1 energy = {vqd_e1} with FCI error {error_fci_e1}\n')

    _end_time = time.time(); print(f'Single point ends')
    _total_time = _end_time - _start_time
    print(f'Single point\'s total_time: {_total_time:.2f} seconds, {_total_time/60:.2f} minutes\n')

    _his_data={f'r={r}':{'e0':his_data_e0,'e1':his_data_e1}}
    his_data.append(_his_data)

end_time = time.time()
total_time = end_time - start_time
print(f'Procedure ends at : {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))}')
print(f'total_time: {total_time:.2f} seconds, {total_time/60:.2f} minutes\n\n')

# with open('his_data.pkl', 'wb') as f:
#     pickle.dump(his_data, f)