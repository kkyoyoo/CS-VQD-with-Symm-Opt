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



n_qubits_used = 6
print(f'Number of qubits used: {n_qubits_used}')

#generate Hartree-Fock state from Symmer
filename = '../../hamiltonian_data/HCl_STO-3G_SINGLET_JW.json'
with open(filename, 'r') as infile:
    data_dict = json.load(infile)
hf_state   = QuantumState(np.asarray(data_dict['data']['hf_array']))


basis = 'sto-3g'; multiplicity = 1; charge = 0
number_of_points=50
his_data=[]

start_time = time.time()
print(f'Procedure begins at : {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))}')
for i in range(number_of_points):
    r = i*0.03+0.5; print('\nBondlength: ', r)#r = i*0.005-0.5, r begins from -0.5 don't work;
    geometry = [('H', (0, 0, 0)), ('Cl', (0, 0, r))]

    #pySCF data
    mol=gto.M(atom=geometry, basis=basis, symmetry=True)
    mf = scf.RHF(mol)
    mf.kernel()
    cisolver = fci.FCI(mol, mf.mo_coeff)


    nroots = 3  
    e, fcivec = cisolver.kernel(nroots=nroots)

    for i in range(nroots):
        print(f'state {i} energy: ', e[i])
    
    # generate Hamiltonian from openfermion -> symmer
    hamiltonian = openfermionpyscf.generate_molecular_hamiltonian(geometry, basis, multiplicity, charge)
    hamiltonian_ferm_op = openfermion.get_fermion_operator(hamiltonian)
    qubit_hamiltonian = openfermion.jordan_wigner(hamiltonian_ferm_op)
    hamiltonian_pauli = PauliwordOp.from_openfermion(qubit_hamiltonian)

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
    hf_cs = cs_vqe.project_state(QT.tapered_ref_state)# Hartree-Fock in contextual subspace

    vqd = VQD(observable=H_cs,ref_state=hf_cs,excitation_ops=UCC_cs,excited_order=1, beta_list=[10], method='BFGS')

    vqd.verbose = 1
    vqd_result, interim_data = vqd.run()
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

with open('his_data.pkl', 'wb') as f:
    pickle.dump(his_data, f)