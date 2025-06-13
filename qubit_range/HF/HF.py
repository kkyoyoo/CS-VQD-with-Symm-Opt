import numpy as np
import json


from symmer.operators import PauliwordOp, QuantumState
from symmer.projection import QubitTapering
from symmer.projection import ContextualSubspace

from evolution import VQD

from pyscf import gto, scf, fci


##############################
#Pyscf data
##############################


mol = gto.M(
    atom = '''
    H 0 0 0.099457
    F  0 0 -0.895109
    ''',
    basis = 'sto-3g',
    symmetry = True,
)


mf = scf.RHF(mol)
mf.kernel()


cisolver = fci.FCI(mol, mf.mo_coeff)
e, fcivec = cisolver.kernel()


print('Ground state energy: ', e)


nroots = 3  
e, fcivec = cisolver.kernel(nroots=nroots)


for i in range(nroots):
    print(f'state {i} energy: ', e[i])

############################################################################################



##############################
#Smmyer data
##############################
filename = '../../hamiltonian_data/HF_STO-3G_SINGLET_JW.json'

with open(filename, 'r') as infile:
    data_dict = json.load(infile)



fci_energy = data_dict['data']['calculated_properties']['FCI']['energy']
hf_state   = QuantumState(np.asarray(data_dict['data']['hf_array'])) # Hartree-Fock state
hf_energy  = data_dict['data']['calculated_properties']['HF']['energy']
H = PauliwordOp.from_dictionary(data_dict['hamiltonian'])

##############################
#VQD    excited state
##############################
import time

start_time = time.time()
print(f'Procedure begins at : {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))}')

QT = QubitTapering(H)
H_taper   = QT.taper_it(ref_state=hf_state)

cs_vqe = ContextualSubspace(H_taper, noncontextual_strategy='SingleSweep_magnitude', unitary_partitioning_method='LCU')

UCC_q = PauliwordOp.from_dictionary(data_dict['data']['auxiliary_operators']['UCCSD_operator'])
UCC_taper = QT.taper_it(aux_operator=UCC_q)#taper consistently with the identified symmetry of the Hamiltonian

max_qubits = 8
error_fci_e0={i:0 for i in range(1,max_qubits+1)}
error_fci_e1={i:0 for i in range(1,max_qubits+1)}

his_data_e0={i:0 for i in range(1,max_qubits+1)}
his_data_e1={i:0 for i in range(1,max_qubits+1)}

print('The Hamiltonian before contextualized has {} qubits\n'.format(H_taper.n_qubits))

for num_qubit in range(1, max_qubits+1):
    print(f'Number of qubits: {num_qubit}')
    cs_vqe.update_stabilizers(n_qubits = num_qubit, strategy='aux_preserving', aux_operator=UCC_taper)

    H_cs = cs_vqe.project_onto_subspace()

    UCC_cs = cs_vqe.project_onto_subspace(UCC_taper)
    hf_cs = cs_vqe.project_state(QT.tapered_ref_state)# Hartree-Fock in contextual subspace

    vqd = VQD(observable=H_cs,ref_state=hf_cs,excitation_ops=UCC_cs,excited_order=1, beta_list=[10], method='BFGS')

    vqd.verbose = 1
    vqd_result, interim_data = vqd.run()
    his_data_e0[num_qubit]=interim_data[0]
    his_data_e1[num_qubit]=interim_data[1]
    
    vqd_e0 = vqd_result[0]['fun']
    vqd_e1 = vqd_result[1]['energy']
    
    error_fci_e0[num_qubit]=abs(vqd_e0-(e[0]))
    error_fci_e1[num_qubit]=abs(vqd_e1-(e[1]))

    print(f'Converged VQD e0 energy = {vqd_e0} with FCI error {error_fci_e0[num_qubit]}')
    print(f'Converged VQD e1 energy = {vqd_e1} with FCI error {error_fci_e1[num_qubit]}\n\n')


end_time = time.time()
print(f'Procedure ends at : {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))}')

total_time = end_time - start_time
print(f'total_time: {total_time:.2f} seconds, {total_time/60:.2f} minutes')