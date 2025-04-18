import pickle
import networkx as nx
from rdkit import Chem
from collections import defaultdict
import json
from networkx.readwrite import node_link_data
import numpy as np


# This code generates the graph_data.json and raw_data.json files for the challenges,
# prints information in shown in training_data_info.txt
# and fixes some errors present in the oriGinal dataset

# This code is not needed for the challenges and can be ignored
# it is here for transparencey

# However it can be used and modified if you think more node and edge features would 
# help your GNN predictions, go to line 323 to see how this can be done

# Original data stored in pickle files
mf_file_path = 'database_2023_05_30/MF_list.pkl'
be_file_path = 'database_2023_05_30/BE_list.pkl'
cl_file_path = 'database_2023_05_30/Core_level_list.pkl'
sm_file_path = 'database_2023_05_30/Isomeric_SMILES_list.pkl'
cf_file_path = 'database_2023_05_30/CF_list.pkl'
be_comm_file_path = 'database_2023_05_30/BE_comment_list.pkl'
rf_comm_file_path = 'database_2023_05_30/Reference_comment_list.pkl'
comm_file_path    = 'database_2023_05_30/Comment_list.pkl'

# Load the pickle file
with open(mf_file_path, 'rb') as file:
    mf_data = pickle.load(file)
with open(be_file_path, 'rb') as file:
    be_data = pickle.load(file)
with open(cl_file_path, 'rb') as file:
    cl_data = pickle.load(file)
with open(sm_file_path, 'rb') as file:
    sm_data = pickle.load(file)
with open(cf_file_path, 'rb') as file:
    cf_data = pickle.load(file)

# Load pickle comm files
with open(be_comm_file_path, 'rb') as file:
    be_comm_data = pickle.load(file)
with open(rf_comm_file_path, 'rb') as file:
    rf_comm_data = pickle.load(file)
with open(comm_file_path, 'rb') as file:
    comm_data = pickle.load(file)

mf_data_clean = []
be_data_clean = []
cl_data_clean = []
sm_data_clean = []
cf_data_clean = []

be_comm_data_clean = []
rf_comm_data_clean = []
comm_data_clean    = []

none_data_points = []

mf_non_count = 0
be_non_count = 0
cl_non_count = 0
sm_non_count = 0

for i,val in enumerate(mf_data):
    if val == None and i not in none_data_points:
        none_data_points.append(i)
        mf_non_count += 1
for i,val in enumerate(be_data):
    if val == None and i not in none_data_points:
        none_data_points.append(i)
        be_non_count += 1
for i,val in enumerate(cl_data):
    if val == None and i not in none_data_points:
        none_data_points.append(i)
        cl_non_count += 1
for i,val in enumerate(sm_data):
    if val == None and i not in none_data_points:
        none_data_points.append(i)
        sm_non_count += 1

for i,val in enumerate(mf_data):
    if i not in none_data_points:
        mf_data_clean.append(val)
        be_data_clean.append(be_data[i])
        cl_data_clean.append(cl_data[i])
        sm_data_clean.append(sm_data[i])
        cf_data_clean.append(cf_data[i])

        be_comm_data_clean.append(be_comm_data[i])
        rf_comm_data_clean.append(rf_comm_data[i])
        comm_data_clean.append(comm_data[i])

# Errors in the original data I had to manually find and fix
for i in range(len(be_data_clean)):

    if cl_data_clean[i] == "C":
        cl_data_clean[i] = "C 1s"

    if cl_data_clean[i] == 'F':
        cl_data_clean[i] = "F 1s"

    if cl_data_clean[i] == "I 4S":
    #individual_molecule 27 
        cl_data_clean[i] = "I 4s"

    if cl_data_clean[i] == "3p3/2":
    #individual_molecule 45
        cl_data_clean[i] = "Br 3p3/2"

    if cl_data_clean[i] == "Ca 2S":
    #individual_molecule 509
        cl_data_clean[i] = "Ca 2s"

    if cl_data_clean[i] == "Cd 3p5/2":
    #individual_molecule 510
        cl_data_clean[i] = "Cd 3p"

    if cl_data_clean[i] == "Si 2p3":
    #individual_molecule 559
        cl_data_clean[i] = "Si 2p3/2"

    if cl_data_clean[i] == "S 293/2":
    #individual_molecule 810
        cl_data_clean[i] = "S 2p3/2"

    if sm_data_clean[i] == "ULUNHHQLVSLOBA-UHFFFAOYSA-N":
    #individual_molecule 84 
        sm_data_clean[i] = "C[GeH2]Cl" 

    if sm_data_clean[i] == "C[GeH2]Se[GeH2]C":
    #individual_molecule 184
        sm_data_clean[i] = "C[GeH2][Se][GeH2]C"

    if sm_data_clean[i] == "C[GeH2]Te[GeH2]C":
    #individual_molecule 185
        sm_data_clean[i] = "C[GeH2][Te][GeH2]C"

    if sm_data_clean[i] == "C1C=CC=[C]1.C1C=CC=[C]1.[Fe":
    #individual_molecule 465
        sm_data_clean[i] = "C1C=CC=[C]1.C1C=CC=[C]1.[Fe]"

    if sm_data_clean[i] == "InChI=1S/C4H4.6CO.2Fe/c1-3-4-2;6*1-2;;/h1-4H;;;;;;;;":
    #individual_molecule 466
        sm_data_clean[i] = "[Fe](C#O)(C#O)(C#O)C=CC=C[Fe](C#O)(C#O)(C#O)"

    if sm_data_clean[i] == "InChI=1S/3C5H2F6O2.V/c3*6-4(7,8)2(12)1-3(13)5(9,10)11;/h3*1,12H;/q;;;+3/p-3/b3*2-1+;":
    #individual_molecule 503
        sm_data_clean[i] = "C(=C(\O)/C(F)(F)F)\C(=O)C(F)(F)F.C(=C(\O)/C(F)(F)F)\C(=O)C(F)(F)F.C(=C(\O)/C(F)(F)F)\C(=O)C(F)(F)F.[V]"

    if sm_data_clean[i] == "C1=CC(=C2C=CC=C2)C=C1.C1=CC(=C2C=CC=C2)C=C1.[Fe].[Fe":
    #individual_molecule 506
        sm_data_clean[i] = "C1=CC(=C2C=CC=C2)C=C1.C1=CC(=C2C=CC=C2)C=C1.[Fe].[Fe]"

    if sm_data_clean[i] == "C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-](F)(F)F.[Mn]":
    #earlier individual_molecule 589
        sm_data_clean[i] = "[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-](F)(F)F.[Mn]"

    if sm_data_clean[i] == "Ti](I)(I)(I)I":
    #earlier individual_molecule 642
        sm_data_clean[i] = "[Ti](I)(I)(I)I"

    if sm_data_clean[i] == "[N+](=O)([O-])[O-].[N+](=O)([O-])[O-].[N+](=O)([O-])[O-].[N+](=O)([O-])[O-].[Sn+4":
    #earlier individual_molecule 872
        sm_data_clean[i] = "[N+](=O)([O-])[O-].[N+](=O)([O-])[O-].[N+](=O)([O-])[O-].[N+](=O)([O-])[O-].[Sn+4]"

with open("training_molecules.txt", "r") as f:
    training_molecules = [line.strip() for line in f]

# Initialize an empty list to store unique dictionaries
atom_types = []

# Iterate through the input list
for val in mf_data_clean:
    for key in val.keys():
    # Add the dictionary to unique_list if it's not already there
        if key not in atom_types:
            atom_types.append(key)

# Print the result
print(f"Numer of atom types: {len(atom_types)}")
print("atom_types:")
print(atom_types)
print("")

atom_type_numbers = {}
for n in atom_types:
    atom_type_numbers[n] = 0 
    for i,vali in enumerate(training_molecules):

        for j,valj in enumerate(sm_data_clean):

            if vali == valj:
                    
                for key in mf_data_clean[j].keys():
                    if key == n:
                        atom_type_numbers[n] += mf_data_clean[j][key]  

print("Number of each atom type:")
print(atom_type_numbers)
print(" ")

orbital_type_numbers = {}

# Initialize dictionary keys by extracting orbitals from cl_data_clean
for cl in cl_data_clean:
    parts = cl.split(maxsplit=1)  # Expecting "Atom Orbital"
    if len(parts) == 2:
        orbital = parts[1]
    else:
        orbital = "1s"  # Default orbital if not provided.
    orbital_type_numbers[orbital] = 0

# Iterate over the molecules and matching smiles to count the occurrences
for i, vali in enumerate(training_molecules):
    for j, valj in enumerate(sm_data_clean):
        if vali == valj:
            parts = cl_data_clean[j].split(maxsplit=1)
            if len(parts) == 2:
                orbital = parts[1]
            else:
                orbital = "1s"
            orbital_type_numbers[orbital] += 1

print(f"Numer of orbital types: {len(orbital_type_numbers)}")
print("Number of each orbital type:")
print(orbital_type_numbers)
print(" ")

molec_be = {}
molec_cl = {}
molec_cf = {}
molec_mf = {}
molec_sm = {}
og_id    = {}

molec_be_comm = {}
molec_rf_comm = {}
molec_comm    = {}

raw_data = {}
graph_data = {}

single_atom_count = 0

total_be = []

fail_count = 0

PRINT_GRAPHS = False

for i,vali in enumerate(training_molecules):

    molec_be[i] = []
    molec_cl[i] = []
    molec_cf[i] = []
    molec_mf[i] = []
    molec_sm[i] = []
    og_id[i]    = []

    molec_be_comm[i] = []
    molec_rf_comm[i] = []
    molec_comm[i]    = []

    for j,valj in enumerate(sm_data_clean):
            
        if vali == valj:

            molec_be[i].append(be_data_clean[j])
            molec_cl[i].append(cl_data_clean[j])
            molec_cf[i].append(cf_data_clean[j])
            molec_mf[i].append(mf_data_clean[j])
            molec_sm[i].append(sm_data_clean[j])

            molec_be_comm[i].append(be_comm_data_clean[j])
            molec_rf_comm[i].append(rf_comm_data_clean[j])
            molec_comm[i].append(comm_data_clean[j])

            total_be.append(be_data_clean[j])

    ## additional dataset error correction 
    if molec_sm[i][0] == "[N+]([O-])(F)(F)F":
        molec_cl[i][0] = "O 1s"

    if molec_sm[i][0] == "S1SSSSSSS1":
        molec_cl[i][0] = "S 2s"

    if molec_sm[i][0] == "[Ti]":
        molec_sm[i][0] = "[Tl]"
        molec_cl[i][0] = "Tl 4f7/2"
        vali = "[Tl]"


    raw_data[molec_sm[i][0]] = {"chemical_formula": [], "orbitals": [], "binding_energies": [], "references": []}

    raw_data[molec_sm[i][0]]["chemical_formula"] = molec_cf[i]
    raw_data[molec_sm[i][0]]["orbitals"]         = molec_cl[i]
    raw_data[molec_sm[i][0]]["binding_energies"] = molec_be[i]
    raw_data[molec_sm[i][0]]["references"]       = molec_rf_comm[i]

    core_level_dict = defaultdict(list)
    for core_level, energy in zip(molec_cl[i], molec_be[i]):
        parts = core_level.split(maxsplit=1)  # Split into atom type and orbital
        atom_type, orbital = parts
        core_level_dict[atom_type].append((orbital, energy))

    # Group energies by (atom_type, orbital) pair.
    grouped_energies = defaultdict(list)
    for atom_type, value_list in core_level_dict.items():
        for orbital, energy in value_list:
            grouped_energies[(atom_type, orbital)].append(energy)

    # Compute averages if there are multiple values; otherwise, use the original value.
    averages = {}
    for key, energies in grouped_energies.items():
        if len(energies) > 1:
            averages[key] = sum(energies) / len(energies)
        else:
            averages[key] = energies[0]

    # Update the core_level_dict with the averaged values.
    for atom_type, value_list in core_level_dict.items():
        seen_orbitals = set()
        unique_values = []
        for orbital, _ in value_list:
            if orbital not in seen_orbitals:
                seen_orbitals.add(orbital)
                # Assign the averaged energy (or the original if only one exists)
                unique_values.append((orbital, averages[(atom_type, orbital)]))
        core_level_dict[atom_type] = unique_values

    smiles = vali

    # Sanitize=False means all molecules in "training_molecules" can be made into graphs
    # however less features can be included in the graphs
    # if you want to test weher a smaller dataset with more features could give better results
    # try:
    # skipping the molecules that give errors with sanitize=True and uncomment the inclusion of additional features
    # in G.add_node( ) and G.add_edge( )

    mol = Chem.MolFromSmiles(smiles, sanitize=False)

    # Count how many fail if sanitize=True
    if mol is None:
    
        fail_count += 1
        continue

    # Create node features and graph
    G = nx.Graph()

    orbitals_list = []
    binding_energies_list = []

    atom_count = 0
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_symbol = atom.GetSymbol()
        orbitals = []
        binding_energies = []

        atom_count += 1

        # Add orbitals and binding energies for the atom type
        if atom_symbol in core_level_dict:
            for orbital, energy in core_level_dict[atom_symbol]:
                orbitals.append(orbital)
                binding_energies.append(energy)

        # Pad to the maximum number of orbitals for this atom type
        max_orbs = max(len(v) for v in core_level_dict.values())
        #while len(orbitals) < max(len(v) for v in core_level_dict.values()):
        while len(orbitals) < max_orbs:
            orbitals.append(-1)  # No orbital information
            binding_energies.append(-1)  # No binding energy

        orbitals_list.append(orbitals)
        binding_energies_list.append(binding_energies)

        # Add the atom as a node to the graph
        G.add_node(
            atom_idx,
            atom_type=atom_symbol,
            formal_charge=atom.GetFormalCharge(),
            #aromatic = 1 if atom.GetIsAromatic() else 0 , # binary
            #hybridization = int(atom.GetHybridization()), # use as integer
            #hybridization = str(atom.GetHybridization()),# or as string and use one-hot (determine number of types)
            #radical_electrons=int(atom.GetNumRadicalElectrons()), # integer
            orbitals=orbitals,
            binding_energies=binding_energies
        )

    if atom_count == 1:
        single_atom_count += 1
    
    # Add edges based on RDKit bonds
    for bond in mol.GetBonds():
        start_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()

        G.add_edge(start_idx, end_idx, 
                    bond_type=str(bond_type),
                    #stereo=int(bond.GetStereo()), 
                    #aromatic= 1 if bond.GetIsAromatic() else 0,
                    #conjugated=1 if bond.GetIsConjugated() else 0,
        )

    if PRINT_GRAPHS == True:

        print(i)
        print("Graph Nodes:")
        for node, data in G.nodes(data=True):
                print(f"Node {node}: {data}")
        print(" ")
        print("\nGraph Edges:")
        for u, v, data in G.edges(data=True):  # Correct unpacking
                print(f"Edge ({u}, {v}): {data}")
        print(" ")

    graph_data[molec_sm[i][0]] = G

with open("raw_data.json", "w") as f:
    json.dump(raw_data, f, indent=2)

graph_data_json = {
    smiles: node_link_data(G, edges="edges") for smiles, G in graph_data.items()
}
with open("graph_data.json", "w") as f:
    json.dump(graph_data_json, f, indent=2)

print(f"failed rdkit: {fail_count}")
print(f"Total number of binding energies: {len(total_be)}\n")

print(f"Total number of Graphs: {len(graph_data)}")

print(f"Number of single atoms: {single_atom_count}")
print(f"Number of molecules: {len(graph_data) - single_atom_count}")


