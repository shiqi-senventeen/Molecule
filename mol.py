from rdkit import Chem
# from rdkit.Chem import Draw
# mol = Chem.MolFromSmiles('C[C@H](O)c1ccccc1')
# # Draw.MolToImage(mol, size=(150,150), kekulize=True)
# Draw.ShowMol(mol, size=(500,500), kekulize=True)
# Draw.MolToFile(mol, 'data/output.png', size=(500, 500))
from rdkit import Chem
import torch
from Transformer import *
import numpy
from smiles_encodding import smiles_to_transformer_input

# m3d = Chem.MolFromSmiles('CNC(=O)N(N(CCCl)S(C)(=O)=O)S(C)(=O)=O')
# m3d = Chem.AddHs(m3d)
# AllChem.EmbedMolecule(m3d, randomSeed=3)
# AllChem.MMFFOptimizeMolecule(m3d)
# Draw.MolToImage(m3d, size=(250,250))
# Draw.ShowMol(m3d, size=(500,500), kekulize=True)
'''
从配体sdf文件提取smiles序列
'''
mol_1 = Chem.MolFromMolFile(r'D:\Pyprograms\Drugs\data\1a0q_ligand.sdf',sanitize=False)
smi_1 = Chem.MolToSmiles(mol_1)
# print(smi_1)

mol_2 = Chem.MolFromMolFile(r'D:\Pyprograms\Drugs\data\1a0t_ligand.sdf',sanitize=False)
smi_2 = Chem.MolToSmiles(mol_2)
# print(smi_2)
print(smi_1)
print("读取smiles成功")
# 读取smiles成功
# '''
# 将smiles输入transformer，得到embedding
# '''
#
# smi_list = []
# for char in smi_1:
#     x = ord(char)
#     tensor_x = torch.tensor(x)
#     smi_list.append(tensor_x)
# print(smi_list)


smi_1_encodding = smiles_to_transformer_input(smi_1)
print(smi_1_encodding)
print(smi_1_encodding.size())
# smi_2_encodding = smiles_encodding.smiles_to_transformer_input(smi_2)
# print(smi_2_encodding)
# model_pre = PrepareForMultiHeadAttention(1,8,8,bias=True)
# output = model_pre(smi_1_encodding)
smi_1_encodding = smi_1_encodding.to(dtype=torch.float32)
print(smi_1_encodding)
model_pre = PrepareForMultiHeadAttention(45,4,9,bias=False)
output = model_pre(smi_1_encodding)