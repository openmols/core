"""Turn a ligand with a covalent warhead into its adduct with a residue side
chain, ready for Meeko's tethered covalent docking.

    python covalent_prep.py ligand.sdf adduct.sdf acrylamide CYS

The adduct starts with a short stand-in for the residue, from CA out to the
nucleophile (for cysteine: CH3 = CA, CH2 = CB, S = SG). Meeko superposes the
first two carbons on the real residue's CA and CB, and the rest of the
molecule is what gets docked.
"""
import sys

from rdkit import Chem
from rdkit.Chem import AllChem

# Nucleophilic atom plus the side chain back to CA (CA is the CH3).
NUCLEOPHILE = {
    "CYS": "S[CH2][CH3]",
    "LYS": "N[CH2][CH2][CH2][CH2][CH3]",
    "SER": "O[CH2][CH3]",
    "TYR": "Oc1ccc(cc1)[CH2][CH3]",
}

# warhead -> reaction from the free ligand to the adduct; {NU} is the
# nucleophile above, bonded where the warhead reacts.
REACTIONS = {
    # Michael addition to the beta carbon.
    "acrylamide": "[CH2:1]=[CH:2][C:3](=[O:4])[N:5]>>[CH2:1]({NU})[CH2:2][C:3](=[O:4])[N:5]",
    "vinylsulfonamide": "[CH2:1]=[CH:2][S:3](=[O:4])(=[O:5])[N:6]>>[CH2:1]({NU})[CH2:2][S:3](=[O:4])(=[O:5])[N:6]",
    # Displacement of chloride.
    "chloroacetamide": "[Cl][CH2:1][C:2](=[O:3])[N:4]>>[CH2:1]({NU})[C:2](=[O:3])[N:4]",
    # Sulfonyl fluoride exchange: sulfonamide (Lys) or sulfonate ester (Ser, Tyr).
    "sulfonylfluoride": "[S:1](=[O:2])(=[O:3])[F]>>[S:1](=[O:2])(=[O:3]){NU}",
    # Aldehyde and lysine amine condense to a Schiff base.
    "aldehyde": "[CX3H1:1](=[O])[#6:2]>>[CH:1](={NU})[#6:2]",
}

# Which warheads can react with which residue.
ALLOWED = {
    "CYS": {"acrylamide", "vinylsulfonamide", "chloroacetamide"},
    "LYS": {"acrylamide", "vinylsulfonamide", "chloroacetamide", "sulfonylfluoride", "aldehyde"},
    "SER": {"sulfonylfluoride"},
    "TYR": {"sulfonylfluoride"},
}


def main() -> int:
    if len(sys.argv) != 5 or sys.argv[3] not in REACTIONS or sys.argv[4].upper() not in NUCLEOPHILE:
        print(
            "usage: covalent_prep.py in.sdf out.sdf {%s} {%s}" % ("|".join(REACTIONS), "|".join(NUCLEOPHILE)),
            file=sys.stderr,
        )
        return 2
    src, dst, warhead, residue = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].upper()
    if warhead not in ALLOWED[residue]:
        print("a %s group does not react with %s" % (warhead, residue), file=sys.stderr)
        return 2
    mol = Chem.MolFromMolFile(src, removeHs=True)
    if mol is None:
        print("cannot read %s" % src, file=sys.stderr)
        return 2
    reaction = AllChem.ReactionFromSmarts(REACTIONS[warhead].replace("{NU}", NUCLEOPHILE[residue]))
    products = reaction.RunReactants((mol,))
    if not products:
        print("no %s group found in the ligand" % warhead, file=sys.stderr)
        return 3
    adduct = products[0][0]
    Chem.SanitizeMol(adduct)
    adduct = Chem.AddHs(adduct)
    if AllChem.EmbedMolecule(adduct, randomSeed=7) != 0:
        print("could not build 3D coordinates for the adduct", file=sys.stderr)
        return 4
    try:
        AllChem.MMFFOptimizeMolecule(adduct)
    except Exception:  # an unparameterised group is not fatal
        pass
    writer = Chem.SDWriter(dst)
    writer.write(adduct)
    writer.close()
    print("adduct: %s" % Chem.MolToSmiles(Chem.RemoveHs(adduct)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
