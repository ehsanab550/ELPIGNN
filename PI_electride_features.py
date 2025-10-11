"""
@author: Ehsanab
"""

import os
import glob
import numpy as np
import pandas as pd
import warnings
from scipy.spatial import distance
from pymatgen.core import Structure
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
#from pymatgen.core.periodic_table import Element
#from scipy.stats import describe
from matminer.featurizers.conversions import StrToComposition, CompositionToOxidComposition, StructureToComposition
from matminer.featurizers.composition import OxidationStates, Stoichiometry, BandCenter, AtomicOrbitals, TMetalFraction, YangSolidSolution, AtomicPackingEfficiency, ValenceOrbital, IonProperty
from matminer.featurizers.structure import SiteStatsFingerprint, StructuralHeterogeneity, ChemicalOrdering, MaximumPackingEfficiency
from matminer.featurizers.base import MultipleFeaturizer

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Define file paths
poscar_path = r"/ele_PO&OSZ/*/separatePOSCAR/*/*/POSCAR"
oszicar_path = r"/ele_PO&OSZ/*/separatePOSCAR/*/*/OSZICAR"

# Get file lists
filePOS = glob.glob(poscar_path)
filePOS = list(filter(lambda f: os.stat(f).st_size > 0, filePOS))
fileOSZ = glob.glob(oszicar_path)
fileOSZ = list(filter(lambda f: os.stat(f).st_size > 0, fileOSZ))

print(f"Found {len(filePOS)} POSCAR files and {len(fileOSZ)} OSZICAR files")

# Initialize lists to store data
formula = []
reduced_formula = []
comp_formula = []
spacegroup = []
E_total = []
natomnum_cell = []
formula_dic = []
structures = []
dens = []
file_paths = []
second_dirs = []
material_names = []
indicators = []
formulas = []
third_dirs = []

# Process POSCAR files
for file_name in filePOS:
    try:
        # Extract directory information
        path_parts = file_name.split('/')
        second_dir = path_parts[-3]  # The YES_... or NO_... directory
        third_dir = path_parts[-2]   # The EA... directory
        
        # Extract indicator and material name from second_dir
        if second_dir.startswith('YES_'):
            indicator = 'YES'
            material_name = second_dir[4:]
        elif second_dir.startswith('NO_'):
            indicator = 'NO'
            material_name = second_dir[3:]
        else:
            indicator = 'UNKNOWN'
            material_name = second_dir
        
        # Parse the structure
        with open(file_name, 'r') as f:
            content = f.read()
        struct = Structure.from_str(content, fmt="poscar")
        finder = SpacegroupAnalyzer(struct)
        spaceG = finder.get_space_group_number()
        Formula = struct.composition.formula
        Formula_comp = struct.composition.reduced_composition
        Formula_redu = struct.composition.reduced_formula
        num_compos = struct.composition.num_atoms
        dic_formula = struct.composition.as_dict()
        densi = struct.density
        
        # Append all data
        formula.append(Formula)
        reduced_formula.append(Formula_redu)
        comp_formula.append(Formula_comp)
        natomnum_cell.append(num_compos)
        spacegroup.append(spaceG)
        formula_dic.append(dic_formula)
        structures.append(struct)
        dens.append(densi)
        file_paths.append(file_name)
        second_dirs.append(second_dir)
        material_names.append(material_name)
        indicators.append(indicator)
        formulas.append(material_name)  # Using material_name as formula
        third_dirs.append(third_dir)
    except Exception as e:
        print(f"Error processing {file_name}: {str(e)}")
        # Append placeholders for failed structure
        formula.append("")
        reduced_formula.append("")
        comp_formula.append("")
        natomnum_cell.append(0)
        spacegroup.append(0)
        formula_dic.append({})
        structures.append(None)
        dens.append(0)
        file_paths.append(file_name)
        second_dirs.append("")
        material_names.append("")
        indicators.append("")
        formulas.append("")
        third_dirs.append("")

# Process OSZICAR files
E_total = []
for filename in fileOSZ: 
    try:
        with open(filename) as fn:
            content = fn.read()
            # Find last energy value in OSZICAR
            energy_lines = [line for line in content.split('\n') if 'F=' in line]
            if energy_lines:
                last_energy_line = energy_lines[-1]
                energy = float(last_energy_line.split('F=')[-1].split()[0])
                E_total.append(energy)
            else:
                E_total.append(0)
                print(f"No energy found in {filename}")
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        E_total.append(0)

# Create DataFrame with basic information
dataE = pd.DataFrame({
    'second_dir': second_dirs,
    'material_name': material_names,
    'indicator': indicators,
    'formula': formulas,
    'third_dir': third_dirs,
    'structure': structures,
    'E_total': E_total,
    'SG': spacegroup,
    'formula_redu': reduced_formula,
    'formula_comp': comp_formula,
    'Natom/cell': natomnum_cell,
    'density': dens,
    'file_path': file_paths
})

# Remove the columns we don't need
dataE = dataE.drop(columns=['file_path', 'formula_comp'])

# Function to calculate electride-specific features
def calculate_electride_specific_features(struct):
    """
    Calculate electride-specific features not already in the dataset
    Focuses on void analysis, electron localization, and electride-specific properties
    """
    features = {}
    
    # 1. Advanced void analysis features (critical for electrides)
    features.update(calculate_void_features(struct))
    
    # 2. Electron localization features
    features.update(calculate_electron_localization_features(struct))
    
    # 3. Dimensionality and connectivity features
    features.update(calculate_dimensionality_features(struct))
    
    # 4. Electride-specific composition features
    features.update(calculate_electride_composition_features(struct))
    
    # 5. Bonding asymmetry features
    features.update(calculate_bonding_asymmetry_features(struct))
    
    return features

def calculate_void_features(struct):
    """Calculate void-related features critical for electride materials"""
    features = {}
    
    # Get atomic coordinates and radii
    coords = struct.cart_coords
    radii = []
    for site in struct:
        el = site.specie
        radius = el.atomic_radius or el.covalent_radius or 1.4
        radii.append(radius)
    
    # Create a grid to sample void spaces
    n_points = 300  # Reduced for performance
    void_radii = []
    
    # Sample random points in the unit cell
    for _ in range(n_points):
        # Generate a random point in the unit cell
        point = np.random.rand(3)
        cart_point = struct.lattice.get_cartesian_coords(point)
        
        # Calculate distance to nearest atom, adjusted for atomic radius
        min_dist = float('inf')
        for i, coord in enumerate(coords):
            dist = distance.euclidean(cart_point, coord) - radii[i]
            if dist < min_dist:
                min_dist = dist
        
        if min_dist > 0.1:  # Only consider significant voids
            void_radii.append(min_dist)
    
    # Calculate void statistics
    if void_radii:
        features['void_max_radius'] = max(void_radii)
        features['void_avg_radius'] = np.mean(void_radii)
        features['void_std_radius'] = np.std(void_radii)
        features['void_volume_fraction'] = len(void_radii) / n_points
    else:
        features['void_max_radius'] = 0
        features['void_avg_radius'] = 0
        features['void_std_radius'] = 0
        features['void_volume_fraction'] = 0
    
    return features

def calculate_electron_localization_features(struct):
    """Calculate features related to electron localization in voids"""
    features = {}
    
    # Use a simpler approach to estimate void centers
    # Create a grid of points and find those farthest from atoms
    n_grid = 12  # Reduced for performance
    grid_points = []
    
    for i in range(n_grid):
        for j in range(n_grid):
            for k in range(n_grid):
                point = [i/n_grid, j/n_grid, k/n_grid]
                grid_points.append(point)
    
    # Find points with maximum distance to any atom
    void_centers = []
    coords = struct.cart_coords
    
    for point in grid_points:
        cart_point = struct.lattice.get_cartesian_coords(point)
        min_dist = min(distance.euclidean(cart_point, coord) for coord in coords)
        if min_dist > 1.0:  # Threshold for void center
            void_centers.append(cart_point)
    
    # Calculate statistics about void distribution
    if len(void_centers) > 0:
        # Distance between voids
        void_distances = []
        for i in range(len(void_centers)):
            for j in range(i+1, min(i+5, len(void_centers))):  # Limit for performance
                void_distances.append(distance.euclidean(void_centers[i], void_centers[j]))
        
        features['void_center_count'] = len(void_centers)
        features['void_avg_distance'] = np.mean(void_distances) if void_distances else 0
        features['void_std_distance'] = np.std(void_distances) if void_distances else 0
        
        # Calculate geometric center instead of center of mass
        all_coords = np.array([site.coords for site in struct])
        geometric_center = np.mean(all_coords, axis=0)
        
        # Distance of voids from geometric center
        void_center_distances = [distance.euclidean(center, geometric_center) for center in void_centers]
        features['void_center_avg_distance'] = np.mean(void_center_distances) if void_center_distances else 0
        features['void_center_std_distance'] = np.std(void_center_distances) if void_center_distances else 0
    else:
        features['void_center_count'] = 0
        features['void_avg_distance'] = 0
        features['void_std_distance'] = 0
        features['void_center_avg_distance'] = 0
        features['void_center_std_distance'] = 0
    
    return features

def calculate_dimensionality_features(struct):
    """Calculate features related to structural dimensionality"""
    features = {}
    
    # Analyze connectivity using a simpler approach
    try:
        # Use a distance-based approach
        coords = struct.cart_coords
        elements = [site.specie for site in struct]
        
        # Calculate average bond length
        bond_lengths = []
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                # Only consider bonds between different elements
                if elements[i] != elements[j]:
                    dist = distance.euclidean(coords[i], coords[j])
                    if dist < 3.0:  # Reasonable bond length threshold
                        bond_lengths.append(dist)
        
        # Calculate connectivity statistics
        if bond_lengths:
            features['bond_length_avg'] = np.mean(bond_lengths)
            features['bond_length_std'] = np.std(bond_lengths)
            features['bond_length_max'] = max(bond_lengths)
            features['bond_length_min'] = min(bond_lengths)
        else:
            features['bond_length_avg'] = 0
            features['bond_length_std'] = 0
            features['bond_length_max'] = 0
            features['bond_length_min'] = 0
        
        # Dimensionality indicators
        lattice = struct.lattice
        features['aspect_ratio_ab'] = lattice.a / lattice.b
        features['aspect_ratio_ac'] = lattice.a / lattice.c
        features['aspect_ratio_bc'] = lattice.b / lattice.c
        
        # Volume per atom
        features['volume_per_atom'] = lattice.volume / len(struct)
        
    except Exception as e:
        print(f"Error in dimensionality features: {str(e)}")
        features['bond_length_avg'] = 0
        features['bond_length_std'] = 0
        features['bond_length_max'] = 0
        features['bond_length_min'] = 0
        features['aspect_ratio_ab'] = 0
        features['aspect_ratio_ac'] = 0
        features['aspect_ratio_bc'] = 0
        features['volume_per_atom'] = 0
    
    return features

def calculate_electride_composition_features(struct):
    """Calculate composition features specific to electride materials"""
    features = {}
    
    comp = struct.composition
    elements = comp.elements
    
    # Calculate electron donation potential
    electropositive_elements = [e for e in elements if e.X < 1.3]  # Highly electropositive
    electronegative_elements = [e for e in elements if e.X > 2.5]  # Highly electronegative
    
    features['electropositive_count'] = sum(comp[e] for e in electropositive_elements)
    features['electronegative_count'] = sum(comp[e] for e in electronegative_elements)
    features['ep_en_ratio'] = features['electropositive_count'] / max(1, features['electronegative_count'])
    
    # Calculate average electronegativity difference
    en_values = [e.X for e in elements for _ in range(int(comp[e]))]
    features['en_avg'] = np.mean(en_values)
    features['en_std'] = np.std(en_values)
    features['en_range'] = max(en_values) - min(en_values)
    
    # Calculate electron count metrics
    valence_electrons = []
    for e in elements:
        # Simple valence electron count (s + p for main group, d for transition metals)
        if e.is_transition_metal:
            valence_count = e.Z - 18  # Rough estimate for transition metals
        else:
            valence_count = e.group  # Main group elements
        valence_electrons.extend([valence_count] * int(comp[e]))
    
    features['valence_avg'] = np.mean(valence_electrons)
    features['valence_std'] = np.std(valence_electrons)
    
    return features

def calculate_bonding_asymmetry_features(struct):
    """Calculate features related to bonding asymmetry"""
    features = {}
    
    try:
        coords = struct.cart_coords
        elements = [site.specie for site in struct]
        bond_asymmetries = []
        
        for i in range(len(coords)):
            # Find nearest neighbors
            distances = []
            for j in range(len(coords)):
                if i != j:
                    dist = distance.euclidean(coords[i], coords[j])
                    if dist < 3.0:  # Reasonable bond length threshold
                        distances.append(dist)
            
            # Calculate bond length asymmetry for this atom
            if len(distances) > 1:
                avg_dist = np.mean(distances)
                asymmetry = np.std(distances) / avg_dist if avg_dist > 0 else 0
                bond_asymmetries.append(asymmetry)
        
        # Bond asymmetry statistics
        if bond_asymmetries:
            features['bond_asymmetry_avg'] = np.mean(bond_asymmetries)
            features['bond_asymmetry_std'] = np.std(bond_asymmetries)
            features['bond_asymmetry_max'] = max(bond_asymmetries)
        else:
            features['bond_asymmetry_avg'] = 0
            features['bond_asymmetry_std'] = 0
            features['bond_asymmetry_max'] = 0
            
    except Exception as e:
        print(f"Error in bonding asymmetry features: {str(e)}")
        features['bond_asymmetry_avg'] = 0
        features['bond_asymmetry_std'] = 0
        features['bond_asymmetry_max'] = 0
    
    return features

# Function to calculate physics-informed features
def calculate_electride_features(struct):
    """Calculate 35 physics-based features for electride materials"""
    features = {}
    
    # 1. Basic lattice features (8 features)
    latt = struct.lattice
    features['a'] = latt.a
    features['b'] = latt.b
    features['c'] = latt.c
    features['alpha'] = latt.alpha
    features['beta'] = latt.beta
    features['gamma'] = latt.gamma
    features['volume'] = latt.volume
    features['density'] = struct.density
    
    # 2. Composition-based features (5 features)
    comp = struct.composition
    features['num_atoms'] = len(struct)
    features['num_unique_elements'] = len(comp)
    features['avg_atomic_number'] = np.mean([e.Z for e in comp.elements])
    features['electropositive_sum'] = sum([comp[e] for e in comp if e.is_alkali or e.is_alkaline])
    features['anionic_elements'] = sum([comp[e] for e in comp if e.X > 2.5])
    
    # 3. Atomic size features (5 features)
    atomic_radii = []
    for site in struct:
        el = site.specie
        radius = el.atomic_radius or el.covalent_radius or 1.5
        atomic_radii.append(radius)
    
    features['max_atomic_radius'] = max(atomic_radii)
    features['min_atomic_radius'] = min(atomic_radii)
    features['avg_atomic_radius'] = np.mean(atomic_radii)
    features['std_atomic_radius'] = np.std(atomic_radii)
    features['atomic_radius_range'] = features['max_atomic_radius'] - features['min_atomic_radius']
    
    # 4. Packing efficiency features (2 features)
    atomic_volumes = sum([(4/3)*np.pi*(r**3) for r in atomic_radii])
    features['packing_efficiency'] = atomic_volumes / latt.volume
    features['free_volume_ratio'] = 1 - features['packing_efficiency']
    
    # 5. Bonding environment features (5 features)
    bond_lengths = []
    coord_numbers = []
    try:
        vnn = VoronoiNN()
        for i in range(len(struct)):
            env = vnn.get_nn_info(struct, i)
            coord_numbers.append(len(env))
            for neighbor in env:
                bond_lengths.append(neighbor['weight'])  # 'weight' is bond length
    except:
        # Fallback for structures where Voronoi fails
        pass
    
    features['avg_coord_num'] = np.mean(coord_numbers) if coord_numbers else 0
    features['min_bond_length'] = min(bond_lengths) if bond_lengths else 0
    features['max_bond_length'] = max(bond_lengths) if bond_lengths else 0
    features['avg_bond_length'] = np.mean(bond_lengths) if bond_lengths else 0
    if bond_lengths:
        features['bond_length_range'] = max(bond_lengths) - min(bond_lengths)
    else:
        features['bond_length_range'] = 0
    
    # 6. Electronic structure proxies (3 features)
    en_diff = []
    elements = [site.specie for site in struct]
    for i in range(len(elements)):
        for j in range(i+1, len(elements)):
            en_diff.append(abs(elements[i].X - elements[j].X))
    
    features['max_en_diff'] = max(en_diff) if en_diff else 0
    features['min_en_diff'] = min(en_diff) if en_diff else 0
    features['avg_en_diff'] = np.mean(en_diff) if en_diff else 0
    
    # 7. Structural complexity (1 feature)
    try:
        sga = SpacegroupAnalyzer(struct)
        features['symmetry_ops'] = len(sga.get_symmetry_operations())
    except:
        features['symmetry_ops'] = 1
    
    # 8. Cation-anion arrangement (1 feature) - FIXED CENTER OF MASS
    try:
        center_of_mass = struct.center_of_mass
        anion_distances = []
        for site in struct:
            if site.specie.X > 2.5:  # Anionic elements
                dist = np.linalg.norm(site.coords - center_of_mass)
                anion_distances.append(dist)
        features['anion_com_distance_avg'] = np.mean(anion_distances) if anion_distances else 0
    except:
        features['anion_com_distance_avg'] = 0
    
    # 9. Dimensionality proxies (2 features)
    features['c_over_a_ratio'] = latt.c / latt.a
    features['b_over_a_ratio'] = latt.b / latt.a
    
    # 10. Special electride features (3 features)
    features['metal_fraction'] = sum(comp[e] for e in comp if e.is_metal) / len(struct)
    features['low_x_metal_content'] = sum(comp[e] for e in comp if e.X < 1.5) 
    features['high_x_nonmetal_content'] = sum(comp[e] for e in comp if e.X > 2.5) 
    
    return features

# Function for matminer featurization
def featurize_data(final_df):
    """Apply composition and structure featurization to the DataFrame"""
    try:
        # Step 1: Convert formula string to Composition objects
        stc = StrToComposition()
        df = stc.featurize_dataframe(final_df, "formula", pbar=False, ignore_errors=True)
        
        # Add element count feature
        df['composition_count'] = df['composition'].apply(len)
        
        # Step 2: Convert to oxidation compositions
        df = CompositionToOxidComposition().featurize_dataframe(df, "composition", ignore_errors=True)
        
        # Step 3: Featurize with oxidation states
        os_feat = OxidationStates()
        df = os_feat.featurize_dataframe(df, "composition_oxid", ignore_errors=True)
        
        # Step 4: Composition-based featurization (without oxidation states)
        comp_featurizers = MultipleFeaturizer([
            Stoichiometry(),
            BandCenter(),
            AtomicOrbitals(),
            TMetalFraction(),
            YangSolidSolution(),
            AtomicPackingEfficiency(),
            ValenceOrbital(),
            IonProperty(fast=True)
        ])
        df = comp_featurizers.featurize_dataframe(df, "composition", ignore_errors=True)
        
        # Step 5: Structure-based featurization
        struct_featurizers = MultipleFeaturizer([
            SiteStatsFingerprint.from_preset("CoordinationNumber_ward-prb-2017"),
            StructuralHeterogeneity(),
            ChemicalOrdering(),
            MaximumPackingEfficiency()
        ])
        df = struct_featurizers.featurize_dataframe(df, "structure", ignore_errors=True)
        
        return df
    except Exception as e:
        print(f"Error in featurize_data: {str(e)}")
        # Return the original dataframe if featurization fails
        return final_df.copy()

# Calculate electride-specific features for each structure
electride_features = []
for i, struct in enumerate(structures):
    if struct is not None:
        try:
            features = calculate_electride_specific_features(struct)
            electride_features.append(features)
        except Exception as e:
            print(f"Error calculating electride features for structure {i}: {str(e)}")
            # Create empty features dict for failed structures
            electride_features.append({})
    else:
        electride_features.append({})

# Calculate physics-informed features for each structure
pi_features = []
for i, struct in enumerate(structures):
    if struct is not None:
        try:
            features = calculate_electride_features(struct)
            pi_features.append(features)
        except Exception as e:
            print(f"Error calculating PI features for structure {i}: {str(e)}")
            # Create empty features dict for failed structures
            pi_features.append({})
    else:
        pi_features.append({})

# Create DataFrames with features
electride_df = pd.DataFrame(electride_features)
pi_df = pd.DataFrame(pi_features)

# Apply matminer featurization
final_df = dataE.copy()
featurized_df = featurize_data(final_df)

# Check if featurization was successful
if featurized_df is None:
    print("Featurization failed, using original dataframe")
    featurized_df = final_df.copy()

# Combine all features
# First, identify and remove duplicate columns
common_columns = set(electride_df.columns) & set(featurized_df.columns)
electride_df = electride_df.drop(columns=common_columns, errors='ignore')

common_columns = set(pi_df.columns) & set(featurized_df.columns)
pi_df = pi_df.drop(columns=common_columns, errors='ignore')

common_columns = set(pi_df.columns) & set(electride_df.columns)
pi_df = pi_df.drop(columns=common_columns, errors='ignore')

# Combine all DataFrames
final_combined_df = pd.concat([featurized_df, electride_df, pi_df], axis=1)

# Reorder columns to put the requested columns first
requested_columns = ['second_dir', 'material_name', 'indicator', 'formula', 'third_dir', 'structure']
other_columns = [col for col in final_combined_df.columns if col not in requested_columns]
final_columns = requested_columns + other_columns
final_combined_df = final_combined_df[final_columns]

print(f"Total features before cleaning: {len(final_combined_df.columns)}")

# Save the intermediate result
intermediate_path = r"/ELECTRIDE_clean/electride_final_features.csv"
final_combined_df.to_csv(intermediate_path, index=False)
print(f"Saved intermediate dataset to {intermediate_path}")

# Now apply the column cleaning process
print("Starting column cleaning process...")

# Columns to always keep
keep_cols = ['second_dir', 'material_name', 'indicator', 'formula', 'third_dir', 'structure']
remaining_cols = [col for col in final_combined_df.columns if col not in keep_cols]

# Create a temporary DataFrame for processing other columns
temp_df = final_combined_df[remaining_cols].copy()

# 1. Remove non-numeric columns (except the ones we want to keep)
numeric_cols = []
for col in temp_df.columns:
    if temp_df[col].dtype in [np.int64, np.float64]:
        numeric_cols.append(col)
    else:
        # Try to convert to numeric
        try:
            temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
            if not temp_df[col].isnull().all():  # Only keep if conversion was successful for at least some values
                numeric_cols.append(col)
        except:
            # Remove column if cannot be converted to numeric
            continue

temp_df = temp_df[numeric_cols]

# 2. Remove constant columns (columns with only one unique value)
non_constant_cols = []
for col in temp_df.columns:
    if temp_df[col].nunique() > 1:
        non_constant_cols.append(col)

temp_df = temp_df[non_constant_cols]

# 3. Remove duplicate columns (columns with identical values)
unique_cols = []
seen_values = []
for col in temp_df.columns:
    col_values = temp_df[col].values
    # Create a hashable representation of the column values
    col_hash = tuple(col_values)
    if col_hash not in seen_values:
        seen_values.append(col_hash)
        unique_cols.append(col)

temp_df = temp_df[unique_cols]

# Combine with kept columns
final_cleaned_df = pd.concat([final_combined_df[keep_cols], temp_df], axis=1)

#print(f"Total features after cleaning: {len(final_cleaned_df.columns)}")

# Save the final cleaned dataset
final_output_path = r"/PI_electride_features_total.csv"
final_cleaned_df.to_csv(final_output_path, index=False)
#print(f"Saved final cleaned dataset to {final_output_path}")

# Show top 5 rows of the final cleaned dataframe
#print("\nTop 5 rows of the final cleaned dataframe:")
#print(final_cleaned_df.head().to_markdown(index=False, floatfmt=".4f"))

#print(f"Processing completed. Final shape: {final_cleaned_df.shape}")
#print(f"Saved to: {final_output_path}")