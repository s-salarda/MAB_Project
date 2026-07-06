from chimerax.core.commands import run
from chimerax.core.models import Surface
import numpy as np

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
pqr_file     = "/Users/kalenrobeson/Documents/Research/Projects_and_Data/MAB/DelphiForce_4_better_alignment/files_pqr/wt_sim2_frame494_2_2.pqr"
residue_file = "/Users/kalenrobeson/Documents/Research/Projects_and_Data/MAB/DelphiForce_4_better_alignment/files_pqr/wt_sim2_frame494_2_2.residue"
actin_file   = "/Users/kalenrobeson/Documents/Research/Projects_and_Data/MAB/DelphiForce_4_better_alignment/files_pqr/human_actc1_5mer_for_kalen_2.pqr"
color_file   = "/Users/kalenrobeson/Documents/Research/Projects_and_Data/MAB/DelphiForce_4_better_alignment/visualize_vectors/color_myosin_human.cxc"

# ---------------------------------------------------------------------------
# Visualization parameters
# ---------------------------------------------------------------------------
force_scale        = 800    # Arrow length = magnitude * force_scale (Å)
shaft_scale        = 40   # Shaft radius = magnitude * shaft_scale (Å)
cone_radius_factor = 2    # Cone radius = shaft_radius * cone_radius_factor
cone_height_factor = 5    # Cone height = shaft_radius * cone_height_factor
force_threshold    = 0.001    # Forces below this magnitude are not drawn
divisions          = 12     # Polygon divisions per cylinder/cone — lower=faster, higher=smoother

def hex_to_rgba(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4)) + (255,)

residue_color = hex_to_rgba("#A9A9A9") # dark gray
total_color   = hex_to_rgba("#000000") # black

# Total force arrow — fixed radii so it always stands out
total_shaft_radius = 1.2
total_cone_radius  = 3.0
total_cone_height  = 4.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _rotation_matrix(uz):
    """3x3 matrix rotating +Z onto unit vector uz."""
    z  = np.array([0.0, 0.0, 1.0])
    uz = np.asarray(uz, dtype=float)
    axis  = np.cross(z, uz)
    sin_a = np.linalg.norm(axis)
    cos_a = float(np.dot(z, uz))
    if sin_a < 1e-8:
        return np.eye(3) if cos_a > 0 else np.diag([1.0, -1.0, -1.0])
    axis /= sin_a
    x, y, zz = axis
    c, s = cos_a, sin_a
    return np.array([
        [c + x*x*(1-c),    x*y*(1-c) - zz*s,  x*zz*(1-c) + y*s],
        [y*x*(1-c) + zz*s, c + y*y*(1-c),      y*zz*(1-c) - x*s],
        [zz*x*(1-c) - y*s, zz*y*(1-c) + x*s,   c + zz*zz*(1-c)],
    ])


def _cylinder_geometry(base, tip, radius, divs):
    """Return (verts, normals, triangles) for a cylinder from base to tip."""
    direction = np.asarray(tip, dtype=float) - np.asarray(base, dtype=float)
    height    = np.linalg.norm(direction)
    if height < 1e-9 or radius < 1e-9:
        return None
    R      = _rotation_matrix(direction / height)
    angles = np.linspace(0, 2*np.pi, divs, endpoint=False)
    ca, sa = np.cos(angles), np.sin(angles)

    local = np.array([[radius*c, radius*s, h]
                      for h in (0.0, height)
                      for c, s in zip(ca, sa)])
    verts = local @ R.T + np.asarray(base)

    norms_local = np.array([[c, s, 0.0]
                             for _ in range(2)
                             for c, s in zip(ca, sa)])
    norms = norms_local @ R.T

    tris = []
    for i in range(divs):
        j = (i + 1) % divs
        tris += [[i, j, divs+i], [j, divs+j, divs+i]]
    return verts, norms, np.array(tris, dtype=np.int32)


def _cone_geometry(base, tip, radius, divs):
    """Return (verts, normals, triangles) for a cone from base to tip."""
    direction = np.asarray(tip, dtype=float) - np.asarray(base, dtype=float)
    height    = np.linalg.norm(direction)
    if height < 1e-9 or radius < 1e-9:
        return None
    R      = _rotation_matrix(direction / height)
    angles = np.linspace(0, 2*np.pi, divs, endpoint=False)
    ca, sa = np.cos(angles), np.sin(angles)

    local = np.array([[radius*c, radius*s, 0.0] for c, s in zip(ca, sa)]
                     + [[0.0, 0.0, height]])
    verts = local @ R.T + np.asarray(base)

    slant = np.sqrt(radius**2 + height**2)
    norms_local = np.array(
        [[c*height/slant, s*height/slant, radius/slant] for c, s in zip(ca, sa)]
        + [[0.0, 0.0, 1.0]]
    )
    norms    = norms_local @ R.T
    apex_idx = divs
    tris     = [[i, (i+1) % divs, apex_idx] for i in range(divs)]
    return verts, norms, np.array(tris, dtype=np.int32)


def _merge_geometries(geom_list):
    """Concatenate a list of (verts, norms, tris) into one mesh."""
    all_v, all_n, all_t = [], [], []
    offset = 0
    for verts, norms, tris in geom_list:
        all_v.append(verts)
        all_n.append(norms)
        all_t.append(tris + offset)
        offset += len(verts)
    return (np.concatenate(all_v).astype(np.float32),
            np.concatenate(all_n).astype(np.float32),
            np.concatenate(all_t).astype(np.int32))


def _add_surface(session, verts, norms, tris, rgba, name):
    """Create and register a Surface model from raw geometry."""
    model = Surface(name, session)
    model.set_geometry(verts, norms, tris)
    model.color = rgba
    session.models.add([model])
    return model


# ---------------------------------------------------------------------------
# Step 1 — Open PQR and parse CA positions (COM fallback)
# ---------------------------------------------------------------------------
run(session, f"open {pqr_file}")

residue_coords = {}
ca_positions   = {}

with open(pqr_file) as f:
    for line in f:
        if not line.startswith(("ATOM", "HETATM")):
            continue
        parts = line.split()
        try:
            atom_name = parts[2]
            chain     = parts[4]
            resnum    = int(parts[5])
            x, y, z   = float(parts[6]), float(parts[7]), float(parts[8])
        except (IndexError, ValueError):
            continue
        key = (chain, resnum)
        residue_coords.setdefault(key, []).append((x, y, z))
        if atom_name == "CA":
            ca_positions[key] = np.array([x, y, z])

residue_centers = {
    key: ca_positions[key] if key in ca_positions
         else np.array(coords).mean(axis=0)
    for key, coords in residue_coords.items()
}

all_atoms  = np.array([pos for coords in residue_coords.values() for pos in coords])
mol_center = all_atoms.mean(axis=0)

ca_count  = sum(1 for k in residue_coords if k in ca_positions)
com_count = sum(1 for k in residue_coords if k not in ca_positions)
print(f"Molecular center: {mol_center}")
print(f"Positions: {ca_count} CA,  {com_count} COM fallback")
print(f"force_scale={force_scale}  shaft_scale={shaft_scale}  threshold={force_threshold}")

# ---------------------------------------------------------------------------
# Step 2 — Parse residue forces
# ---------------------------------------------------------------------------
force_data  = []
total_force = None

with open(residue_file) as f:
    for line in f:
        if line.strip().startswith("Total force:"):
            parts = line.split()
            try:
                total_force = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
                print(f"Total force (raw): {total_force}")
            except (IndexError, ValueError):
                print("Warning: could not parse Total force line")
            continue

        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            chain      = parts[1]
            resnum     = int(parts[2])
            fx, fy, fz = float(parts[5]), float(parts[6]), float(parts[7])
        except (ValueError, IndexError):
            continue
        mag = np.sqrt(fx**2 + fy**2 + fz**2)
        if mag < force_threshold:
            continue
        key = (chain, resnum)
        if key in residue_centers:
            force_data.append((key, fx, fy, fz, mag))
        else:
            print(f"Warning: no position for chain {chain} res {resnum}")

# ---------------------------------------------------------------------------
# Step 3 — Build all per-residue arrows as one batched Surface
# ---------------------------------------------------------------------------
if not force_data:
    print("No forces above threshold — no per-residue arrows drawn.")
else:
    max_mag = max(d[4] for d in force_data)
    min_mag = min(d[4] for d in force_data)
    print(f"Drawing {len(force_data)} arrows  |  "
          f"mag range: {min_mag:.6f} – {max_mag:.6f}  |  "
          f"length range: {min_mag*force_scale:.3f} – {max_mag*force_scale:.3f} Å")

    geoms = []
    for key, fx, fy, fz, mag in force_data:
        origin = residue_centers[key]
        unit   = np.array([fx, fy, fz]) / mag

        length  = mag * force_scale
        shaft_r = mag * shaft_scale
        cone_r  = shaft_r * cone_radius_factor
        cone_h  = shaft_r * cone_height_factor

        shaft_tip = origin + unit * (length - cone_h)
        arrow_tip = origin + unit * length

        g = _cylinder_geometry(origin, shaft_tip, shaft_r, divisions)
        if g:
            geoms.append(g)
        g = _cone_geometry(shaft_tip, arrow_tip, cone_r, divisions)
        if g:
            geoms.append(g)

    if geoms:
        verts, norms, tris = _merge_geometries(geoms)
        _add_surface(session, verts, norms, tris, residue_color, "per_residue_forces")
        print("Per-residue force surface added.")

# ---------------------------------------------------------------------------
# Step 4 — Total force arrow as its own Surface
# ---------------------------------------------------------------------------
if total_force is not None:
    mag = np.linalg.norm(total_force)
    if mag < 1e-6:
        print("Total force magnitude is zero — skipping.")
    else:
        unit      = total_force / mag
        length    = mag * force_scale
        shaft_tip = mol_center + unit * (length - total_cone_height)
        arrow_tip = mol_center + unit * length

        geoms = []
        g = _cylinder_geometry(mol_center, shaft_tip, total_shaft_radius, divisions)
        if g:
            geoms.append(g)
        g = _cone_geometry(shaft_tip, arrow_tip, total_cone_radius, divisions)
        if g:
            geoms.append(g)

        if geoms:
            verts, norms, tris = _merge_geometries(geoms)
            _add_surface(session, verts, norms, tris, total_color, "total_force")
            print(f"Total force arrow added.  Magnitude: {mag:.6f}  "
                  f"Length: {length:.3f} Å  (scale={force_scale})")
else:
    print("No total force line found in residue file.")

run(session, f"open {actin_file}")
run(session, f"open {color_file}")

