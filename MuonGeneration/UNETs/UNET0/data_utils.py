"""
Data utilities: POCA extraction, GT creation, matching
"""

import numpy as np
import ROOT
import tarfile
import tempfile
import shutil
import random
from pathlib import Path


# ===========================================================================
# POCA EXTRACTION FROM TAR OR DIRECTORY
# ===========================================================================

def extract_poca_from_source(source_path, output_dir, world_size=128.0):
    """
    Extrae proyecciones POCA desde un TAR o directorio a tensores 128x128x3.
    Procesa archivo por archivo para ahorrar espacio en disco y salta duplicados.
    """
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[PHASE 1] EXTRACT POCA FROM {'TAR' if source_path.is_file() else 'DIRECTORY'}")
    print(f"  Source: {source_path}")
    print(f"  Output: {output_dir}")
    
    poca_files = []
    skipped_count = 0

    if source_path.is_file() and source_path.suffix == '.tar':
        # Crear un directorio temporal en el disco principal para evitar llenar /tmp (RAM)
        temp_dir = Path(tempfile.mkdtemp(prefix="poca_single_", dir=output_dir.parent))
        
        with tarfile.open(source_path, 'r') as tar:
            members = [m for m in tar.getmembers() if m.name.endswith('.root')]
            total = len(members)
            print(f"  Found {total} .root files in TAR. Processing one by one...")

            for idx, member in enumerate(members, 1):
                # 1. Definir metadata y ruta de salida
                metadata = Path(member.name).stem.replace("POCA_merged__", "")
                out_path = output_dir / f"tensor_2D_POCA_{metadata}.npy"

                # 2. Check si ya existe para saltarlo
                if out_path.exists():
                    poca_files.append((out_path, metadata))
                    skipped_count += 1
                    continue

                # 3. Extracción unitaria
                tar.extract(member, path=temp_dir)
                root_file = temp_dir / member.name
                
                try:
                    # 4. Cargar datos con ROOT RDataFrame
                    df = ROOT.RDataFrame("events", str(root_file))
                    df = df.Filter("abs(theta) > 0.00000001")
                    res = df.AsNumpy(columns=["poca_x", "poca_y", "poca_z"])
                    
                    x, y, z = res["poca_x"], res["poca_y"], res["poca_z"]
                    
                    # 5. Crear histogramas 2D (Proyecciones XY, XZ, YZ)
                    world_min, world_max = -world_size / 2.0, world_size / 2.0
                    edges = np.linspace(world_min, world_max, 129)
                    
                    xy_hist, _, _ = np.histogram2d(x, y, bins=edges)
                    xz_hist, _, _ = np.histogram2d(x, z, bins=edges)
                    yz_hist, _, _ = np.histogram2d(y, z, bins=edges)
                    
                    # 6. Normalización a uint8 (0-255)
                    def norm(p):
                        p = p.astype(np.float32).T
                        mx = np.max(p)
                        return np.uint8((p / mx * 255) if mx > 0 else p)
                    
                    tensor = np.stack([norm(xy_hist), norm(xz_hist), norm(yz_hist)], axis=2)
                    
                    # 7. Guardar resultado
                    np.save(out_path, tensor)
                    poca_files.append((out_path, metadata))
                    
                    if idx % max(1, total // 10) == 0:
                        print(f"  [{idx}/{total}] ✓ {out_path.name}")

                except Exception as e:
                    print(f"  [ERROR] Processing {member.name}: {e}")
                finally:
                    # 8. ELIMINACIÓN INMEDIATA DEL ROOT EXTRAÍDO
                    if root_file.exists():
                        root_file.unlink()
                        # Limpiar carpetas vacías creadas por tar.extract si las hay
                        for parent in root_file.parents:
                            if parent == temp_dir: break
                            if not any(parent.iterdir()): parent.rmdir()

        shutil.rmtree(temp_dir)
    
    else:
        # Lógica para cuando source_path es un directorio directo
        root_files = list(source_path.glob("*.root"))
        total = len(root_files)
        print(f"  Found {total} .root files in directory.")

        for idx, root_file in enumerate(root_files, 1):
            metadata = root_file.stem.replace("POCA_merged__", "")
            out_path = output_dir / f"tensor_2D_POCA_{metadata}.npy"

            if out_path.exists():
                poca_files.append((out_path, metadata))
                skipped_count += 1
                continue

            try:
                df = ROOT.RDataFrame("events", str(root_file))
                df = df.Filter("abs(theta) > 0.00000001")
                res = df.AsNumpy(columns=["poca_x", "poca_y", "poca_z"])
                
                x, y, z = res["poca_x"], res["poca_y"], res["poca_z"]
                world_min, world_max = -world_size / 2.0, world_size / 2.0
                edges = np.linspace(world_min, world_max, 129)
                
                xy_hist, _, _ = np.histogram2d(x, y, bins=edges)
                xz_hist, _, _ = np.histogram2d(x, z, bins=edges)
                yz_hist, _, _ = np.histogram2d(y, z, bins=edges)
                
                def norm(p):
                    p = p.astype(np.float32).T
                    mx = np.max(p)
                    return np.uint8((p / mx * 255) if mx > 0 else p)
                
                tensor = np.stack([norm(xy_hist), norm(xz_hist), norm(yz_hist)], axis=2)
                
                np.save(out_path, tensor)
                poca_files.append((out_path, metadata))
                
                if idx % max(1, total // 10) == 0:
                    print(f"  [{idx}/{total}] ✓ {out_path.name}")
            except Exception as e:
                print(f"  [ERROR] Processing {root_file.name}: {e}")

    print(f"  ✓ Finalizado: {len(poca_files)} totales ({skipped_count} ya existían, {len(poca_files)-skipped_count} procesados)")
    return poca_files






# ===========================================================================
# GROUND TRUTH FROM JSON
# ===========================================================================

def create_gt_from_json(json_path, output_dir_3d, output_dir_2d, world_size=128.0, voxel_size=1.0):
    """
    Create ground truth tensors from JSON geometry → 128×128×128 and 128×128×3.
    Checks if output files already exist to skip processing.
    
    Returns:
        (Path to 3D tensor, Path to 2D tensor, metadata_str)
    """
    import json
    import numpy as np
    from pathlib import Path
    
    output_dir_3d = Path(output_dir_3d)
    output_dir_2d = Path(output_dir_2d)
    output_dir_3d.mkdir(parents=True, exist_ok=True)
    output_dir_2d.mkdir(parents=True, exist_ok=True)
    
    metadata = Path(json_path).stem
    path_3d = output_dir_3d / f"tensorGT_3D_{metadata}.npy"
    path_2d = output_dir_2d / f"tensorGT_2D_{metadata}.npy"
    
    # --- CHECK SI YA EXISTEN ---
    if path_3d.exists() and path_2d.exists():
        return path_3d, path_2d, metadata

    # --- PROCESAMIENTO (solo si no existen) ---
    # Load geometry
    with open(json_path) as f:
        geometry = json.load(f)
    
    # Create 3D tensor
    n = int(world_size / voxel_size)
    tensor_3d = np.zeros((n, n, n), dtype=np.uint8)
    world_min = -world_size / 2.0
    
    # Fill from voxels
    for voxel in geometry.get("TheVoxels", []):
        if voxel.get("materialVoxel", "air").lower() == "air":
            continue
        
        x_min = voxel["xPosVoxel"] - voxel["xSizeVoxel"] / 2
        x_max = voxel["xPosVoxel"] + voxel["xSizeVoxel"] / 2
        y_min = voxel["yPosVoxel"] - voxel["ySizeVoxel"] / 2
        y_max = voxel["yPosVoxel"] + voxel["ySizeVoxel"] / 2
        z_min = voxel["zPosVoxel"] - voxel["zSizeVoxel"] / 2
        z_max = voxel["zPosVoxel"] + voxel["zSizeVoxel"] / 2
        
        xi_min = max(0, min(int((x_min - world_min) / voxel_size), n))
        xi_max = max(0, min(int((x_max - world_min) / voxel_size), n))
        yi_min = max(0, min(int((y_min - world_min) / voxel_size), n))
        yi_max = max(0, min(int((y_max - world_min) / voxel_size), n))
        zi_min = max(0, min(int((z_min - world_min) / voxel_size), n))
        zi_max = max(0, min(int((z_max - world_min) / voxel_size), n))
        
        # Asignación por slices
        tensor_3d[yi_min:yi_max, xi_min:xi_max, zi_min:zi_max] = 1
    
    # Create 2D projections
    # axis 2 es Z (XY), axis 0 es Y (XZ), axis 1 es X (YZ)
    xy = np.max(tensor_3d, axis=2)
    xz = np.max(tensor_3d, axis=0).T
    yz = np.max(tensor_3d, axis=1).T
    tensor_2d = np.stack([xy, xz, yz], axis=2).astype(np.uint8)
    
    # Save
    np.save(path_3d, tensor_3d)
    np.save(path_2d, tensor_2d)
    
    return path_3d, path_2d, metadata



# ===========================================================================
# MATCHING & PAIRING
# ===========================================================================


def auto_match_datasets(poca_dir, gt_dir):
    """
    Empareja archivos POCA y GT limpiando los prefijos específicos detectados.
    """
    poca_dir = Path(poca_dir)
    gt_dir = Path(gt_dir)

    # 1. Obtener archivos
    poca_paths = list(poca_dir.glob("tensor_2D_POCA_*.npy"))
    gt_paths = list(gt_dir.glob("tensorGT_2D_*.npy"))

    # 2. Crear diccionario de GT
    # Eliminamos 'tensorGT_2D__' (con doble guion si lo tiene) o 'tensorGT_2D_'
    # Usamos lstrip para asegurar que quitamos el guion bajo sobrante del inicio
    gt_lookup = {}
    for f in gt_paths:
        # Quitamos el prefijo y los guiones bajos sobrantes al inicio del ID
        core_id = f.name.replace("tensorGT_2D_", "").lstrip("_")
        gt_lookup[core_id] = f

    pairs = []
    
    # 3. Emparejar
    for poca_path in poca_paths:
        # Quitamos el prefijo POCA
        core_id = poca_path.name.replace("tensor_2D_POCA_", "").lstrip("_")
        
        if core_id in gt_lookup:
            pairs.append((poca_path, gt_lookup[core_id]))
        else:
            # Esto te ayudará a ver si algún nombre no encaja por un carácter
            print(f"⚠️ Sin match para: {core_id}")

    print(f"\n✅ Emparejamiento completado:")
    print(f"  - Total POCA: {len(poca_paths)}")
    print(f"  - Total GT:   {len(gt_paths)}")
    print(f"  - PARES:      {len(pairs)}")
    
    return pairs


# ===========================================================================
# H5 DATASET CREATION
# ===========================================================================

def create_h5_dataset(pairs, output_h5, train_ratio=0.8, val_ratio=0.1):
    """
    Create H5 dataset from POCA-GT pairs (buffered streaming - no RAM overflow)
    Optimized for training speed (reads/writes without heavy CPU compression overhead).
    
    Args:
        pairs: List of (poca_path, gt_path)
        output_h5: Path to output .h5 file
        train_ratio, val_ratio: Split ratios (test = 1 - train - val)
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py not installed. Install with: pip install h5py")
    
    output_h5 = Path(output_h5)
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[PHASE 4] CREATE H5 DATASET (BUFFERED STREAMING MODE)")
    print(f"  Output: {output_h5}")
    
    # Mezclar y separar conjuntos
    random.shuffle(pairs)
    n_total = len(pairs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]
    
    print(f"  Train: {len(train_pairs)} ({100*len(train_pairs)/n_total:.1f}%)")
    print(f"  Val:   {len(val_pairs)} ({100*len(val_pairs)/n_total:.1f}%)")
    print(f"  Test:  {len(test_pairs)} ({100*len(test_pairs)/n_total:.1f}%)")
    
    # Tamaño del bloque temporal en RAM para optimizar la escritura en disco (I/O)
    buffer_size = 100 
    
    with h5py.File(output_h5, 'w') as f:
        for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
            if not split_pairs:
                continue
            
            print(f"\n  [{split_name.upper()}] Writing {len(split_pairs)} pairs...")
            
            grp = f.create_group(split_name)
            
            # ELIMINADO 'gzip': Habilitamos chunking automático optimizado para lectura aleatoria en entrenamiento
            ds_poca = grp.create_dataset("poca", shape=(len(split_pairs), 128, 128, 3), 
                                         dtype=np.uint8, chunks=(1, 128, 128, 3))
            ds_gt = grp.create_dataset("gt", shape=(len(split_pairs), 128, 128, 3), 
                                       dtype=np.uint8, chunks=(1, 128, 128, 3))
            
            # Inicializar buffers temporales
            poca_buffer = []
            gt_buffer = []
            buffer_start_idx = 0
            
            for idx, (poca_p, gt_p) in enumerate(split_pairs):
                try:
                    # Cargar un par a la vez en RAM
                    poca_data = np.load(Path(poca_p)).astype(np.uint8)
                    gt_data = np.load(Path(gt_p)).astype(np.uint8)
                    
                    poca_buffer.append(poca_data)
                    gt_buffer.append(gt_data)
                    
                    # Cuando el buffer se llena o es el último elemento, volcamos en bloque a H5
                    if len(poca_buffer) == buffer_size or (idx + 1) == len(split_pairs):
                        end_idx = idx + 1
                        
                        # Escritura contigua en disco (Altamente eficiente)
                        ds_poca[buffer_start_idx:end_idx] = np.array(poca_buffer)
                        ds_gt[buffer_start_idx:end_idx] = np.array(gt_buffer)
                        
                        # Resetear buffers
                        poca_buffer = []
                        gt_buffer = []
                        buffer_start_idx = end_idx
                    
                    # Progreso en consola
                    if (idx + 1) % max(1, len(split_pairs) // 5) == 0 or (idx + 1) == len(split_pairs):
                        print(f"    [{idx + 1}/{len(split_pairs)}] ✓")
                        
                except Exception as e:
                    print(f"    [ERROR] Pair {idx} ({poca_p}): {e}")
                    # Si falla un archivo individual, vaciamos el buffer acumulado hasta el momento para no perder índice
                    if poca_buffer:
                        end_idx = buffer_start_idx + len(poca_buffer)
                        ds_poca[buffer_start_idx:end_idx] = np.array(poca_buffer)
                        ds_gt[buffer_start_idx:end_idx] = np.array(gt_buffer)
                        buffer_start_idx = end_idx
                        poca_buffer = []
                        gt_buffer = []

    print(f"\n  ✓ H5 created: {output_h5.name}")
    return output_h5