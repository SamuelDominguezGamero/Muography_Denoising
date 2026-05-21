# Proyecciones Ortogonales - Guía de Orientaciones

## Comparación de Scripts: `create_ground_truth_from_json.py` vs `visualize_json.py`

### 1. Formato del Tensor

| Script | Formato | Orden de Índices | Notas |
|--------|---------|------------------|-------|
| `create_ground_truth_from_json.py` | `(Y, X, Z)` | `tensor[y_idx, x_idx, z_idx]` | Binario (0/1) |
| `visualize_json.py` | `(Z, Y, X)` | `geometry_3d[z_idx, y_idx, x_idx]` | Material ID (0/1) |

---

### 2. Cálculo de Proyecciones

#### create_ground_truth_from_json.py (Tensor: Y, X, Z)

```python
# Proyección XY (Vista superior): X horizontal, Y vertical
projection_xy = np.max(tensor, axis=2)  # Elimina Z → forma (Y, X)
# Cuando se dibuja: imshow muestra Y en vertical (eje 0), X en horizontal (eje 1) ✓

# Proyección XZ (Vista frontal): X horizontal, Z vertical  
projection_xz = np.max(tensor, axis=0).T  # Elimina Y, transpone → forma (Z, X)
# Cuando se dibuja: imshow muestra Z en vertical (eje 0), X en horizontal (eje 1) ✓

# Proyección YZ (Vista lateral): Y horizontal, Z vertical
projection_yz = np.max(tensor, axis=1).T  # Elimina X, transpone → forma (Z, Y)
# Cuando se dibuja: imshow muestra Z en vertical (eje 0), Y en horizontal (eje 1) ✓
```

#### visualize_json.py (Tensor: Z, Y, X)

```python
# Proyección XY (Vista superior): X horizontal, Y vertical
xy_proj = np.max(geometry_3d, axis=0)  # Elimina Z → forma (Y, X)
# Cuando se dibuja: imshow muestra Y en vertical (eje 0), X en horizontal (eje 1) ✓

# Proyección XZ (Vista frontal): X horizontal, Z vertical
xz_proj = np.max(geometry_3d, axis=1)  # Elimina Y → forma (Z, X)
# Cuando se dibuja: imshow muestra Z en vertical (eje 0), X en horizontal (eje 1) ✓

# Proyección YZ (Vista lateral): Y horizontal, Z vertical
yz_proj = np.max(geometry_3d, axis=2)  # Elimina X → forma (Z, Y)
# Cuando se dibuja: imshow muestra Z en vertical (eje 0), Y en horizontal (eje 1) ✓
```

---

### 3. Consistencia de Ejes

Cuando se dibuja una proyección con `imshow(array, origin='lower')`:
- **Primer índice (eje 0)** → Dimensión **vertical** (Y en imshow)
- **Segundo índice (eje 1)** → Dimensión **horizontal** (X en imshow)

| Proyección | Deseado | create_ground_truth | visualize_json | Status |
|------------|---------|-------------------|-----------------|--------|
| XY | X horiz, Y vert | (Y, X) | (Y, X) | ✅ Consistente |
| XZ | X horiz, Z vert | (Z, X)† | (Z, X) | ✅ Consistente |
| YZ | Y horiz, Z vert | (Z, Y)† | (Z, Y) | ✅ Consistente |

†: Con transposición `.T` para obtener la forma correcta

---

### 4. Guardado en Archivo .npy

#### create_ground_truth_from_json.py

```python
projections_stacked = np.stack([projection_xy, projection_xz, projection_yz], axis=2)
# Resultado: shape (128, 128, 3)
# 
# tensor_2d[:, :, 0] = XY: (128, 128) con dim0=Y vertical, dim1=X horizontal
# tensor_2d[:, :, 1] = XZ: (128, 128) con dim0=Z vertical, dim1=X horizontal
# tensor_2d[:, :, 2] = YZ: (128, 128) con dim0=Z vertical, dim1=Y horizontal
```

---

### 5. Visualización

#### PNG Local (visualize_2d_local)

```python
# Ejes correctamente etiquetados:
axes[0].set_xlabel('X (cm)'), axes[0].set_ylabel('Y (cm)')  # XY
axes[1].set_xlabel('X (cm)'), axes[1].set_ylabel('Z (cm)')  # XZ
axes[2].set_xlabel('Y (cm)'), axes[2].set_ylabel('Z (cm)')  # YZ
```

#### Interactivo (visualize_2d_interactive)

```python
# Carga desde archivo .npy y dibuja con orientaciones correctas
tensor_2d = np.load(tensor_2d_path)  # shape (128, 128, 3)

projection_xy = tensor_2d[:, :, 0]  # (128, 128)
projection_xz = tensor_2d[:, :, 1]  # (128, 128)
projection_yz = tensor_2d[:, :, 2]  # (128, 128)

# Se dibuja directamente sin modificación adicional
axes[i].imshow(projection_xy/xz/yz, origin='lower', ...)
```

---

### 6. Verificación de Consistencia

Para verificar que todo está correcto:

```bash
# 1. Generar tensor
python3 create_ground_truth_from_json.py --json geometry.json

# 2. Comparar con visualize_json.py
python3 visualize_json.py --json /ruta/completa/geometry.json --save /tmp/compare.png

# 3. Las imágenes deben ser idénticas
```

---

## Cambios Aplicados

### Versión Anterior ❌
- XZ: `projection_xz = np.max(tensor, axis=0)` → forma (X, Z) 
  - Resultado: X en vertical, Z en horizontal (INCORRECTO)
  
- YZ: `projection_yz = np.max(tensor, axis=1)` → forma (Y, Z)
  - Resultado: Y en vertical, Z en horizontal (INCORRECTO)

### Versión Actual ✅
- XZ: `projection_xz = np.max(tensor, axis=0).T` → forma (Z, X)
  - Resultado: Z en vertical, X en horizontal (CORRECTO)
  
- YZ: `projection_yz = np.max(tensor, axis=1).T` → forma (Z, Y)
  - Resultado: Z en vertical, Y en horizontal (CORRECTO)

---

## Implicaciones

La transposición asegura que:

1. ✅ **Orientaciones correctas**: Z siempre en vertical en XZ e YZ
2. ✅ **Consistencia con visualize_json.py**: Mismos ejes y orientaciones
3. ✅ **Archivo .npy guardado correctamente**: Cada canal tiene la forma esperada
4. ✅ **Visualizaciones interactivas**: Al cargar el .npy, las orientaciones son correctas sin necesidad de transposición adicional

---

## Resumen Visual

```
XY Projection (Vista Superior)          XZ Projection (Vista Frontal)
┌─────────────────────────────┐         ┌─────────────────────────────┐
│ ▲ Y                         │         │ ▲ Z                         │
│ │                           │         │ │                           │
│ │  Geometría XY             │         │ │  Geometría XZ             │
│ │                           │         │ │                           │
│ └──────────────────────────>│         │ └──────────────────────────>│
│       X →                   │         │       X →                   │
└─────────────────────────────┘         └─────────────────────────────┘

YZ Projection (Vista Lateral)
┌─────────────────────────────┐
│ ▲ Z                         │
│ │                           │
│ │  Geometría YZ             │
│ │                           │
│ └──────────────────────────>│
│       Y →                   │
└─────────────────────────────┘
```
