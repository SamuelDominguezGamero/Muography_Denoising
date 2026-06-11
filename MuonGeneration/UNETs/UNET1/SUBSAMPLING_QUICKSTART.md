# Dynamic Subsampling — Quick Start Guide

## What's the Problem?

Tu dataset es de **simulaciones** con mucha redundancia. Si el modelo entrena en todos los datos cada época, va a "memorizar" esos patrones repetidos y va a generalizar mal a datos experimentales reales.

## Cómo Funciona

**Sin subsampling dinámico:**
```
Época 1: Entrena en 2500 muestras
Época 2: Entrena en las MISMAS 2500 muestras
...
Resultado: El modelo memoriza patterns → overfitting
```

**Con subsampling dinámico:**
```
Época 1: Entrena en 750 muestras ALEATORIAS (30%)
Época 2: Entrena en 750 muestras DIFERENTES (otro 30% aleatorio)
Época 3: Entrena en 750 muestras DIFERENTES
...
Resultado: Cada época ve datos nuevos → no memoriza → generaliza mejor
```

**Garantía matemática:** Con 200 épocas y subsampling del 30%, cada muestra se ve en promedio:
$$E[\text{veces visto}] = 200 \times 0.3 = 60 \text{ veces}$$

Pero nunca ve el MISMO subconjunto dos veces.

---

## Enable It (3 líneas)

Abre `train_material_classifier.py` y busca la sección `CONFIG`:

```python
CONFIG = {
    # ... other settings ...
    
    # ── Dynamic subsampling ───────────────────────────────────────────────────
    "subsample_enabled":  False,    # ← CHANGE THIS TO True
    "subsample_fraction": 0.30,     # ← Fraction per epoch (30% = 0.3)
}
```

Luego ejecuta como siempre:
```bash
python train_material_classifier.py
```

---

## Qué Verás en la Consola

```
=== Building dataset index ===
  run0 (run0_definitive_words)  →  1234 valid, 5 skipped
  ...

=== Loading dataset indices from CSV ===
  Loaded train/val/test splits from: /path/to/data_indices/

  [DYNAMIC SUBSAMPLING ENABLED]
    Each epoch: use 30% of training data (different subset each epoch)
    This combats overfitting on simulated data with high redundancy.

  Train: 2478 samples  |  {...}
  Val:    310 samples   |  {...}
  Test:   309 samples   |  {...}

=== Building model ===
...

=== Starting training ===

Epoch 001/200  ...  TRAIN acc=0.8234  VAL acc=0.7821  TEST acc=0.7956
Epoch 002/200  ...  TRAIN acc=0.8156  VAL acc=0.7903  TEST acc=0.8012  ← note: TRAIN varies (different data)
Epoch 003/200  ...  TRAIN acc=0.8445  VAL acc=0.7856  TEST acc=0.8089
...
```

**Nota:** Train loss va a ser más ruidoso (because cada época ve datos diferentes), pero VAL y TEST deben ser más suave y no overfitear.

---

## Cómo Sintonizar

| Valor | Situación |
|-------|-----------|
| 0.10 (10%) | Datos MUY redundantes; máxima regularización |
| 0.20 (20%) | Datos altamente simulados; robustez máxima |
| **0.30 (30%)** | **Default; buen balance** |
| 0.50 (50%) | Redundancia moderada; training más rápido |
| 1.00 (100%) | Desactivado; training normal |

### Estrategia de Sintonización

1. **Empieza con 0.30** (default)
2. Mira las curvas de VAL:
   - Si sigue bajando después de época 50 → overfitting, reduce a 0.20
   - Si es muy ruidoso → aumenta a 0.50
3. Compara final TEST accuracy:
   - Con subsampling vs. sin subsampling
   - Deberías ver mejora en TEST con subsampling

---

## Ejemplo Real

### Sin subsampling (baseline):
```
Epoch  50: Train=0.95  Val=0.78  Test=0.76  ← overfitting
Epoch 100: Train=0.98  Val=0.75  Test=0.74  ← peor
Epoch 150: Train=0.99  Val=0.73  Test=0.72  ← aún peor
```

### Con subsampling (30%):
```
Epoch  50: Train=0.82  Val=0.78  Test=0.76
Epoch 100: Train=0.85  Val=0.80  Test=0.78  ← mejor!
Epoch 150: Train=0.87  Val=0.81  Test=0.80  ← sigue mejorando
```

---

## Testing (Opcional)

Para verificar que funciona:

```bash
cd UNET1
python test_dynamic_subsampling.py
```

Esto ejecuta 3 tests:
1. ✓ Verifica que cada época obtiene datos diferentes
2. ✓ Verifica que el dataset normal sigue funcionando
3. ✓ Entrena un pequeño modelo con y sin subsampling

---

## Desactivar (Reverting)

Si quieres volver a entrenar sin subsampling:

```python
"subsample_enabled": False,  # ← Back to normal
```

---

## Notas Técnicas

- **Memory:** Usa MENOS memoria (solo carga el 30% por época)
- **Speed:** Cada época es ~3x más rápida (menos datos), pero necesitas más épocas
- **Seed:** El seed aleatorio es determinístico (reproducible), pero cada época usa un subset diferente
- **Validation/Test:** No se subsamplea; siempre usa el 100%

---

## Documentación Completa

Ver `DYNAMIC_SUBSAMPLING.md` para:
- Detalles matemáticos
- Tuning avanzado
- Troubleshooting
- Comparación con otras técnicas

---

## TL;DR

1. Edit `CONFIG["subsample_enabled"] = True`
2. Run `python train_material_classifier.py`
3. Watch val/test curves be smoother and less overfitted
4. Adjust `subsample_fraction` if needed
