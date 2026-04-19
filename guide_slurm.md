# SLURM Quick Reference - Guía de Comandos Básicos

## 📍 Conectarse al Cluster

```bash
ssh dominguezs@login1.ifca.es
# o
ssh dominguezs@login2.ifca.es  # Si login1 está ocupado
```

---

## 💾 Gestión de Cuota de Disco

### Ver cuota actual

```bash
quota -s
```

**Información:**
- Muestra tu cuota total
- Espacio usado
- Espacio disponible

### Ver espacio en directorios específicos

```bash
# Tamaño total del proyecto
du -sh /gpfs/users/dominguezs/Muography_Denoising

# Tamaño por subdirectorio
du -sh /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/*

# Top 10 carpetas más grandes
du -sh /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/* | sort -h | tail -10
```

### Limpiar directorios grandes

```bash
# ⚠️ CUIDADO: Esto borra datos intermedios (pero NO merged_poca_data)

# Borrar archivos crudos (.root)
rm -rf /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/data_raw/*

# Borrar archivos preprocesados (.root)
rm -rf /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/data_preprocessed/*

# Borrar POCA individuales (si ya fueron mergeados)
rm -rf /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/post_POCA_data/*

# Borrar logs viejos
rm -rf /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/logs/*

# Ver cuota después de limpiar
quota -s
```

**Estimado de espacio liberado:**
- `.root` (raw): ~200 MB por archivo × N geoms
- `.root` (preprocessed): ~150 MB por archivo × N geoms
- Logs: ~1-2 MB por archivo

---

## 📊 Monitoreo de Jobs

### Ver tus jobs activos

```bash
# Todos tus jobs
squeue -u dominguezs

# Formato más legible
squeue -u dominguezs -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"

# Más detalles
squeue -u dominguezs -l
```

**Columnas:**
- `JOBID` - ID del job
- `STAT` - Estado (R=running, PD=pending, CA=cancelled, CG=completing, F=failed)
- `TIME` - Tiempo ejecutándose
- `NODES` - Nodos usados
- `REASON` - Razón si está pending (Resources, Priority, etc)

### Ver jobs de todos (más lento)

```bash
squeue
```

### Watch en tiempo real

```bash
# Actualizar cada 2 segundos
watch -n 2 'squeue -u dominguezs -o "%.18i %.9P %.8j %.8u %.2t %.10M %R"'

# Ctrl+C para salir
```

### Ver un job específico

```bash
squeue -j 1234567  # Reemplaza con tu job ID
```

### Ver detalles de job completado

```bash
sacct -j 1234567 --format=JobID,JobName,Elapsed,State,ExitCode
```

---

## 🚀 Enviar Jobs

### Comando básico: sbatch

```bash
sbatch script.sh
```

**Retorna**: `Submitted batch job 1234567`

### Script de ejemplo (job.sh)

```bash
#!/bin/bash
#SBATCH --job-name=mi_simulacion
#SBATCH --output=log_%j.out
#SBATCH --error=log_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=general  # o gpu si tienes GPU

# Tu comando aquí
python3 mi_script.py
```

### Flags importantes

| Flag | Ejemplo | Descripción |
|------|---------|-------------|
| `--job-name` | `--job-name=test` | Nombre del job (visible en squeue) |
| `--mem` | `--mem=16G` | Memoria total a reservar |
| `--ntasks` | `--ntasks=1` | Número de tareas (procesos) |
| `--cpus-per-task` | `--cpus-per-task=4` | CPUs por tarea (para parallelismo) |
| `--time` | `--time=01:30:00` | Límite de tiempo HH:MM:SS |
| `--output` | `--output=log_%j.out` | Archivo stdout (%j = job ID) |
| `--error` | `--error=log_%j.err` | Archivo stderr |
| `--chdir` | `--chdir=/path` | Directorio de trabajo |
| `--dependency` | `--dependency=afterok:123:456` | Esperar otros jobs |
| `--partition` | `--partition=gpu` | Partición (cola) específica |
| `--array` | `--array=1-100` | Array jobs (1-100) |

### Ejemplo avanzado: Job con dependencia

```bash
# Job 1: Genera datos
sbatch job1_simulate.sh  # Returns: 1234567

# Job 2: Espera a que Job 1 termine exitosamente
sbatch --dependency=afterok:1234567 job2_process.sh

# Otros tipos de dependencia:
# --dependency=afterok:123      Tras exitoso
# --dependency=afterany:123     Tras finalizarse (exitoso o error)
# --dependency=afternotok:123   Tras error
# --dependency=afterok:123:456  Tras exitosos ambos
```

### Enviar múltiples jobs con wrap

```bash
# En una línea
sbatch --job-name=test --wrap="python3 script.py arg1 arg2"

# Con dirección de salida
sbatch --job-name=test --output=log_%j.out --wrap="python3 script.py"
```

---

## 🛑 Cancelar Jobs

### Cancelar un job específico

```bash
scancel 1234567
```

### Cancelar todos tus jobs

```bash
scancel -u dominguezs
```

### Cancelar por nombre

```bash
scancel -n mi_simulacion  # Cancela todos los jobs llamados "mi_simulacion"
```

### Cancelar múltiples specific

```bash
scancel 123 456 789  # Cancela esos 3 jobs
```

---

## 📋 Información del Cluster

### Ver particiones disponibles

```bash
sinfo
```

**Salida:**
```
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
gpu*         up 3-00:00:00      2   idle node[51-52]
general      up 7-00:00:00     10 alloc node[1-10]
```

### Ver nodos específicos

```bash
sinfo -N  # Ver todos los nodos

sinfo -n node01  # Ver nodo específico
```

### Ver límites (walltime máximo, etc)

```bash
sinfo -o "%20N %10c %10m %20G %20l"  # Nodos, CPUs, Memoria, GPU, Walltime
```

---

## 📈 Monitoreo de Performance

### Ver uso de recursos en tiempo real

```bash
# En el nodo de compute (si tienes SSH directo)
ssh node01
top  # Presiona 'q' para salir
```

### Ver histórico de un job

```bash
sacct -j 1234567 --format=JobID,JobName,Elapsed,TotalCPU,MaxRSS,State
```

**Campos útiles:**
- `MaxRSS` - Pico máximo de memoria usado (en KB, ÷1024÷1024 para GB)
- `Elapsed` - Tiempo total de ejecución
- `State` - COMPLETED, FAILED, TIMEOUT, etc

### Ver uso de GPU (si disponible)

```bash
# Antes de lanzar job
squeue --format="%i %j %b"  # Ver GPUs asignadas

# Durante job (en el nodo)
nvidia-smi
```

---

## 🔍 Troubleshooting: Estados de Jobs

### Job en estado PD (Pending)

```bash
squeue -j 1234567 -o "%.18i %.9P %.8j %.8u %.2t %.10M %R"
```

**Razones comunes en columna %R:**
- `Resources` - Sin recursos disponibles (esperar)
- `Priority` - Baja prioridad (otros jobs adelante)
- `Dependency` - Esperando job dependiente
- `ReqNodeNotAvail` - Nodo no disponible

### Job falló (Estado F o CA)

```bash
# Ver por qué falló
cat /gpfs/.../logs/log_jobid.err

# Ver exit code
sacct -j 1234567 --format=JobID,ExitCode
```

**Exit codes comunes:**
- `0` - Exitoso
- `1` - Error en script
- `124` - Timeout (exceeded --time)
- `137` - OOM Killed (out of memory)

---

## ⏱️ Estimaciones de Tiempo

### Calcular walltime necesario

```bash
# Tu job de 1M muones típicamente tarda:
# - Geant4: ~5-10 min
# - makeHLTuple: ~5-10 min
# - POCA: ~2-5 min
# Total: ~15-25 min por seed

# Con buffer de seguridad:
--time=01:00:00  # 1 hora (conservador)
```

### Optimizar tiempo

- Reducir muones por job (ej: 500k en lugar de 1M)
- Usar `--ntasks > 1` si script soporta paralelismo
- Usar partición más rápida (si disponible)

---

## 🎯 Workflow Típico: Lanzar Pipeline Completa

```bash
# 1. Conectar
ssh dominguezs@login1.ifca.es

# 2. Limpiar quota
du -sh /gpfs/.../Muography_Denoising/MuonGeneration/data
rm -rf /gpfs/.../data/data_raw/*
quota -s

# 3. Lanzar simulación
cd /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/test
python3 loop_configuration_files.py

# 4. Monitorear
watch -n 5 'squeue -u dominguezs'

# 5. Una vez terminal, transferir datos locales
exit  # Salir del cluster
scp -r dominguezs@login1.ifca.es:/gpfs/.../merged_poca_data ~/Descargas/

# 6. Entrenar UNET (en local)
cd /home/samuel/Work/Muography_Denoising/MuonGeneration/UNETs
python3 create_h5_dataset.py
python3 UNET0_2D.py
```

---

## 📝 Ejemplos Prácticos

### Script: Lanzar 100 jobs de prueba

```bash
#!/bin/bash
for i in {1..100}; do
    sbatch --job-name=test_$i --wrap="sleep 10"
done

# Ver todos
squeue -u dominguezs | wc -l
```

### Script: Cancelar jobs que fallan (con patrón)

```bash
# Cancelar todos los jobs llamados "muon_seed*"
scancel -n "muon_seed*"
```

### Script: Esperar a que terminen todos tus jobs

```bash
#!/bin/bash
while true; do
    count=$(squeue -u dominguezs -h | wc -l)
    if [ $count -eq 0 ]; then
        echo "✓ Todos los jobs terminaron"
        break
    fi
    echo "Esperando... Jobs activos: $count"
    sleep 10
done
```

---

## 🆘 Help y Documentación

```bash
# Ayuda de cualquier comando
sbatch --help
squeue --help
scancel --help

# Documentación oficial (en el cluster)
man sbatch
man squeue

# Contactar admin
# Envía email a: slurm-admin@ifca.es (o tu institución)
```

---

## 📌 Cheat Sheet Rápido

```bash
# Ver mis jobs
squeue -u dominguezs

# Enviar job
sbatch script.sh

# Cancelar job
scancel 1234567

# Ver cuota
quota -s

# Ver espacio usado
du -sh /gpfs/users/dominguezs/Muography_Denoising

# Monitorear en tiempo real
watch -n 2 'squeue -u dominguezs'

# Limpiar disco
rm -rf /gpfs/users/dominguezs/Muography_Denoising/MuonGeneration/data/data_raw/*

# Ver detalles de job completado
sacct -j 1234567 --format=JobID,Elapsed,MaxRSS,State

# Cancelar todos mis jobs
scancel -u dominguezs
```

---

**Última actualización**: Abril 2026  
**Cluster**: IFCA (SLURM)  
**Usuario**: dominguezs
