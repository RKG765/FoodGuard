# FoodGuard — DGX Kubernetes Training Pipeline (Advanced)
> **DGX Server:** `10.1.0.176` | **User:** `dgx-s-bmu-soet-230868`  
> **Your Namespace:** `dgx-s-bmu-soet-230868-restricted`  
> **Your GPU allocation:** 18 GB slice (MIG partition)

---

## ADVANCED UPGRADES IN THIS VERSION

1.  **Persistent Storage (PVC):** Replaced `emptyDir` with a Persistent Volume Claim (PVC). Data and checkpoints survive pod deletions. No more data loss risks.
2.  **Avoid OOM & 2x Faster Training:** Implemented Automatic Mixed Precision (AMP/fp16) and `num_workers` tuning to max out the MIG performance without crashing memory.

---

## Prerequisites Check (inside DGX SSH session)

```bash
ssh dgx-s-bmu-soet-230868@10.1.0.176

# Confirm your namespace and permissions
kubectl get pods     -n dgx-s-bmu-soet-230868-restricted
kubectl get pvc      -n dgx-s-bmu-soet-230868-restricted

# Check the GPU MIG availability on the worker node
kubectl describe node bmu-worker | grep -i mig
```

> ⚠️ **CRITICAL: GPU Resource Naming**
> Look at the output of the command above. Your pod manifest MUST use the exact name shown (e.g., `nvidia.com/mig-1g.18gb`). If you use the wrong name, your pod will stay in `Pending`.

---

## STAGE 1 — Transfer Local Files to DGX Headnode

Run from **your local laptop** (PowerShell / Git Bash):

```powershell
# Create workspace on headnode first
ssh dgx-s-bmu-soet-230868@10.1.0.176 "mkdir -p ~/foodguard/{data,checkpoints,logs,k8s}"

# Transfer source code and dataset
scp -r .\src         dgx-s-bmu-soet-230868@10.1.0.176:~/foodguard/
scp -r .\config      dgx-s-bmu-soet-230868@10.1.0.176:~/foodguard/
scp -r .\scripts     dgx-s-bmu-soet-230868@10.1.0.176:~/foodguard/
scp .\*.py           dgx-s-bmu-soet-230868@10.1.0.176:~/foodguard/
scp .\requirements.txt dgx-s-bmu-soet-230868@10.1.0.176:~/foodguard/
scp .\dataset_index.csv dgx-s-bmu-soet-230868@10.1.0.176:~/foodguard/

# Transfer dataset (compress first)
tar -czf dataset_4class.tar.gz dataset_4class/
scp dataset_4class.tar.gz dgx-s-bmu-soet-230868@10.1.0.176:~/foodguard/data/
```

SSH in and extract the dataset on the headnode:
```bash
ssh dgx-s-bmu-soet-230868@10.1.0.176
cd ~/foodguard/data
tar -xzf dataset_4class.tar.gz && rm dataset_4class.tar.gz
```

---

## STAGE 2 — Setup Persistent Storage & Create Pod

Instead of losing data when the pod dies, we create a **Persistent Volume Claim (PVC)**. This acts like a permanent hard drive for your workspace.

### 2.1 Write the PVC & Pod Manifest

```bash
cat > ~/foodguard/k8s/training-pod.yaml << 'YAML'
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: foodguard-pvc
  namespace: dgx-s-bmu-soet-230868-restricted
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi

---
apiVersion: v1
kind: Pod
metadata:
  name: foodguard-trainer
  namespace: dgx-s-bmu-soet-230868-restricted
  labels:
    app: foodguard
    stage: training
spec:
  restartPolicy: Never

  containers:
    - name: trainer
      image: pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
      imagePullPolicy: IfNotPresent

      command: ["sleep", "infinity"]

      resources:
        limits:
          nvidia.com/mig-1g.18gb: 1   # <-- MUST MATCH EXACTLY WITH `kubectl describe node`
          memory: "16Gi"
          cpu: "6"
        requests:
          cpu: "4"
          memory: "12Gi"

      volumeMounts:
        - name: workspace
          mountPath: /workspace

  volumes:
    - name: workspace
      persistentVolumeClaim:
        claimName: foodguard-pvc
YAML
```

### 2.2 Apply the PVC and Pod

```bash
kubectl apply -f ~/foodguard/k8s/training-pod.yaml \
  -n dgx-s-bmu-soet-230868-restricted

# Watch until Running (usually 30–90 seconds)
kubectl get pod foodguard-trainer \
  -n dgx-s-bmu-soet-230868-restricted -w
```

---

## STAGE 3 — Populate the Persistent Volume

Because you are using a PVC, **you only need to do this step ONCE**. If you delete the pod and recreate it tomorrow, all files will still be in `/workspace`!

```bash
NS="dgx-s-bmu-soet-230868-restricted"

# Push files from headnode into the persistent volume inside the pod
kubectl cp ~/foodguard/src           foodguard-trainer:/workspace/src           -n $NS
kubectl cp ~/foodguard/config        foodguard-trainer:/workspace/config        -n $NS
kubectl cp ~/foodguard/scripts       foodguard-trainer:/workspace/scripts       -n $NS
kubectl cp ~/foodguard/train_4class_detector.py  foodguard-trainer:/workspace/ -n $NS
kubectl cp ~/foodguard/inference.py              foodguard-trainer:/workspace/ -n $NS
kubectl cp ~/foodguard/evaluate.py               foodguard-trainer:/workspace/ -n $NS
kubectl cp ~/foodguard/requirements.txt          foodguard-trainer:/workspace/ -n $NS
kubectl cp ~/foodguard/dataset_index.csv         foodguard-trainer:/workspace/ -n $NS

# Create data directory and push dataset
kubectl exec -it foodguard-trainer -n $NS -- mkdir -p /workspace/data
kubectl cp ~/foodguard/data/dataset_4class  foodguard-trainer:/workspace/data/dataset_4class  -n $NS
```

---

## STAGE 4 — Install Dependencies & Pre-flight Check

### 4.1 Install Python dependencies

```bash
NS="dgx-s-bmu-soet-230868-restricted"

kubectl exec -it foodguard-trainer -n $NS -- bash -c "
  pip install --upgrade pip
  pip install --no-cache-dir -r /workspace/requirements.txt
"
```

### 4.2 Run Health Check

```bash
kubectl exec -it foodguard-trainer -n $NS -- python3 -c \
  "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"
```

---

## STAGE 5 — Run High-Performance Training

To run 2x faster and avoid OOM, we will apply **Automatic Mixed Precision (AMP)** via PyTorch and optimize DataLoaders.

> **Code Requirement**: Ensure your `train_4class_detector.py` uses `torch.cuda.amp.autocast()` and a `GradScaler`. If it doesn't have an explicit `--fp16` flag, standard PyTorch 2.x runs quite efficiently, but you must ensure your `num_workers` is set to `4` in your `DataLoader` to feed the MIG GPU fast enough.

```bash
# Exec and launch in background using nohup
kubectl exec foodguard-trainer -n $NS -- bash -c "
  mkdir -p /workspace/checkpoints /workspace/logs
  
  nohup python3 /workspace/train_4class_detector.py \
    --data_dir=/workspace/data/dataset_4class \
    --output_dir=/workspace/checkpoints \
    --epochs=50 \
    --batch_size=32 \
    --num_workers=4 \
    --fp16 \
    --lr=1e-4 \
  > /workspace/logs/train.log 2>&1 &
  
  echo 'Training PID:' \$!
"
```
*Note: Because of `--fp16` (Mixed Precision), we can safely use `--batch_size=32` without OOM on the 18GB slice. If your code does not support `--fp16` natively, drop `--batch_size=16`.*

**Stream the log any time:**
```bash
kubectl exec -it foodguard-trainer -n $NS -- tail -f /workspace/logs/train.log
```

**Watch GPU utilization:**
```bash
kubectl exec -it foodguard-trainer -n $NS -- watch -n 2 nvidia-smi
```

---

## STAGE 6 — Retrieve Checkpoints

```bash
NS="dgx-s-bmu-soet-230868-restricted"

# Checkpoints survive pod deletion now, but you still want them local!
# Copy checkpoints FROM pod PVC → DGX headnode
kubectl cp foodguard-trainer:/workspace/checkpoints ~/foodguard/checkpoints -n $NS

# Then from your LOCAL laptop:
scp -r dgx-s-bmu-soet-230868@10.1.0.176:~/foodguard/checkpoints/ ./checkpoints_dgx/
```

---

## STAGE 7 — Cleanup

```bash
NS="dgx-s-bmu-soet-230868-restricted"

# Delete pod — this releases your GPU slice for others
kubectl delete pod foodguard-trainer -n $NS

# DO NOT delete the PVC unless you are completely done with the project!
# If you do: kubectl delete pvc foodguard-pvc -n $NS  <-- ERASES EVERYTHING
```
