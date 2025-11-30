# PATE Paper Experimental Parameters Summary

## 1. Datasets Used

### MNIST
- Train set: 60,000 images  
- Test set: 10,000 images  
- Student training subset: 9,000 images (100–1,000 labeled)

---

## 2. Teacher Ensemble Parameters

### Number of Teachers
- Tested: 10, 100, 250  
- **Main configuration: 250 teachers**

### Teacher Architectures
#### MNIST Teachers
- CNN:
  - 2 convolutional layers  
  - Max pooling  
  - 1 fully connected layer  
- Baseline accuracy: 99.18%


### Teacher Accuracy (Partition-Trained)
- MNIST: ~83.86%  

### Vote Gap
- >60% for n ≥ 100  
- Ensures robust noisy aggregation

---

## 3. Noisy Aggregation Parameters

### Mechanism
- Noisy plurality vote
- Laplace noise added to teacher vote counts

### Noise Distribution
- Laplace(scale = 1/γ)

### γ Values Tested
- 0.01 to 1.0

### Main Experiment Noise Level
- Laplacian scale = 20  
- γ = 0.05  
- Per-query cost: ε = 0.05

---

## 4. Student Model Parameters

### Student Type
- GAN-based semi-supervised model (Salimans et al., 2016)

### Labeled Data Used
#### MNIST
- 100, 500, or 1,000 labels


### Student Accuracy
- MNIST: **98.00%**

---

## 5. Differential Privacy Parameters

### DP Method
- Moments Accountant (Abadi et al., 2016)

### Final Privacy Guarantees

| Dataset | ε | δ | Queries | Accuracy |
|--------|-------|--------------|----------|-----------|
| MNIST | **2.04** | 1e-5 | 100 | 98.00% |
| MNIST | 8.03 | 1e-5 | 1000 | 98.10% |


---

## 6. Baselines

### Non-private Baselines
- MNIST: 99.18%  

### Comparison to Other DP Work
- DP-SGD: 97% @ ε = 8  
- PATE: 98% @ ε = 2.04 (improved)

---

## 7. Additional Experiments (Appendix C)

### Tabular Datasets
- **Adult**, **Diabetes**

### Settings
- 250 random forest teachers  
- Laplace noise scale = 0.05  
- 500 private labels

### Accuracy
- Adult: **83% @ ε = 2.66**  
- Diabetes: **93.94% @ ε = 1.44**

---

## One-Page Summary Checklist

### Teachers
- 250 teachers  
- CNNs for MNIST, RFs for tabular  
- Teacher accuracy ~83%

### Aggregation
- Laplace noise scale = 20  
- γ = 0.05  
- ε per query = 0.05

### Student
- Semi-supervised GAN  
- 100–1,000 labels (MNIST)  


### DP
- Moments Accountant  
- MNIST: ε = 2.04  

### Accuracy
- MNIST: 98%  

