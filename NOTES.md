# Overfitting & Underfitting in Neural Networks

Neural networks are prone to **overfitting**.

### Why?

Neural networks are **high-capacity models** — they can learn very complex **patterns**, including **noise** in the training data.

The decision boundary **fits the data too well**. It starts becoming **highly non-linear** to adapt to small fluctuations in the data.

---

### 📈 Causes of Overfitting:
- Too many layers or neurons: High capacity models can learn noise.
- Not enough training data
- Too many training epochs

### 🚨 Signs of Overfitting:
- Low training error, but
- High validation/test error

> We need to make sure that the NN doesn’t fit the noise in the data.

---

Neural networks can also be prone to **underfitting**, though it's less common in practice.

Underfitting happens when a neural network fails to learn the underlying patterns in the data.  
It performs poorly on both the training and test sets.

---

### 🧱 Causes of Underfitting:
- Insufficient training: Not enough epochs
- Model too simple: The architecture has too few layers or neurons

### 🛑 Signs of Underfitting:
- Training Loss → High  
- Validation Loss → High

