# MiniBioBERT: A Lightweight BERT Model for Biomedical Text Classification

A minimal implementation of BERT in Julia with architectural improvements for biomedical text classification. This project demonstrates transformer-based NLP on a small dataset with optimized training techniques.

## 📋 Overview

MiniBioBERT is a lightweight transformer model designed to classify biomedical texts into two categories:
- **Treatment**: Medical interventions, drugs, procedures
- **Symptom**: Medical conditions, complaints, signs

The model combines:
- **Multi-Head Attention**: 4 attention heads for diverse feature extraction
- **Sinusoidal Positional Encoding**: Learned position representations for sequence order
- **Feed-Forward Networks**: Dense layers with dropout regularization
- **Layer Normalization**: Stable training with residual connections
- **Batch Processing**: Efficient mini-batch training with DataLoaders

## 🎯 Key Improvements Over Original

| Feature | Original | Improved |
|---------|----------|----------|
| Embedding Dimension | 8 | **32** |
| FFN Hidden Dimension | 8 | **64** |
| Attention Heads | 1 | **4** |
| Positional Encoding | Random | **Sinusoidal** |
| Training | Sample-by-sample | **Batch Processing** |
| Softmax | Redundant (double) | **Removed** |
| History Tracking | None | **Loss & Accuracy** |
| Regularization | None | **Dropout (0.1)** |

## 🏗️ Model Architecture

```
Input (vocab_ids)
    ↓
Embedding Layer (100 → 32)
    ↓
Positional Embedding (sinusoidal, max_len=8)
    ↓
Transformer Block:
  ├─ Multi-Head Attention (4 heads, 32→32)
  ├─ Layer Norm + Residual
  ├─ FFN (32→64→32 with Dropout)
  └─ Layer Norm + Residual
    ↓
Global Average Pooling (seq_len → 1)
    ↓
Classification Head (32 → 2)
    ↓
Output (logits for 2 classes)
```

## 📦 Requirements

- Julia 1.10+
- Flux.jl (deep learning)
- MLUtils.jl (data utilities, DataLoader)
- Optimisers.jl (Adam optimizer)
- OneHotArrays.jl (one-hot encoding)
- StatsBase.jl (statistics)
- CSV.jl & DataFrames.jl (data handling)
- NNlib.jl (batched operations)

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/KashishhKhann/MiniBioBERT.git
cd MiniBioBERT
```

### 2. Install Julia Dependencies
```bash
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

This will install all packages listed in `Project.toml`.

## 🏃 Running the Code

### Run Full Pipeline (10-Fold Cross-Validation)
```bash
julia --project=. mini_bio_bert_v1.1.jl
```

This will:
1. Load 20 biomedical sample sentences
2. Build vocabulary (72 unique tokens)
3. Perform 10-fold cross-validation
4. Train model for 200 epochs per fold
5. Display summary statistics with confidence intervals
6. Test on new unseen sentences

### Expected Output
```
Initializing MiniBioBERT with 10-Fold Cross-Validation...
Sample sentences: 20
Vocabulary size: 72
Total parameters: 5342

Starting 10-fold cross-validation...
Data split into 10 folds:
  Fold 1: 2 samples
  Fold 2: 2 samples
  ... (8 more folds)

Training fold 1/10...
  Training samples: 18, Test samples: 2
  Fold 1 results: Train Acc: 1.0, Test Acc: 1.0

[Additional folds...]

============================================================
CROSS-VALIDATION RESULTS
============================================================
Mean Test Accuracy: 0.75 ± 0.3536
95% Confidence Interval: [0.4971, 1.0029]

Training final model on full dataset...
  Epoch 40 | Loss: 0.0045 | Acc: 1.0
  Epoch 200 | Loss: 0.0002 | Acc: 1.0

Testing final model with new sentences:
'Paracetamol reduces fever' → treatment (confidence: 0.678)
'Chemotherapy destroys cancer cells' → treatment (confidence: 0.995)
'Radiotherapy treats tumors' → treatment (confidence: 0.968)
```

## 🔧 Configuration

Edit constants in `mini_bio_bert_v1.1.jl` to customize:

```julia
const VOCAB_SIZE = 100        # Maximum vocabulary size
const MAX_LEN = 8             # Maximum sequence length
const EMBED_DIM = 32          # Embedding dimension
const FFN_DIM = 64            # Feed-forward hidden dimension
const DROPOUT_RATE = 0.1      # Dropout probability
const NUM_HEADS = 4           # Number of attention heads
const NUM_CLASSES = 2         # Number of output classes
```

## 📊 Training Parameters

```julia
train_model!(model, X_train, Y_train; 
    epochs=200,               # Training epochs per fold
    lr=0.001,                 # Learning rate (Adam)
    batch_size=4,             # Mini-batch size
    verbose=false             # Print training progress
)
```

## 🔬 Implementation Details

### Multi-Head Attention
```julia
function (m::MultiHeadAttention)(x)
    # Split embedding into multiple heads
    # Compute scaled dot-product attention
    # Concatenate and project back
end
```

### Sinusoidal Positional Encoding
```julia
pos_embed[i+1, pos+1, 1] = sin(pos / (10000^(i/embed_dim)))  # Even indices
pos_embed[i+1, pos+1, 1] = cos(pos / (10000^((i-1)/embed_dim)))  # Odd indices
```

### Batch Processing with DataLoader
```julia
loader = MLUtils.DataLoader((X_train, Y_train), 
    batchsize=batch_size, shuffle=true)
for (x_batch, y_batch) in loader
    # Process batches efficiently
end
```

## 📈 Results & Analysis

The model achieves:
- **Cross-validation**: Stratified 10-fold with 95% confidence intervals
- **Metrics**: Accuracy, with per-fold and aggregate statistics
- **Generalization**: Tests on held-out biomedical sentences

**Actual Training Results (10-Fold Cross-Validation):**

Individual Fold Results:
```
Fold 1: Train=1.0, Test=1.0 (18 train, 2 test) ✓
Fold 2: Train=1.0, Test=1.0 (18 train, 2 test) ✓
Fold 3: Train=1.0, Test=0.5 (18 train, 2 test)
Fold 4: Train=1.0, Test=0.5 (18 train, 2 test)
Fold 5: Train=1.0, Test=0.0 (18 train, 2 test)
Fold 6: Train=1.0, Test=1.0 (18 train, 2 test) ✓
Fold 7: Train=1.0, Test=1.0 (18 train, 2 test) ✓
Fold 8: Train=1.0, Test=0.5 (18 train, 2 test)
Fold 9: Train=1.0, Test=1.0 (18 train, 2 test) ✓
Fold 10: Train=1.0, Test=1.0 (18 train, 2 test) ✓
```

**Summary Statistics:**
```
Mean Test Accuracy:      0.75 ± 0.3536
Min Test Accuracy:       0.0
Max Test Accuracy:       1.0
Accuracy Range:          1.0
95% Confidence Interval: [0.4971, 1.0029]
```

**Final Model Training (Full Dataset):**
```
Epoch 40   | Loss: 0.0045 | Acc: 1.0
Epoch 80   | Loss: 0.0011 | Acc: 1.0
Epoch 120  | Loss: 0.0005 | Acc: 1.0
Epoch 160  | Loss: 0.0003 | Acc: 1.0
Epoch 200  | Loss: 0.0002 | Acc: 1.0  ← Convergence achieved
```

**Note on Sample Dataset**: The included 20-sample biomedical dataset is for demonstration only. The high training accuracy (100%) and variable test accuracy reflect the small dataset size. For production use, train on larger, more diverse biomedical corpora.

**Example Predictions from Final Model:**
| Sentence | Prediction | Confidence |
|----------|-----------|------------|
| "Paracetamol reduces fever" | Treatment | 0.678 |
| "Chemotherapy destroys cancer cells" | Treatment | 0.995 |
| "Radiotherapy treats tumors" | Treatment | 0.968 |
| "Vomiting is side effect" | Treatment | 0.865 |

## 🔄 Evaluation Protocol

1. **Data Splitting**: 10-fold cross-validation
2. **Training**: 200 epochs with batch_size=4
3. **Optimization**: Adam optimizer (lr=0.001)
4. **Metrics**: Accuracy averaged over batches
5. **Statistical Analysis**: Mean ± std with 95% CI

## 📁 Project Structure

```
MiniBioBERT/
├── mini_bio_bert_v1.1.jl    # Main model and training code
├── Project.toml             # Julia project dependencies
└── README.md               # This file
```

## 🔮 Future Improvements

- [ ] Larger biomedical datasets
- [ ] Additional classification tasks (NER, relation extraction)
- [ ] Pretrained embeddings (BioBERT, SciBERT)
- [ ] Attention visualization
- [ ] Model saving/loading
- [ ] GPU acceleration (CUDA)
- [ ] Integration with MLJ.jl

## 📚 References

- **Attention Is All You Need**: Vaswani et al. (2017)
  - https://arxiv.org/abs/1706.03762
- **BERT: Pre-training of Deep Bidirectional Transformers**: Devlin et al. (2018)
  - https://arxiv.org/abs/1810.04805
- **Flux.jl Documentation**: https://fluxml.ai/
- **Julia Language**: https://julialang.org/

## 💡 Tips for Customization

### Add More Training Data
Modify `create_sample_data()` to include more biomedical sentences:
```julia
function create_sample_data()
    sentences = [
        "Your sentences here",
        # ... more samples
    ]
    labels = [1, 0, 1, ...]  # 1 for treatment, 0 for symptom
    return sentences, labels
end
```

### Adjust Model Size
```julia
const EMBED_DIM = 64   # Larger embeddings
const FFN_DIM = 256    # Larger hidden layer
const NUM_HEADS = 8    # More attention heads
```

### Change Training Strategy
```julia
k_fold_cross_validation(sentences, labels, vocab,
    k=5,                # 5-fold instead of 10
    epochs=500,         # More epochs
    lr=0.0001,          # Lower learning rate
    batch_size=2        # Smaller batches
)
```

## 📝 License

MIT License - See LICENSE file for details

## 👤 Author

[KashishhKhann](https://github.com/KashishhKhann)

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report issues
- Suggest improvements
- Submit pull requests
- Add more biomedical data

## ⭐ Citation

If you use MiniBioBERT in your research, please cite:

```bibtex
@software{minibibert2026,
  title={MiniBioBERT: A Lightweight BERT Model for Biomedical Text Classification},
  author={Khann, Kashish},
  year={2026},
  url={https://github.com/KashishhKhann/MiniBioBERT}
}
```

---

**Questions?** Open an issue on GitHub or check the Julia discourse community!
