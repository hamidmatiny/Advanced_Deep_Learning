# Quick Reference Guide
## Advanced Deep Learning Project

### 🚀 Running the Project

```bash
# Navigate to project directory
cd 05_Final_project

# Start Jupyter
jupyter notebook advanced_deep_learning_project.ipynb

# Or use Jupyter Lab
jupyter lab advanced_deep_learning_project.ipynb
```

### ⏱️ Execution Timeline

| Task | Duration | GPU | CPU |
|------|----------|-----|-----|
| Setup & Load Data | 2 min | 2 min | 2 min |
| From-Scratch Training (10 epochs) | 30 min | 2 hours |
| HF Fine-tuning (3 epochs) | 15 min | 1 hour |
| Generation & Analysis | 5 min | 5 min |
| **Total** | **50 min** | **~3 hours** |

### 🔧 Key Hyperparameters

```python
# From-Scratch Transformer
embed_dim = 64              # Embedding dimension
num_heads = 4               # Number of attention heads
num_layers = 3              # Depth of network
block_size = 128            # Context window
ff_dim = 256                # Feed-forward hidden size
dropout = 0.1               # Dropout rate
learning_rate = 3e-4        # Adam learning rate
num_epochs = 10             # Training epochs

# Hugging Face Fine-tuning
learning_rate_hf = 5e-5     # Lower learning rate
num_epochs_hf = 3           # Fewer epochs needed
batch_size = 8              # Mini-batch size
seq_length = 128            # Sequence length
```

### 📊 Model Specifications

#### From-Scratch Model
- **Type**: Decoder-only autoregressive transformer
- **Vocabulary**: 65 characters (Shakespeare)
- **Parameters**: ~680,000
- **Layers**: 3
- **Attention Heads**: 4
- **Forward Pass Size**: O(N²) where N = sequence length

#### Hugging Face Model (GPT-2)
- **Type**: Decoder-only transformer
- **Vocabulary**: 50,257 subword tokens
- **Parameters**: 124,439,808
- **Layers**: 12
- **Attention Heads**: 12
- **Inference Speed**: ~50 tokens/second on GPU

### 🎯 Generation Strategies

```python
# 1. Greedy Decoding (Deterministic)
idx_next = logits.argmax(dim=-1)

# 2. Top-K Sampling (Quality)
probs = F.softmax(logits, dim=-1)
top_k_probs, top_k_indices = torch.topk(probs, k=50)
idx_next = torch.multinomial(top_k_probs, num_samples=1)

# 3. Temperature Sampling (Diversity)
logits_scaled = logits / temperature  # Lower = more confident
probs = F.softmax(logits_scaled, dim=-1)
idx_next = torch.multinomial(probs, num_samples=1)
```

### 📈 Monitoring Training

```python
# Key metrics to track:
loss_train = []      # Per-epoch training loss
loss_val = []        # Per-epoch validation loss
grad_norm = []       # Gradient magnitude (for debugging)
learning_curve = []  # Loss over iterations

# Watch for:
✓ Smooth loss decrease → Good hyperparameters
✓ Oscillating loss → Learning rate too high
✓ No change in loss → Learning rate too low
✗ NaN values → Numerical instability (overflow/underflow)
```

### 🔍 Interpreting Attention Weights

```python
# Heatmap color coding:
Dark/Cool (Blue)     → Low attention (0.0)
Medium (Yellow)      → Moderate attention (0.5)
Bright (Red)         → High attention (1.0)

# What to look for:
→ Diagonal: Strong self-attention (causal masking)
→ Banding: Attention to nearby tokens (local patterns)
→ Specific peaks: Long-range dependencies
→ Uniform: Diffuse attention (less structured)
```

### 💾 Saving & Loading

```python
# Save from-scratch model
torch.save(model_scratch.state_dict(), 'model_scratch.pt')
torch.save({
    'epoch': epoch,
    'model_state': model_scratch.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'config': config,
}, 'checkpoint.pt')

# Load from-scratch model
model_scratch.load_state_dict(torch.load('model_scratch.pt'))

# Save Hugging Face model
model_hf_finetuned.save_pretrained('./model_hf_finetuned')

# Load Hugging Face model
model_loaded = AutoModelForCausalLM.from_pretrained('./model_hf_finetuned')
```

### 🐛 Debugging Checklist

- [ ] Data loaded correctly (check shapes)
- [ ] Model builds without error
- [ ] Single batch training works
- [ ] Loss is decreasing
- [ ] Validation loss follows training loss
- [ ] Attention weights sum to ~1.0
- [ ] Generated text is coherent
- [ ] No NaN/Inf values
- [ ] GPU memory is available
- [ ] Correct device (cuda/mps/cpu)

### 📝 Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| Out of Memory | Batch too large | Reduce batch_size or seq_length |
| Loss stays constant | Learning rate too low | Increase learning_rate |
| Loss explodes | Learning rate too high | Decrease learning_rate or use gradient clipping |
| Attention is NaN | Softmax numerical issue | Use LogSumExp or different attention implementation |
| Generation is gibberish | Undertrained model | Train longer or use pre-trained model |
| Slow training | CPU usage | Use GPU or optimize batch size |

### 📚 Key Files to Review

**From scratch implementation:**
- `Head` class - Single attention head
- `MultiHeadAttention` - Parallel attention
- `TransformerBlock` - Residual + attention + FFN
- `TransformerLanguageModel` - Full model

**Training loop:**
- `get_batch()` - Data loading
- `estimate_loss()` - Validation
- Training loop - Main optimization

**Generation & Analysis:**
- `generate()` method - Text generation
- `visualize_attention_scratch()` - Attention extraction
- `attention_weights` - HF attention extraction

### 🎓 Learning Suggestions

**Week 1: Understanding**
- Read "Attention Is All You Need" paper
- Study the attention mechanism code
- Visualize attention weights
- Understand causal masking

**Week 2: Implementation**
- Modify model architecture (add more layers)
- Change hyperparameters (embed_dim, heads, etc.)
- Experiment with different optimizers
- Try different datasets

**Week 3: Optimization**
- Implement learning rate scheduling
- Add mixed precision training
- Experiment with different attention variants
- Profile code for bottlenecks

**Week 4: Production**
- Quantize models for deployment
- Optimize inference speed
- Test on edge devices
- Build inference API

### 🌐 Resources

**Immediate:**
- Notebook comments and docstrings
- README.md (comprehensive overview)
- This quick reference guide

**Educational:**
- Original Transformer paper (https://arxiv.org/abs/1706.03762)
- NanoGPT repository (https://github.com/karpathy/nanoGPT)
- Hugging Face course (https://huggingface.co/course)

**Community:**
- Hugging Face Discord (https://discord.gg/JfAtqEZZVe)
- Reddit: r/MachineLearning, r/LanguageModels
- GitHub Issues on project repos

### 💡 Pro Tips

1. **Always use GPU**: Training is 50-100x faster
2. **Start small**: Use fewer epochs for testing
3. **Monitor everything**: Plot losses, attention, gradients
4. **Save checkpoints**: Recover from failures
5. **Version everything**: Track hyperparameters and results
6. **Document experiments**: Future you will thank you
7. **Visualize attention**: Understand what model is learning
8. **Test generation**: Immediate feedback on quality

### ✅ Success Criteria

- [ ] From-scratch model trains without errors
- [ ] Training loss decreases by >50%
- [ ] Fine-tuning completes in <30 minutes
- [ ] Generated text is readable
- [ ] Attention maps show interpretable patterns
- [ ] All visualizations render correctly
- [ ] Model can be saved and loaded
- [ ] README is clear and complete

---

**Last Updated**: January 2026
**Status**: ✅ Complete and tested
**Next Steps**: Run the notebook and explore!
