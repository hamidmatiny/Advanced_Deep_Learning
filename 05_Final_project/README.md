# 🔥 Advanced Deep Learning Project
## Combining From-Scratch Transformers with Hugging Face Fine-tuning

A comprehensive hands-on project demonstrating the complete pipeline from building a Transformer architecture from scratch to fine-tuning state-of-the-art pre-trained models.

---

## 📋 Project Overview

This notebook combines **two complementary approaches** to NLP:

### Part 1: From-Scratch Transformer Implementation
- **Inspired by**: Andrej Karpathy's NanoGPT approach
- **Components**: Embeddings, Positional Encoding, Multi-Head Attention, Feed-Forward Networks
- **Features**: Causal masking, Layer Normalization, Residual Connections
- **Purpose**: Educational understanding of transformer internals

### Part 2: Hugging Face Model Fine-tuning
- **Base Model**: GPT-2 (pretrained on ~40GB of text)
- **Task**: Domain adaptation to Shakespeare text
- **Approach**: Transfer learning with fine-tuning
- **Benefits**: Faster convergence, better quality, production-ready

---

## 🎯 Key Features

### 1. **Complete Transformer Architecture**
```
Token Input
    ↓
[Token Embedding + Positional Encoding]
    ↓
[Multi-Head Attention Block]
    ↓
[Feed-Forward Network]
    ↓
[Stack N Blocks]
    ↓
Output Logits → Next Token Prediction
```

### 2. **Attention Mechanism Visualization**
- Heatmaps showing which tokens the model attends to
- Multi-head attention patterns across 4+ heads
- Diagonal (self-attention) analysis
- Future token dependency tracking

### 3. **Before/After Comparison**
- Training curves for both models
- Loss reduction metrics
- Parameter count comparison
- Generation quality samples

### 4. **Text Generation**
- Greedy decoding
- Top-K sampling
- Temperature-controlled randomness
- Shakespeare domain-specific output

---

## 📊 Architecture Comparison

| Aspect | From-Scratch | Hugging Face (GPT-2) |
|--------|--------------|---------------------|
| **Parameters** | ~680K | ~124M (182x larger) |
| **Training Data** | Shakespeare only | 40GB+ of web text |
| **Training Time** | ~1 hour | 5+ days (pretrained) |
| **Quality** | Educational | Production-ready |
| **Code Clarity** | Fully transparent | Black-box but optimized |
| **Best Use** | Learning | Real-world deployment |

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install torch transformers datasets matplotlib seaborn numpy scikit-learn
```

### 2. Run the Notebook
```bash
jupyter notebook advanced_deep_learning_project.ipynb
```

### 3. Expected Runtime
- **From-scratch training**: ~30 minutes (10 epochs)
- **HF fine-tuning**: ~15 minutes (3 epochs)
- **Total**: ~1 hour on GPU, ~3+ hours on CPU

### 4. Output Files Generated
```
training_progress_scratch.png      # Loss curves for from-scratch model
finetuning_progress_hf.png         # Loss curves for fine-tuned model
model_comparison.png               # Side-by-side comparison
attention_weights_scratch.png      # Attention visualization (4 heads)
attention_weights_hf.png           # Attention visualization (8 heads)
model_scratch_best.pt              # Best checkpoint
./model_hf_finetuned/              # Fine-tuned model directory
```

---

## 📚 Learning Outcomes

After completing this project, you'll understand:

### Transformers
- ✅ Token and positional embeddings
- ✅ Scaled dot-product attention
- ✅ Multi-head attention mechanisms
- ✅ Encoder-only vs decoder architectures
- ✅ Causal masking for autoregressive models

### Training
- ✅ How to build training loops from scratch
- ✅ Loss tracking and validation
- ✅ Model checkpointing and resumption
- ✅ Hyperparameter tuning strategies
- ✅ Early stopping and learning rate scheduling

### Transfer Learning
- ✅ Loading pre-trained models
- ✅ Fine-tuning vs full training trade-offs
- ✅ Domain adaptation techniques
- ✅ When to use transfer learning
- ✅ Practical deployment considerations

### Attention Analysis
- ✅ How to extract attention weights
- ✅ Interpreting attention patterns
- ✅ Visualizing model decisions
- ✅ Debugging model behavior
- ✅ Understanding model alignment

---

## 🔍 Key Code Sections

### 1. Attention Mechanism
```python
class Head(nn.Module):
    """Scaled Dot-Product Attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V"""
    
    def forward(self, x):
        scores = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        att = F.softmax(scores, dim=-1)
        return att @ v
```

### 2. Model Training
```python
for epoch in range(num_epochs):
    # Training batch
    logits, loss = model(X, Y)
    loss.backward()
    optimizer.step()
    
    # Validation
    with torch.no_grad():
        val_logits, val_loss = model(X_val, Y_val)
```

### 3. Text Generation
```python
@torch.no_grad()
def generate(self, idx, max_tokens=100):
    for _ in range(max_tokens):
        logits, _ = self(idx[:, -self.block_size:])
        probs = F.softmax(logits[:, -1, :], dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

---

## 📈 Expected Results

### From-Scratch Transformer
- Initial loss: ~4.5 (random chance)
- Final loss: ~2.1 (strong learning)
- **Loss reduction: ~53%**

### Hugging Face Fine-tuning
- Initial loss: ~3.2 (pre-trained, cold start)
- Final loss: ~1.8 (well-adapted)
- **Loss reduction: ~44%**

### Generation Quality
- **Before fine-tuning**: Generic English text
- **After fine-tuning**: Shakespeare-like prose with archaic language

---

## 🎓 Mathematical Background

### Attention Formula
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- **Q** (Query): What are we looking for?
- **K** (Key): What information do we have?
- **V** (Value): What information do we return?
- **√d_k**: Scaling factor to prevent saturation

### Multi-Head Attention
$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,...,\text{head}_h)W^O$$

Where each head computes attention independently, allowing the model to:
- Attend to different representation subspaces
- Focus on both local and global patterns
- Capture diverse linguistic phenomena

---

## 💡 Practical Applications

### Use From-Scratch Implementation For:
1. **Learning**: Understanding how transformers work
2. **Research**: Experimenting with attention mechanisms
3. **Prototyping**: Testing novel architectures
4. **Small datasets**: When pre-training isn't applicable

### Use Hugging Face Fine-tuning For:
1. **Production**: Real-world deployments
2. **Large-scale**: When scaling is important
3. **Fast iteration**: Quick experimentation
4. **Domain adaptation**: Specializing models
5. **Cost efficiency**: Leveraging pre-trained weights

---

## 🔗 References

### Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al. (2017)
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - Radford et al. (2019)

### Implementations
- [NanoGPT](https://github.com/karpathy/nanoGPT) - Andrej Karpathy's educational implementation
- [Hugging Face Transformers](https://github.com/huggingface/transformers) - Production-grade library
- [Neural Network from scratch](https://github.com/karpathy/micrograd) - Minimal autograd engine

### Courses
- [Stanford CS224N](https://web.stanford.edu/class/cs224n/) - NLP with Deep Learning
- [Hugging Face Course](https://huggingface.co/course/) - Free comprehensive guide
- [Karpathy's Video Lecture](https://www.youtube.com/watch?v=kCc8FmEb1KY) - Let's build GPT

---

## 🛠️ Troubleshooting

### Out of Memory
- Reduce `batch_size` (8 → 4)
- Reduce `seq_length` (128 → 64)
- Use CPU instead of GPU
- Reduce model `embed_dim` (64 → 32)

### Training is Too Slow
- Increase `batch_size` if memory allows
- Use GPU (CUDA/Metal)
- Reduce number of epochs
- Reduce dataset size for testing

### Poor Generation Quality
- Train for more epochs
- Use larger model
- Increase learning rate
- Add more training data

### Attention Visualization Not Showing
- Ensure model has completed training
- Check if attention weights are being extracted
- Verify input sequence length

---

## 📝 Project Structure

```
05_Final_project/
├── advanced_deep_learning_project.ipynb    # Main notebook
├── README.md                                # This file
├── input.txt                                # Shakespeare dataset (auto-downloaded)
├── training_progress_scratch.png           # Generated plots
├── finetuning_progress_hf.png
├── model_comparison.png
├── attention_weights_scratch.png
├── attention_weights_hf.png
├── model_scratch_best.pt                   # Model checkpoints
└── model_hf_finetuned/
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer_config.json
    ├── tokenizer.json
    ├── merges.txt
    └── vocab.json
```

---

## 🎉 Conclusion

This project bridges the gap between **theoretical understanding** and **practical application** of transformers. By implementing both from scratch and using state-of-the-art libraries, you'll gain intuition about how modern NLP systems work and when to use each approach.

**Key Takeaway**: Transformers are powerful but not magic. With proper understanding of attention mechanisms and training techniques, you can build and customize them for your specific needs!

---

## 📧 Questions & Feedback

For issues or suggestions, consider:
- Reviewing the referenced papers
- Exploring the Hugging Face documentation
- Experimenting with different hyperparameters
- Extending to other domains (code, chemistry, etc.)

Happy Learning! 🚀
