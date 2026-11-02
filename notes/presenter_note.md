# Presenter notes

What to say and when, for the two 45-minute sessions. Minutes are rough targets.
🖐 = students code. Don't skip these; they're where most of the learning happens.

## Before class

- [ ] Open both notebooks in Colab from the README badges and run them once end to end. Notebook 2 is already set to use a GPU, but check that it reports `device: cuda`.
- [ ] Have a backup copy of the satellite data ready in case Kaggle is down or slow. Zip the `data/` folder (the one holding `cloudy/ desert/ green_area/ water/`) and upload it somewhere only the class can see, e.g. Google Drive with "anyone with the link". Paste a **direct download** link into `BACKUP_URL` in notebook 2. For Google Drive that's `https://drive.google.com/uc?export=download&id=<FILE_ID>`. Test it once.
  The dataset's license is "© Original Authors", so don't publish the images in a public repo or release.
- [ ] Results from a test run (CPU): MNIST ≈ 97% test accuracy after 3 epochs; satellite CNN ≈ 93% test accuracy after 10 epochs. Your numbers will differ slightly on a GPU.

---

## Session 1: Tensors and a simple NN

### Intro & setup (0–3 min)
- PyTorch = NumPy-like arrays + GPU support + automatic gradients. That's basically the whole library; the rest is building blocks on top.
- Have everyone run the first cell now, so slow Colab startups don't hold you up later.

### Tensors (3–13 min)
- Keep it brisk: they know NumPy, so focus on what's different.
- **dtype:** integers give `int64`, models need `float32`. A classic error when building tensors from data.
- **reshape / -1:** they'll flatten images in 10 minutes.
- **Broadcasting:** point at the `m + v` output: the vector was added to every row.
- **`dim`:** "dim says which dimension disappears." Point at `argmax(dim=1)`: this is exactly how we'll get predictions from a model later.
- Likely question: *"`view` vs `reshape`?"* Same idea; `reshape` always works, so use it.

### 🖐 Shape puzzle (13–18 min)
- Give them ~4 minutes. Walk around / watch the chat.
- Hint if stuck: "Which dimension do you want to keep? Put `-1` for the rest." Then: "the mean over which dim?"
- Go through the solution together; ask someone for their answer first.

### Autograd (18–25 min)
- `x.grad` = 6: check that it matches the derivative 2x they know by hand. Autograd isn't magic; it applies the chain rule.
- Line fit: read the five numbered steps out loud. **This is the most important slide of the session**; everything after is these steps with nicer tools.
- *"Why `torch.no_grad()` in the update?"* We don't want the update itself to be recorded as part of the computation.
- *"Why zero the gradients?"* PyTorch adds to `.grad` on every `backward()`. Without zeroing, each step would use the sum of all past gradients.
- b ends up around 0.84, not exactly 1: the noise makes the best line differ a bit from the true one.
- The `nn.Linear` + `optim.SGD` cell: point out the same five steps, just shorter. *(Skip if late.)*

### MNIST MLP (25–42 min)
- Batch shape `[64, 1, 28, 28]`: name each dimension out loud. The order [batch, channels, height, width] comes back in session 2.
- **nn.Module:** `__init__` = what layers exist, `forward` = how data flows through them. `super().__init__()` is required: forgetting it gives a confusing error.
- Point at the parameter count: 784·128 + 128 + 128·10 + 10 = 101,770. Most weights are in the first layer, because every pixel connects to every neuron. This comes up again in session 2.
- *"Why no softmax?"* `CrossEntropyLoss` applies it internally (in a more numerically stable way). Adding a softmax yourself is a common bug.
- `model.train()` / `model.eval()`: just plant the seed ("matters for dropout, next time").
- Training takes ~20 s. While it runs, ask: "what do you expect the accuracy to be?"
- Predictions grid: red = wrong. Some mistakes are understandable even for humans.
- Exercise 2 (extra layer / learning rates) is optional: homework if you're out of time.

### Recap & teaser (42–45 min)
- Repeat the recipe: forward → loss → zero_grad → backward → step.
- Teaser: shift a digit 3 pixels to the right and, for the MLP, it's a completely different input. Ask: "how would *you* design a network that doesn't care where the pattern is?"

---

## Session 2: CNN on satellite images

### Recap + GPU check (0–3 min)
- Ask them to run the first two cells (GPU check + download) while you recap. The download takes a few seconds.
- Quick recap: tensors → `nn.Module` → the five steps. "Today the loop stays the same; the model and the data change."

### The data (3–12 min)
- **Know your data:** the size quirk is a nice real-world lesson. Cloudy/desert are 256×256, green_area/water are 64×64 (and cloudy images are stored in CMYK colour; `ImageFolder` converts them to RGB). After resizing, the downscaled images look smoother, which the model could exploit. Ask: "is a model that gets 93% here necessarily good?"
- **Augmentation:** "Is a flipped satellite image still water? Yes, so we get extra data for free." Ask: "would flips be OK for MNIST?" (No: a flipped 6 can look like a 9.)
- **Train/val/test:** the val set picks the best epoch; the test set is touched once. If you tune on test, your score is optimistic.
- **TransformDataset:** first state the problem ("all three splits share one ImageFolder, so they get one transform"), then the fix. Emphasize the general pattern: `__len__` + `__getitem__` is all you need to load *any* data.
- `permute`: matplotlib wants channels last.
- *(Skip if late: the 4×5 sample grid; just show the augmented batch.)*

### Convolutions + model (12–22 min)
- Edge demo: "a 3×3 grid of numbers turns an image into an edge map. A CNN learns its own grids of numbers." *(Skip if late.)*
- `conv.weight.shape` = `[16, 3, 3, 3]`: 16 filters, each looking at all 3 colour channels.
- Output-size formula: show that padding 1 with kernel 3 keeps the size.
- **🖐 Plan the network:** give ~3 minutes. Hint: "each pool halves height and width; channels = out_channels of the last conv." Answer: 32·16·16 = 8192.
- Run the layer-by-layer shape trace: "this is how you debug a 'mat1 and mat2 shapes cannot be multiplied' error."
- Parameter count: the conv layers have ~5k weights, while an MLP's first layer on raw pixels alone would have 1.5M.

### Training (22–35 min)
- **🖐 Fill in the five steps:** ~3 minutes, from memory. Then scroll to the solution and compare.
- `train()`: the only new idea is saving the best checkpoint (`state_dict` = a dict of all weight tensors).
- Training takes ~30 s on CPU, less on a T4. While it runs, explain what the printed columns mean.
- Curves: validation accuracy jumps around a bit (the test run had a dip at epoch 8). The val set is small (845 images), so a few images change the number noticeably. This is also why we keep the *best* checkpoint, not the last.
- *"Why is val loss sometimes lower than train loss?"* Train loss is averaged over the epoch while the weights are still changing, and train images are augmented (harder).

### Evaluation (35–40 min)
- Test accuracy ≈ 93%. Per-class: cloudy/desert are higher; green_area and water are the weakest and get confused with each other. Show the misclassified grid: many of those images are dark blue-green and honestly ambiguous.
- *(Skip if late: the misclassified grid.)*

### Buffer (40–42 min)
- Questions, or catch-up if training was slow.

### Wrap-up (42–45 min)
- Exercises 3 (BatchNorm/Dropout) and 4 (no augmentation) are for home.
- Transfer learning: one sentence. "In practice you'd start from a pretrained ResNet; same training loop, much better features."
