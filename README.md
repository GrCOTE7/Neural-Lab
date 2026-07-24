# 🧠 Neural Lab v12

> **A visual playground to understand neural networks — built from scratch in a single HTML file.**

<div align="center">

![Neural Lab Banner](screenshots/banner.png)

**Train. Visualise. Experiment. Understand.**

[Features](#-features) • [Screenshots](#-screenshots) • [Quick Start](#-quick-start) • [How it works](#-how-it-works) • [Try it online](https://sudo-luca.github.io/Neural-Lab/Neural-Lab-v12.html)

</div>

---

## ✨ What is Neural Lab?

Neural Lab is an interactive deep learning simulator designed to make neural networks understandable.

No framework.  
No installation.  
No server.

Just open **one HTML file** and start experimenting.

Every computation happens directly in your browser:

- Forward propagation
- Backpropagation
- Gradient calculation
- Weight updates
- Optimiser behaviour
- Decision boundaries
- Network visualisation

Built with **pure HTML + CSS + Vanilla JavaScript**.

> 🌐 **Language note:** the interface is in French by default, with a built-in FR ⇄ EN translation switch (in the Options tab) if you prefer English.

---

## 🎥 See it in action

Watch a neural network learn in real time:
- neurons activating
- weights changing
- loss decreasing
- decision boundaries evolving

---

# 🚀 Features

## 🕸️ Visual Neural Network Playground

Create and observe multilayer perceptrons directly on the canvas.

![Network view](screenshots/network-view.png)

Features:

✅ Custom architectures  
✅ Unlimited hidden layers  
✅ Per-layer activation functions  
✅ Live neuron activation display  
✅ Weight visualisation  
✅ Bias inspection  
✅ Interactive neuron and connection editing  
✅ Guided creation wizard  
✅ Advanced layer-by-layer builder (with ready-made presets: XOR, deep net, autoencoder, 3-class classifier, regression, wide net)  
✅ CNN lab  
✅ RNN lab


---

## 🔥 Train Neural Networks Step by Step

Understand what happens inside a model.

![Training controls](screenshots/training.png)

Training modes:

- Single step
- Single epoch
- Multiple epochs
- Automatic training
- Train until target loss

Every step can be inspected:

```
Input
 ↓
Forward Pass
 ↓
Prediction
 ↓
Loss
 ↓
Backpropagation
 ↓
Gradient Update
```

---

# 🧪 Experiment with Modern Deep Learning

Neural Lab includes many core concepts used in real machine learning.

## Activation Functions

Examples:

- Sigmoid
- ReLU
- Tanh
- Leaky ReLU
- ELU
- Swish
- GELU
- SELU
- Softsign
- Linear
- Softmax (output layer)


## Loss Functions

Available:

- Mean Squared Error (MSE)
- MAE
- Log Loss / Binary Cross Entropy
- Huber Loss
- Hinge Loss


## Optimisers

Compare different learning strategies:

- SGD
- Momentum
- RMSProp
- Adam
- AdamW
- Nesterov (NAG)

Fine-grained control is also available: momentum β, Adam β₁/β₂, epsilon, and L2 regularisation.

## Weight Initialisation

- Xavier / Glorot
- He / Kaiming
- Uniform
- Normal
- Small
- Zero (debug)


---

# 📊 Visual Debugging Tools

Neural Lab is not just a trainer.

It is an exploration environment.

![Tools panel](screenshots/tools.png)


Included tools:

### Gradient Analysis

Find:

- Vanishing gradients
- Exploding gradients
- Layer imbalance


### Weight Heatmap

See how your network parameters evolve.

![Heatmap](screenshots/heatmap.png)


### Decision Boundary

Visualise what your classifier has learned.

![Decision boundary](screenshots/boundary.png)


### Optimiser Race

Run several optimisers against each other and watch who learns faster.

![Optimizer race](screenshots/race.png)


### More tools

- Interactive loss curve (click to zoom / clear)
- Learning-rate scheduler
- Layer-by-layer inspection
- Speed benchmark
- Data perturbation testing
- Network diagram export (PNG)


---

# 🎨 Designed for Learning

Neural Lab includes educational features:

- Formula library
- Step-by-step explanations
- Mathematical notation rendering
- Detailed training logs
- Interactive inspection
- **Manual calculation mode**: compute the forward pass, the delta (error gradient) and the weight updates yourself, then check your answers against the simulator
- Keyboard shortcuts for common actions (step, epoch, auto-train, mutate...)
- Training log export to LaTeX or Markdown


![Formula library](screenshots/formulas.png)


Instead of only seeing:

```
Loss = 0.034
```

you can explore:

```
How was the prediction computed?
Why did the gradient change?
Which weights were updated?
```

---

# 🧬 Go Further: CNN, RNN & Evolution

## CNN Lab

Step-by-step convolution, a gallery comparing every filter, a multi-layer pipeline with receptive-field view, kernel training via gradient descent, and a full "complete model" mode to build, train (real backpropagation) and test a multi-layer CNN on an image dataset you create on the spot.

## RNN Lab

A dedicated space to explore recurrent networks.

## Weight Mutation

Randomly mutate a trained network's weights and evaluate the result — a lightweight way to experiment with evolutionary/genetic approaches.

---

# 💾 Save, Export & Code Generation

- Multiple save/load slots for your trained networks
- Import / export a network as JSON
- Export a trained network as ready-to-use JavaScript code
- Export the full training log as LaTeX or Markdown

---

# 🎨 Customisation

- 6 color themes: Dark, Neon, Ocean, Fire, Matrix, Pastel
- Visual effects & sound (Matrix rain, sound toggle, particle flow, celebration effects)
- Adjustable diagram geometry (node spacing, glow intensity, connection opacity...)

---

# ⚡ Quick Start

## Option 1 — Open directly

Download the project:

```bash
git clone https://github.com/sudo-Luca/Neural-Lab.git
```

Open:

```
Neural-Lab/Neural-Lab-v12.html
```

That's it.

No dependencies.
No build step.

---

## Option 2 — Try online

Any modern browser works:

- Chrome
- Firefox
- Edge
- Safari
- Opera
- Opera GX

Click [here](https://sudo-luca.github.io/Neural-Lab/Neural-Lab-v12.html) and enjoy your experience

---

# 🧩 Example Experiments

## XOR Problem

A classic neural network challenge.

```
0 XOR 0 → 0
0 XOR 1 → 1
1 XOR 0 → 1
1 XOR 1 → 0
```

Try:

```
Input: 2
Hidden: 4
Output: 1
Activation: ReLU
Optimizer: Adam
```

---

## Spiral Classification

A harder nonlinear problem.


---

## Regression

Approximate mathematical functions:

```
y = sin(x)
```

Watch the network learn a curve.

---

## CNN problems

Learn about CNN with the CNN lab in tools tab

---

# 📱 Works Everywhere

Desktop:

![Desktop](screenshots/desktop.png)


Mobile:

![Mobile](screenshots/mobile.png)


The interface automatically adapts:

- Responsive layout
- Touch-friendly controls
- Mobile navigation
- Full-screen panels


---

# 🏗️ Architecture

Neural Lab is intentionally simple:

```
Single HTML file

├── HTML
│   └── Interface
│
├── CSS
│   └── Dark UI system
│
└── JavaScript
    ├── Neural network engine
    ├── Training algorithms
    ├── Canvas renderer
    └── Interactive tools
```

No frameworks.

No hidden magic.

Everything is visible.

---

# 📚 Technical Documentation

For a complete technical description of:

- internal architecture
- neural computation
- backpropagation implementation
- optimiser formulas
- rendering engine
- dataset system
- v12 extensions

See:

➡️ [`Technical-Description.md`](Technical-Description.md)

---

# .🎯 Project Goals

Neural Lab was created with one idea:

> **Understanding neural networks should not require treating them as a black box.**

The goal is to make deep learning concepts:

- visual
- interactive
- experimental
- approachable


---

# 🛠️ Technologies

Built with:

| Technology | Usage |
|-|-|
| HTML5 Canvas | Neural visualisation |
| Vanilla JavaScript | Neural engine |
| CSS Variables | UI system |
| KaTeX | Mathematical rendering |


---

# 🌟 Roadmap

Possible future improvements:

- [x] More architectures (CNN, RNN)
- [x] More datasets
- [ ] More architectures (Transformers, GANs...)
- [ ] More built-in datasets
- [ ] WebGPU acceleration
- [ ] Collaborative experiments

> ✅ CNN/RNN labs and network export/import are already available — see [Go Further](#-go-further-cnn-rnn--evolution) and [Save, Export & Code Generation](#-save-export--code-generation) above.


---

# 🤝 Contributing

Ideas, improvements and experiments are welcome.

Feel free to open:

- Issues
- Feature requests
- Pull requests


---

# 📜 License

See repository license information.

---

<div align="center">

**Made for learning. Built for curiosity. 🧠**

</div>
