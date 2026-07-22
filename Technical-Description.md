# Neural Lab v12

> **A self-contained, pedagogical deep learning simulator built entirely in a single HTML file.**

Neural Lab v12 is an interactive browser-based tool for visualising, training, and understanding multilayer perceptrons (MLPs). It requires no server, no framework, and no installation — just open the HTML file in any modern browser. Every forward pass, backpropagation step, gradient computation, and weight update is performed in vanilla JavaScript and rendered in real time on an HTML5 Canvas, making it ideal for learning, teaching, or experimenting with neural network fundamentals.

---

## Table of Contents

1. [Overview](#overview)
2. [Interface Layout](#interface-layout)
3. [Architecture & Global State](#architecture--global-state)
4. [Activation Functions](#activation-functions)
5. [Loss Functions](#loss-functions)
6. [Weight Initialisation](#weight-initialisation)
7. [Core Neural Network Functions](#core-neural-network-functions)
8. [Optimiser Engine](#optimiser-engine)
9. [Training Control Functions](#training-control-functions)
10. [Visualisation & Canvas Rendering](#visualisation--canvas-rendering)
11. [Dataset Management](#dataset-management)
12. [Right-Panel Tabs & Inspection Tools](#right-panel-tabs--inspection-tools)
13. [v12 Extension Features](#v12-extension-features)
14. [UI Utility Functions](#ui-utility-functions)
15. [Keyboard Shortcuts](#keyboard-shortcuts)
16. [Mobile Support](#mobile-support)
17. [Local Dev](#local-dev)

---

## Overview

Neural Lab v12 is structured as a **single HTML document** (~6000 lines) combining:

- **HTML** — three resizable panels (left config, centre canvas, right details) plus several modal dialogs.
- **CSS** — a custom dark-mode design system using CSS variables for a consistent colour palette.
- **Vanilla JavaScript** — all neural network logic, rendering, and interactivity, split across two `<script>` blocks: the main core (lines ~1764–5123) and the v12 extension block (lines ~5239–6016).

There are no external JavaScript dependencies except the **KaTeX** library (loaded via CDN) for rendering mathematical notation in the formula library.

### Key capabilities

| Category | Details |
|---|---|
| Architectures | Fully-connected MLP, arbitrary depth and width, per-layer activations |
| Activations | Sigmoid, ReLU, Tanh, Leaky ReLU, ELU, Swish, GELU, SELU, Softsign, Linear, Softmax |
| Loss functions | MSE, MAE, Log Loss (BCE), Huber, Hinge |
| Optimisers | SGD, Momentum, RMSProp, Adam, AdamW, Nesterov AG |
| Weight init | Xavier/Glorot, He/Kaiming, Uniform, Normal, Small |
| LR schedulers | Cosine annealing, Step decay, Exponential decay, Warmup-Cosine |
| Training modes | Single step, single epoch, N epochs, auto-run, train-to-target |
| Inspection | Step-by-step log, gradient analysis, layer inspector, weight heatmap, zoomed loss chart |
| Export | PNG canvas, clipboard copy, LaTeX log, Markdown log, JSON model |
| v12 extras | Decision boundary visualiser, optimiser race, network mutation, confetti, Matrix rain, data flow animation, sound feedback, save/load slots |

---

## Interface Layout

The UI is divided into three resizable panels separated by drag handles (`.rsz-col`) and a vertically resizable log panel (`.rsz-row`).

```
┌──────────────┬────────────────────────────┬───────────────────┐
│  LEFT PANEL  │     CENTER — Canvas        │   RIGHT PANEL     │
│  (260 px)    │   Neural network diagram   │   (370 px)        │
│              │                            │                   │
│  Architecture│                            │  Test / Log       │
│  Training    │                            │  Tabs:            │
│  Display     │                            │  - Test           │
│              ├────────────────────────────┤  - Formula lib    │
│              │   LOG PANEL (190 px)       │  - Tools          │
│              │   Structured backprop log  │  - Settings       │
└──────────────┴────────────────────────────┴───────────────────┘
```

On viewports ≤ 900 px the layout switches to a **mobile mode**: a fixed top bar, a bottom navigation bar, and each panel occupies the full screen as a switchable tab.

---

## Architecture & Global State

### Global variables

| Variable | Type | Description |
|---|---|---|
| `NET` | `{L, W, B}` or `null` | The active network. `L` = layer sizes array, `W` = weight tensors (3-D), `B` = bias vectors (2-D) |
| `LOSS_H` | `number[]` | Loss history per epoch |
| `EPOCH` | `number` | Current epoch counter |
| `autoRunning` | `boolean` | Whether the auto-training loop is active |
| `LAST_ACT` | `number[][]` | Activations from the most recent forward pass (per layer) |
| `LAST_Z` | `number[][]` | Pre-activations (z values) from the most recent forward pass |
| `LAST_DELTAS` | `number[][]` | Delta values from the most recent backward pass |
| `VEL` | `{w, b}` | Velocity accumulators for Momentum / NAG |
| `MA` | `{w, b}` | First-moment estimates for Adam/AdamW |
| `MV` | `{w, b}` | Second-moment estimates for Adam/AdamW/RMSProp |
| `ADAM_T` | `number` | Adam step counter (for bias correction) |
| `DISP` | `{W,B,A,I,G,Grid}` | Boolean display toggles for canvas overlays |
| `DS` | `{inputs, outputs, inN, outN}` | Active dataset |
| `POS` | `{x,y,l,j,av}[]` | Computed neuron positions for canvas hit-testing |
| `LINES` | `object[]` | Computed connection lines for canvas hit-testing |
| `LAYER_ACTS` | `string[]` | Per-layer activation function overrides (used by Advanced Builder) |
| `RAF_ID` | `number` | `requestAnimationFrame` handle for the auto-training loop |
| `SELECTED` | `object` | Currently selected neuron (for tooltip) |
| `ε` | `1e-15` | Numerical stability constant |

### Helper constants / micro-functions

| Name | Signature | Description |
|---|---|---|
| `h2r` | `(hex) → {r,g,b}` | Converts a hex colour string to an RGB object |
| `clamp` | `(v, a, b) → number` | Clamps `v` between `a` and `b` |
| `gauss` | `() → number` | Box-Muller Gaussian sample (mean=0, σ=1) |
| `getLR` | `() → number` | Reads the learning rate from the `lrN` input |
| `getAct` | `() → string` | Reads the hidden-layer activation key from `iAct` |
| `getActOut` | `() → string` | Reads the output-layer activation key (`iActOut`), falling back to `getAct()` if set to `"same"` |
| `getLoss` | `() → string` | Reads the loss function key from `iLoss` |
| `getOpt` | `() → string` | Reads the optimiser key from `iOpt` |
| `getV` | `(id) → number` | Reads a numeric value from an input element by ID |
| `fmt` | `(v) → string` | Formats a number to 5 decimal places, or `"NaN"` |
| `syncLR` | `(v) → void` | Synchronises the LR slider, number input, and stats display |

---

## Activation Functions

### `act(z, fn) → number`

Computes the activation output for a single pre-activation value `z` using the function identified by `fn`.

| `fn` | Formula | Notes |
|---|---|---|
| `sigmoid` | `1/(1+e^{-z})` | Clamped to `[-500, 500]` for numerical stability |
| `relu` | `max(0, z)` | Standard rectified linear unit |
| `tanh` | `tanh(z)` | Hyperbolic tangent via `Math.tanh` |
| `leakyrelu` | `z > 0 ? z : 0.01·z` | Fixed leak slope 0.01 |
| `elu` | `z ≥ 0 ? z : e^z − 1` | Exponent clamped to `[-50, 0]` |
| `swish` | `z · sigmoid(z)` | Self-gated activation (Google Brain) |
| `gelu` | `0.5·z·(1 + tanh(√(2/π)·(z + 0.044715·z³)))` | Tanh approximation, used in BERT/GPT |
| `selu` | `scale · (z ≥ 0 ? z : α·(e^z − 1))` | α=1.6732632, scale=1.0507009 |
| `softsign` | `z / (1 + |z|)` | Range (−1, 1), less saturating than tanh |
| `linear` | `z` | Identity, used for regression outputs |
| `softmax` | pass-through (handled externally by `softmax()`) | See below |

### `actD(a, z, fn) → number`

Computes the derivative of the activation for use in backpropagation. Takes both the output activation `a` and the pre-activation `z` because some derivatives are more efficiently expressed in terms of `a`.

| `fn` | Derivative |
|---|---|
| `sigmoid` | `a·(1−a)` |
| `relu` | `z > 0 ? 1 : 0` |
| `tanh` | `1 − a²` |
| `leakyrelu` | `z > 0 ? 1 : 0.01` |
| `elu` | `z ≥ 0 ? 1 : a + 1` |
| `swish` | `s + z·s·(1−s)` where `s = sigmoid(z)` |
| `gelu` | Full analytical derivative via tanh approximation |
| `selu` | `z ≥ 0 ? scale : scale·α·e^z` |
| `softsign` | `1 / (1 + |z|)²` |
| `linear` | `1` |

### `softmax(arr) → number[]`

Applies the softmax function to a vector `arr`. Uses the **max-subtraction trick** (`eᵛ⁻ᵐᵃˣ`) for numerical stability. Returns a normalised probability distribution summing to 1.

### `getActForLayer(l) → string`

Returns the activation function name for layer index `l` (0-indexed). If `LAYER_ACTS` has been populated by the Advanced Builder, it returns `LAYER_ACTS[l]`; otherwise it returns the output activation for the last layer and the global hidden activation for all others.

---

## Loss Functions

### `computeLoss(pred, y) → number`

Computes the scalar loss between prediction array `pred` and target array `y`.

| Key | Formula | Notes |
|---|---|---|
| `mse` | `(1/n)·Σ(p−y)²` | Mean Squared Error |
| `mae` | `(1/n)·Σ|p−y|` | Mean Absolute Error |
| `logloss` | `−(1/n)·Σ[y·log(p̂)+(1−y)·log(1−p̂)]` | Binary Cross-Entropy; prediction clamped to `[ε, 1−ε]` |
| `huber` | `½e² if |e|≤δ, else δ(|e|−½δ)` | Hybrid MSE/MAE; δ read from input `cHD` |
| `hinge` | `(1/n)·Σ max(0, 1−t·p)` | SVM-style; label `y=0` maps to `t=−1` |

### `computeLossGrad(pred, y) → number[]`

Returns the per-output gradient of the loss `∂L/∂p` for each output neuron. Uses the same formula keys as `computeLoss`. This gradient seeds the backpropagation chain rule at the output layer.

---

## Weight Initialisation

### `initW(nIn, nOut, method) → number`

Generates a single initial weight value given the fan-in `nIn` and fan-out `nOut`.

| `method` | Distribution | Best for |
|---|---|---|
| `xavier` | `U(−√(6/(nIn+nOut)), +√(6/(nIn+nOut)))` | Sigmoid, Tanh |
| `he` | `N(0, √(2/nIn))` — Box-Muller | ReLU, Leaky ReLU |
| `uniform` | `U(−1, 1)` | General purpose |
| `normal` | `N(0, 0.1)` | Small random |
| `small` | `U(−0.1, 0.1)` | Avoiding saturation |
| `zero` | `0` | Debugging |
| *(default)* | `U(−0.8, 0.8)` | Fallback |

The `method` defaults to the value of the `cInit` selector in the UI.

---

## Core Neural Network Functions

### `buildNet(customLayers?, customActs?, customInitMethod?) → void`

Constructs a new neural network from scratch, replacing any existing one.

**Parameters:**
- `customLayers` — array of layer sizes (e.g. `[2,4,3,1]`); defaults to the value of `iLayers` in the UI.
- `customActs` — array of per-layer activation names (populates `LAYER_ACTS`); used by the Advanced Builder.
- `customInitMethod` — weight initialisation method string; falls back to the `cInit` selector.

**Behaviour:**
1. Parses and validates the layer size array (minimum two layers).
2. Synchronises `DS.inN`/`DS.outN` with the new architecture if they differ.
3. Initialises `NET.W` (weight matrices) and `NET.B` (bias vectors) using `initW`.
4. Resets all optimiser state (`VEL`, `MA`, `MV`, `ADAM_T`).
5. Resets epoch counter, loss history, and cached pass data.
6. Triggers a full canvas redraw and logs a creation summary.

### `resetW() → void`

Re-randomises all weights and biases of the existing network using the current initialisation method. Resets epoch, loss history, and optimiser state but preserves the network topology.

### `setBias(l, j, val) → void`

Manually sets the bias of neuron `j` in layer `l` to a parsed float value. Redraws and logs the change. Used from click-to-edit interactions in the detail panel.

### `forward(inputs) → {activations, zs}`

Performs a **forward pass** through the entire network.

**Returns:**
- `activations` — 2-D array of shape `[layers+1][neurons]`; `activations[0]` is the input layer.
- `zs` — 2-D array of shape `[layers][neurons]`; the pre-activation value at each non-input neuron.

**Algorithm:**
1. Seeds the activation array with the input vector.
2. For each layer `l`, computes the dot product `z[j] = Σ(W[l][j][k] · a[l][k]) + B[l][j]`.
3. Applies the layer's activation function (via `getActForLayer`).
4. For the last layer, applies **softmax** if the output activation is `"softmax"`.
5. Stores results in `LAST_ACT` and `LAST_Z` for inspection and logging.

### `backward(activations, zs, y) → deltas`

Performs **backpropagation** and returns the delta matrix without updating weights.

**Returns:**
- `deltas` — 2-D array of shape `[layers][neurons]`, one delta per neuron.

**Algorithm:**
1. Computes the output layer deltas: `δ[j] = (∂L/∂a) · f'(z)` using `computeLossGrad` and `actD`.
2. For softmax outputs, approximates `f'(z) = a·(1−a)` (diagonal of Jacobian).
3. Propagates error backwards through hidden layers: `δ[l][k] = (Σⱼ δ[l+1][j] · W[l+1][j][k]) · f'(z[l][k])`.
4. Stores deltas in `LAST_DELTAS` for logging.

### `trainSample(inputs, y, verbose?) → number`

Runs a complete **forward + backward + weight update** cycle for a single training example.

**Returns:** the loss for this sample.

**Optimiser update rules (applied per weight `w`):**

| Optimiser | Update rule |
|---|---|
| `sgd` | `w ← w − lr · g` |
| `momentum` | `v ← 0.9·v + lr·g`, then `w ← w − v` |
| `nag` | Nesterov: look-ahead step before gradient computation |
| `rmsprop` | `E ← 0.9·E + 0.1·g²`, then `w ← w − lr·g / √(E+ε)` |
| `adam` | `m ← β₁·m + (1−β₁)·g`, `v ← β₂·v + (1−β₂)·g²`, bias-corrected update |
| `adamw` | Adam + L2 weight decay: `w ← w·(1−lr·λ)` before update |

If `verbose` mode is active (the "Complet" log tab is shown), this function also generates the detailed step-by-step HTML log covering all four phases: Forward Pass, Loss, Backpropagation, and Weight Update.

---

## Optimiser Engine

The optimiser state is stored globally:

- `VEL.w[l][j][k]` / `VEL.b[l][j]` — velocity (Momentum/NAG)
- `MA.w[l][j][k]` / `MA.b[l][j]` — first moment (Adam/AdamW)
- `MV.w[l][j][k]` / `MV.b[l][j]` — second moment (Adam/RMSProp/AdamW)
- `ADAM_T` — global step counter for Adam bias correction

Hyperparameters are hard-coded as conventional defaults:
- Momentum β = 0.9
- RMSProp β = 0.9
- Adam β₁ = 0.9, β₂ = 0.999, ε = 1e-8
- AdamW λ (weight decay) = 0.01

---

## Training Control Functions

### `doStep() → void`

Runs one training sample (the next in the dataset in round-robin order). Updates stats, redraws the canvas, and draws the loss chart. In v12, also hooks into `updateNetStatsBar()`, `checkCelebrate()`, and `applyScheduler()`.

### `doEpoch() → void`

Iterates over **every sample in the dataset once**, accumulating the total loss, then increments `EPOCH`, pushes the average loss to `LOSS_H`, updates stats, redraws, and logs the epoch summary with trend arrows.

### `doN(n) → void`

Runs `n` training steps in a synchronous loop (using `requestAnimationFrame` batching for large `n` to avoid blocking the UI). Progress is shown in the progress bar (`#pb`).

### `toggleAuto() → void`

Toggles the auto-training loop. When active, calls `doStep()` repeatedly using `requestAnimationFrame`, with a configurable delay from the `autoSpeed` slider (0–500 ms per batch). The `btnAuto` button visually changes to `pulse` animation.

### `trainToTarget() → void` (v12 §16)

Trains the network epoch by epoch until either the average loss drops below the target value from `lossTarget` or a maximum of 10 000 epochs is reached. Updates the `target-status` element with live progress. Plays a success tone on convergence if sound is enabled.

### `stopTargetTrain() → void`

Clears the `targetTrainTimer` timeout used by `trainToTarget()`.

### `acc() → number`

Computes the current classification accuracy (in %) over the full dataset. A prediction is considered correct if `Math.round(p) === y` for all outputs. Returns 0 if the dataset is empty.

### `updateStats(loss) → void`

Updates the four stat cards in the left panel (`sEpoch`, `sLoss`, `sAcc`, `sLR`) and redraws the mini loss chart.

### `drawLossChart() → void`

Renders the small loss curve in the left panel (`#lossChart`) using an HTML5 Canvas. Draws the raw loss curve in `--red` and, when more than 10 data points are available, a 10-point moving average in `--ylw` (dashed).

---

## Visualisation & Canvas Rendering

### `redraw() → void`

The main canvas rendering function. Clears and repaints the entire neural network diagram. Called after every training step, weight change, or resize event.

**Steps:**
1. Clears the canvas and fills the background (from `cBG` colour picker).
2. Optionally draws a **grid** (cross or dot pattern) from `cGridMode`.
3. Computes neuron positions (`POS`) based on layer widths and spacing factors `cSV`/`cSH`.
4. Draws **connections** (weight lines) with:
   - Colour from `cWP`/`cWN` pickers for positive/negative weights.
   - Opacity and thickness proportional to `|w|`.
   - Optional Bézier curves (`cCurve`) or straight lines.
   - Optional arrowheads (`cArrow`) at the midpoints.
   - Weight value labels (if `DISP.W` and network is narrow enough).
   - **Frozen weight indicators** (❄ icon) if `NET.frozenWeights` is set.
5. Draws **neurons** with:
   - Radial glow halos proportional to activation (if `DISP.G`).
   - Fill colour based on activation magnitude (if `cColorNode` is checked).
   - Input / hidden / output layer colour scheme from `cNI`/`cNH`/`cNO` pickers.
   - Neuron index labels (if `DISP.I`).
   - Activation values (if `DISP.A`).
   - Bias values (if `DISP.B`).

### `resizeCvs() → void`

Sets the canvas `width` and `height` to the pixel dimensions of its wrapper div `canvasWrap`. Called on window resize and panel resize.

### `getR() → number`

Returns the neuron radius in pixels from the `cR` input, defaulting to 18.

### `tog(key) → void`

Toggles a display flag in the `DISP` object and updates the corresponding button's `on` class. Keys: `'W'`, `'B'`, `'A'`, `'I'`, `'G'`, `'Grid'`.

---

## Dataset Management

The dataset object `DS` holds `{inputs, outputs, inN, outN}`.

### `loadPreset(name?) → void`

Loads a built-in dataset preset. Available presets include:
- `xor` — XOR gate (4 samples, 2 inputs, 1 output)
- `and` / `or` — basic logic gates
- `circle` — points inside vs outside a circle
- `spiral` — two-class spiral (harder non-linear problem)
- `sin` — sine function regression
- `custom` — clears the dataset for manual entry

Resets epoch and loss history. Rebuilds the dataset editor UI.

### `buildDSEditorFromCurrent() → void`

Regenerates the dataset editor rows in `#dsEditor` from the current `DS.inputs` and `DS.outputs` arrays.

### `readDSFromEditor() → void`

Reads all editable cells from the dataset table and repopulates `DS.inputs` and `DS.outputs`.

### `addDSRow() → void`

Adds a blank sample row to the dataset editor.

### `removeDSRow(i) → void`

Removes the dataset row at index `i` from both the editor and `DS`.

---

## Right-Panel Tabs & Inspection Tools

The right panel uses a tab system (`#tabs`) with four tabs: **Test**, **Formules**, **Outils**, **Settings**.

### Test tab

#### `runTest() → void`
Runs a single inference on the values entered in the test input fields (`tIn0`, `tIn1`, …), then calls `showTestSteps()` to build the step-by-step explanation.

#### `showTestSteps(inputs, y_target) → void`
Generates the full pedagogical breakdown HTML for a single inference:
- **Phase 1 – Forward Pass**: for each neuron, shows the weighted sum computation, bias addition, activation formula, and f'(z) for backprop.
- **Phase 2 – Loss**: computes and displays the loss term per output, with the formula and numerical substitution.
- **Phase 3 – Backpropagation**: shows the chain rule, output-layer deltas, and hidden-layer delta propagation.
- **Phase 4 – Weight Update**: shows the gradient `∂L/∂w = δ·a` and new weight values under the selected optimiser.

All four phases are rendered as collapsible sections (`togglePhase(id)`).

#### `runAllTest() → void`
Tests every sample in the dataset, displays predictions vs. targets, and reports overall accuracy.

#### `togglePhase(id) → void`
Shows or hides a collapsible phase section in the test steps view.

### Formules tab (Formula Library)

#### `buildFLib() → void`
Initialises the formula library UI: extracts unique tags from `FLIB` (the formula data array), renders tag filter pills, and calls `renderFTree`.

#### `setFTag(tag, el) → void`
Filters the library to show only entries with the selected tag. Updates active state on the tag pills.

#### `filterF(query) → void`
Text-search filter applied to formula names, equations, descriptions, and tags. Triggers a re-render.

#### `getFilt() → object[]`
Returns the filtered subset of `FLIB` matching the current tag and text search.

#### `renderFTree(list) → void`
Renders the formula tree as nested collapsible categories and subcategories. Each entry is rendered as a card with its equation, description, pros/cons, and a small activation curve graph if applicable.

#### `showFormula(id) → void`
Expands the detail view for a specific formula from `FLIB`. Renders mathematical notation via `renderMathEq`, shows a symbol legend from `getFormulaSymbols`, and draws an activation curve in a mini canvas via `drawFGraph`.

#### `renderMathEq(eq) → string`
Converts a plain-text mathematical expression into styled HTML using `<span>` elements. Handles:
- Fractions `A/B` using `.math-frac` layout.
- Large sigma `Σ` with `.math-sigma` styling.
- Colour coding for `lr`, Greek letters, `ŷ`, `←`/`→` symbols.

#### `drawFGraph(domId, fnId) → void`
Draws the activation curve for function `fnId` into a canvas element at `fgcvs_<domId>`. Plots values over z ∈ [−4, +4] using the `act()` function.

#### `getFormulaSymbols(f) → object[]`
Returns up to 6 relevant symbol definitions for a formula entry based on its tags. Used to populate the legend block under each formula's detail view.

### Outils (Tools) tab

#### `analyzeGradients() → void` (v12 §12)
Runs a full forward + backward pass on every dataset sample, averages the absolute delta and weight-gradient values per layer, and renders a visual report with:
- Min/avg/max of `|δ|` (error signal) per layer.
- Min/avg/max of `|∂L/∂w|` (actual weight gradient) per layer.
- Log-scale bar charts for each metric.
- Automatic detection and labelling of **vanishing gradients** (max < 1e-4) or **exploding gradients** (max > 50).
- A global summary ratio `max_δ / min_δ` to detect inter-layer imbalance.

#### `inspectLayerStats() → void` (v12 §14)
Reads the selected layer index from `inspectLayer` and displays a statistics card with:
- Number of neurons and total weights.
- Min/max/mean/std of all weights.
- Min/max of biases.
- Count of dead neurons (activation < 1e-6 on last pass).

#### `populateLayerInspect() → void`
Repopulates the `<select id="inspectLayer">` dropdown with one option per weight layer, after building a new network.

#### `drawZoomedLoss() → void` (v12 §13)
Renders a zoomed loss chart in `#zoomedLossChart` showing the last `n` epochs (controlled by `lossZoom` slider). Draws a filled gradient area, the raw loss curve in red, and a 10-point moving average in yellow dashed.

#### `runBenchmark() → void` (v12 §15)
Measures training throughput by running 500 consecutive forward + backward passes and reporting samples/second, total time, and milliseconds per sample.

### Settings tab

The Settings tab contains visual customisation controls (neuron radius, font size, line width, colours, grid mode, curve/arrow toggles) and canvas appearance controls. Changes take effect immediately via `redraw()`.

---

## v12 Extension Features

All features below are defined in the second `<script>` block appended at the end of the document.

### 1. Net Stats Bar — `updateNetStatsBar() → void`
Refreshes the `#net-stats-bar` strip at the top of the canvas area, displaying architecture, total parameter count, epoch, loss, accuracy, LR, and a keyboard shortcut hint.

### 2. Decision Boundary — `renderBoundary() → void`
Renders a 2-D classification boundary in a modal canvas (`#boundary-cvs`). Sweeps a grid of `res×res` pixels over the input space `[−0.2, 1.2]²`, runs a forward pass for each point, and colours pixels cyan (class 0) or purple (class 1) proportionally to the output confidence. Overlays data points as coloured dots.

- `showBoundary() → void` — opens the boundary modal and calls `renderBoundary`.
- `startBoundaryLive() / stopBoundaryLive() → void` — starts/stops a 350 ms refresh interval for live updates during training.

### 3. Network Mutation — `mutateNet(evalAfter?) → void`
Randomly perturbs 40% of weights and 30% of biases by ±`str` (from `mutStr` input). If `evalAfter` is true, evaluates the new loss; if the loss has worsened by more than 50%, the mutation is reverted using a backup. Plays a visual animation on the canvas.

### 4. Save / Load — localStorage-backed model persistence
Five named slots stored in `localStorage` under key `neurallab_v12_saves`.

- `saveToSlot(slot) → void` — serialises `NET.W`, `NET.B`, `NET.L`, `EPOCH`, last 50 loss values, and `LAYER_ACTS` to JSON and stores them.
- `loadSlot(slot) → void` — restores a network from a slot and redraws.
- `deleteSlot(slot) → void` — clears a specific slot.
- `exportNetJSON() → void` — downloads the full network as a `.json` file.
- `importNetJSON(input) → void` — reads a JSON file from a file input and restores the network.

### 5. LR Scheduler — `applyScheduler() → void`
Applies a learning rate schedule based on the current epoch. Called after every training step.

| Schedule | Formula |
|---|---|
| `cosine` | `lr_min + 0.5·(lr_max−lr_min)·(1+cos(π·(t mod T)/T))` |
| `step` | `lr_max · 0.5^⌊t/T⌋` |
| `exp` | `lr_max · 0.99^t` |
| `warmup` | Linear warmup for first 10% of period, then cosine annealing |

`updateSchedUI()` and `updateSchedInfo()` keep the UI controls and info label in sync.

### 6. Optimiser Race — `startRace() / stopRace() → void`
Runs a head-to-head training competition between four optimisers (SGD, Momentum, Adam, RMSProp) on the current dataset, each starting from the same weight initialisation. Displays live progress bars, loss values, accuracy, a shared loss chart, and announces the winner.

- `initRaceNets()` — creates four independent network copies with identical topology and weights but separate optimiser states.
- `raceStep()` — advances all four networks by one epoch and updates the UI.

### 7. Confetti — `triggerConfetti() → void`
Launches a particle animation on `#confetti-canvas` when the network reaches 100% accuracy (`checkCelebrate()`). Particles are coloured rectangles with random velocities, fading out via `animateConfetti()`.

### 8. Matrix Rain — `toggleMatrix() → void`
Overlays a Matrix-style falling characters animation on `#matrix-canvas` using `animateMatrix()`. Characters include digits, katakana, and mathematical symbols.

### 9. Sound Feedback — `toggleSound() / playTone(freq, dur, type?, vol?) → void`
Uses the Web Audio API (`AudioContext`) to play brief synthesised tones. A pitch proportional to training progress is played on each epoch step. A two-note success chord plays when the target loss is reached.

### 10. Data Flow Animation — `toggleFlow() → void`
Animates small coloured dots travelling along weight connections on the canvas via `animateFlow()`. Dots are spawned on random connections and travel from source to destination neuron. Positive-weight connections produce cyan dots; negative-weight connections produce red dots.

### 11. Weight Heatmap — `renderHeatmap() → void`
Renders a grid of small coloured squares in `#weightHeatmap`, one cell per weight. Positive weights are shown in shades of green-cyan; negative weights in shades of red-purple. The intensity is proportional to the weight's magnitude relative to the layer maximum. Cells show the exact weight value on hover.

### 12. Data Perturbation — `addNoiseToDataset() / resetDatasetNoise() → void`
Adds uniform noise `U(−σ, +σ)` (σ from `noiseLvl` input) to all input features. The original dataset is backed up to `DS_BACKUP`. `resetDatasetNoise()` restores the clean backup.

### 13. Export Canvas — `exportCanvasPNG() / copyCanvasClipboard() → void`
- `exportCanvasPNG()` — triggers a PNG download of the current canvas state, named `neural_net_e<EPOCH>.png`.
- `copyCanvasClipboard()` — copies the canvas PNG to the clipboard via the Clipboard API.

### 14. Log Export — `exportLogLatex() / exportLogMarkdown() → void`
- `exportLogLatex()` — converts the log panel's text content to a LaTeX document and triggers a `.tex` download.
- `exportLogMarkdown()` — converts the log to a Markdown document and triggers a `.md` download.

### 15. Wizard — `showWizard() / applyWizard() / closeModal() → void`
A quick-start modal that auto-configures the network (layer sizes, activation, loss, optimiser) based on the selected problem type (binary classification, multiclass, regression), number of inputs, and number of outputs.

### 16. Advanced Builder — `showAdvBuilder() / applyAdvBuilder() / closeAdvBuilder() → void`
A per-layer network configuration modal. Each layer row has independent controls for neuron count, activation function, dropout rate, and L2 regularisation. Includes quick presets (XOR, Deep, Autoencoder, 3-class, Regression, Wide). A live preview canvas shows the architecture before building.

- `abldAddLayer()` — adds a new hidden layer row.
- `abldRemoveLayer(i)` — removes a layer row by index.
- `abldPreset(name)` — applies a named preset configuration.
- `drawAbldPreview()` — redraws the architecture preview canvas.

---

## UI Utility Functions

### Logging

#### `log(html) → void`
Appends a raw HTML string to the `#log` element and auto-scrolls to the bottom.

#### `clearLog() → void`
Empties the log panel.

#### `logBlock(cls, icon, title, body, startOpen?) → void`
Appends a **collapsible structured block** to the log with a coloured header bar (forward=cyan, loss=red, backprop=orange, update=green). Used during verbose training.

#### `toggleLogBlock(id) → void`
Toggles visibility of a log block body identified by `id`.

#### `logEpoch(avg) → void`
Appends a formatted epoch summary line with loss (colour-coded), trend arrow (% change from previous epoch), accuracy, and optimiser info.

### Canvas interaction

#### Canvas click handler
Detects clicks on neurons (within radius `r`) and connections (within 6 px of the line). On neuron click, opens a detail tooltip showing index, layer type, activation, bias, and activation value. On connection click, shows weight details with an inline edit field. `Ctrl+Click` on a neuron forces its activation to a user-specified value.

### Resize system

#### `initColResize(handleId, targetId, side) → void`
Attaches mouse drag listeners to a column resize handle, allowing the left or right panel to be resized within their CSS `min-width`/`max-width` bounds. Double-click resets to the default width (260 px for left, 370 px for right).

#### `initRowResize(handleId, targetId) → void`
Attaches mouse drag listeners to a row resize handle for the log panel. Double-click resets to 190 px height.

### Tooltip

#### `showTip(html, x, y) → void`
Positions and displays the floating tooltip `#tip` near canvas coordinates `(x, y)`.

#### `hideTip() → void`
Hides the tooltip.

### Progress bar

The `#pb` element is updated during `doN()` to show training progress as a horizontal fill.

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `Space` | `doStep()` — single training step |
| `E` | `doEpoch()` — one full epoch |
| `A` | `toggleAuto()` — toggle auto-training |
| `M` | `mutateNet(false)` — mutate weights |
| `B` | `showBoundary()` — decision boundary |
| `R` | `showRace()` — optimiser race |
| `T` | `showTab('tools')` — switch to Tools tab |
| `G` | `analyzeGradients()` — gradient analysis |
| `H` | `renderHeatmap()` — weight heatmap |
| `Escape` | Close all modals, stop live animations |

Shortcuts are disabled when focus is inside `<input>`, `<textarea>`, or `<select>` elements.

---

## Mobile Support

On viewports ≤ 900 px the layout switches entirely via CSS media queries:

- The three panels stack and are shown one at a time, each taking the full viewport.
- A **top bar** (`#mobile-topbar`) replaces the left panel header, showing the app title plus a live epoch/loss pill.
- A **quick-train bar** (`#mobile-trainbar`) below the top bar provides one-tap access to Step, Epoch, Auto, and ×100 buttons.
- A **bottom navigation bar** (`#mobile-nav`) switches between Config, Network, and Details panels.
- Resize handles are hidden on mobile.
- Modals slide up from the bottom as sheets.
- The log panel has a fixed 160 px height.

#### `mobNav(panel) → void`
Activates the named panel (`'left'`, `'center'`, or `'right'`) by toggling the `mob-active` class. On switching to the canvas panel, a 60 ms timeout ensures `resizeCvs()` fires after the layout is applied.

#### `syncMobStats() → void`
Called every 400 ms via `setInterval`. Copies the epoch and loss text content from the left panel stats display to the top bar pill elements.

---

### Local Dev

Tip to have **hotreload** (*nodejs* must be installed)

```bash
npx live-serveer .
```

---

## Colour Palette Reference

All colours are defined as CSS custom properties on `:root`:

| Variable | Value | Usage |
|---|---|---|
| `--bg` | `#05060c` | Main background |
| `--p1` | `#0a0c18` | Panel background |
| `--p2` | `#0d1020` | Section/header background |
| `--brd` | `#192040` | Border colour |
| `--a1` | `#00e5ff` | Primary accent (cyan) |
| `--a2` | `#bf5fff` | Secondary accent (purple) |
| `--grn` | `#00ff9d` | Success / positive |
| `--ylw` | `#ffd600` | Warning / highlight |
| `--red` | `#ff3d5a` | Error / loss |
| `--org` | `#ff8c2a` | Backprop / alert |
| `--text` | `#b8cce8` | Primary text |
| `--dim` | `#445577` | Secondary / muted text |

---

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| KaTeX | 0.16.9 | Mathematical formula rendering in the formula library |
| JetBrains Mono | Google Fonts | Monospace UI font |
| Syne | Google Fonts | Display / heading font |

All other functionality is implemented in vanilla JavaScript with no runtime dependencies.
