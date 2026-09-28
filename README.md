# Mapping Meaning: AI vector explorer

A Python program that opens a 3D teaching simulation in your browser. It works
offline, uses only Python's standard library, and needs no account or API key.

## Run it

Install Python 3.9 or newer if you do not already have it. Save
`ai_mind_model.py` in a folder, open a terminal in that folder, and run:

```sh
python3 ai_mind_model.py
```

On Windows, use:

```sh
py ai_mind_model.py
```

The browser opens automatically. If it does not, copy the address printed in
the terminal into your browser. Keep the terminal open while using the program.
Press **Ctrl+C** in the terminal to stop it.

**There is nothing to install with pip.** The `.py` file contains the entire app.
Python generates nodes, calculates vectors, and samples choices. The browser
draws the 3D view and handles the camera.

## Add X random nodes

Enter a number in **Number of random spheres to add**, then click **Add nodes**.
Every click adds exactly that many new nodes to the existing scene. The limit is
10,000 total nodes, including the lesson examples.

You can also choose the starting number when launching:

```sh
python3 ai_mind_model.py --nodes 100
```

This starts with the 25 example words plus 100 random nodes. To start with
**exactly 500 random nodes and no lesson words**, run:

```sh
python3 ai_mind_model.py --preset empty --nodes 500
```

Random positions are uniformly distributed across X, Y, and Z in a cube. Word
and empty scenes use coordinates from -10 to 10. The character scene uses 0 to
10. Nodes have names such as `node_0001`; their positions have no learned meaning.
The default seed is 42. Use `--seed 7` for a different repeatable arrangement.
Smaller scenes are easier to explore on older computers or a classroom projector.

## Explore the simulation

- **Drag** the view to rotate it. **Scroll** or pinch to zoom.
- **Click a sphere** to inspect its coordinates and similar candidates.
- **Hover** over a sphere to see its name and coordinates.
- Use the **X, Y, Z sliders** to move a gold query vector through the space.
- Turn on **Slowly rotate the space** for a moving classroom display.
- Choose **Reset view** to return the camera to its original position.
- Load **Word neighborhoods**, **Character coordinates**, or **Empty space**.
  Loading replaces the current scene and removes added random nodes.

The character example uses the presentation's coordinates:

| Character | Strength | Magic | Stealth |
|---|---:|---:|---:|
| Warrior | 9 | 1 | 3 |
| Barbarian | 10 | 0 | 2 |
| Wizard | 2 | 10 | 1 |
| Knight | 8 | 2 | 4 |

Rogue and Ranger are additional invented examples.

## Show how vectors lead to a choice

1. In the word lesson, enter `apple` and click **Map input vector**. Related fruit
   words appear among the likely choices.
2. Enter `cat dog` to use the average of those two word vectors.
3. Try `king - man + woman`. The invented coordinates make the result coincide
   exactly with `queen`. This illustrates vector arithmetic; it is not evidence
   that every relationship behaves this way in a real model.
4. Click **Animate a choice**. A moving gold sphere follows the query vector,
   then travels to a candidate sampled from the displayed probabilities.
5. Change **Temperature** and sample again. Low temperature concentrates choices
   on the highest scores. Higher temperature spreads probability more widely.

Only labels already in the scene can be used as input. This program does not
understand unrestricted prompts. Input words are excluded from the candidate
list. Moving a coordinate slider clears that exclusion. Random nodes are
candidates only when **Include random nodes as candidates** is checked; it is
checked automatically in an empty scene.

The list shows at most eight candidates, along with the probability mass they
cover. Sampling considers **all eligible candidates**, so it may choose one
outside that list. Percentages use all candidates as the denominator. Bar
lengths are scaled relative to the leading candidate.

## What the math means

Each node stores three coordinates: `[x, y, z]`. The gold arrow represents the
query vector from the origin `[0, 0, 0]`.

**Distance** compares positions:

```text
distance = sqrt((x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2)
score = -distance
```

**Cosine** compares directions from the origin, so two points on the same ray
can be highly similar even when they are far apart:

```text
cosine = dot(query, candidate) / (length(query) * length(candidate))
score = 6 * cosine
```

The factor 6 is an arbitrary teaching setting that makes cosine probabilities
easier to compare at the provided temperatures. The zero vector has no direction,
so cosine queries at the origin produce a helpful error.

Both modes convert scores to probabilities using a numerically stable softmax:

```text
weight_i = exp((score_i - maximum_score) / temperature)
probability_i = weight_i / sum(all_weights)
```

## How this relates to real AI

This is a **simplified teaching model**, inspired by `Mapping_Meaning.pptx`.
The supplied presentation is reference material; it is not required to run the
program and the program does not modify it.

- The example coordinates are hand-designed. Random coordinates do not create
  meaningful embeddings or train an AI.
- A sphere represents an entire word vector, not a neuron, a physical thought,
  or the complete internal state of a model.
- Real models use many dimensions. Learned dimensions generally do not have
  simple labels such as Strength, Magic, and Stealth.
- Real language models process ordered tokens and context with learned layers,
  then produce vocabulary scores and next-token probabilities. This program
  uses average vectors and similarity scores as an accessible illustration.
- Here, sampling produces one label. It does not generate coherent sentences.

Background reading:

- [Google: embedding space](https://developers.google.com/machine-learning/crash-course/embeddings/embedding-space)
- [Google: measuring similarity](https://developers.google.com/machine-learning/clustering/dnn-clustering/supervised-similarity)
- [Hugging Face: text generation](https://huggingface.co/docs/transformers/llm_tutorial)

## Other options

```sh
python3 ai_mind_model.py --preset characters
python3 ai_mind_model.py --port 8765 --no-browser
python3 ai_mind_model.py --help
```

The program listens only on your own computer, at `127.0.0.1`. Scenes live in
memory; closing Python discards changes. Reloading the browser keeps the current
nodes while Python is still running. Restarting repeats the same seed.

To change the initial vocabulary or coordinates, edit `WORDS` or `CHARACTERS`
near the top of the Python file, then restart the program. Every word label
should be unique and contain only letters, digits, or underscores.

To run the included calculation checks:

```sh
python3 -m unittest -v test_ai_mind_model.py
```
