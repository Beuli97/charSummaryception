# CharSummaryception

Perception-aware memory for SillyTavern. 
Each character in a chat gets their own memory file containing only what they actually witnessed. Compressed into layered summaries so thousands of turns fit in a small context budget.



## What it does

CharSummaryception periodically uses an LLM to summarize recent messages into memory bullets stored in each character's Data Bank.
Summary quality depends heavily on the model — see [Tips](#tips) for recommendations.



## Key features

- **Perception drawer** — A side drawer that opens per-message and lets you toggle which characters saw or heard that message.
- **Partial perception (See / Hear)** — When a witness can see but not hear a message (or vice versa), the LLM extracts only the visible or audible parts so the witness's memory reflects just that channel.
- **Perception inheritance** — A new message from the same speaker inherits the last state of the most recent prior message from the same character.
- **Layered compression** — Memory bullets are organized in layers. Layer 0 holds fresh summaries, each higher layer is a more compressed merge of older ones.
- **Prompt presets** — Each ships dedicated Layer 0 (extraction) and Layer 1 (compression) prompts.
- **Per-layer prompt editor** — Edit the system and user prompts for Layer 0, Layer 1, and (optionally) Layer 2+ overrides.
- **Per-character memory files** — Each character gets a memory file in their Data Bank.
- **Snippet browser** — Browse, edit, add, delete, and protect individual bullets across all layers. Promote selected bullets to a higher layer manually.
- **Data Bank browser** — View, import, export, and delete Data Bank files attached to any character.
- **Backlog catch-up dialog** — When you trigger Summarize Now on a chat with a large unsummarized backlog, choose to process everything, process one batch, or skip the backlog entirely.
- **LLM connections** — Use SillyTavern's active connection (Default), an OpenAI-compatible endpoint, or OpenRouter directly.
- **Group chat support** — Each group member gets their own memory file, extracted in a single pass.
- **Vector Storage compatible** — Memory files are stored as readable, editable markdown with `<memory>` tags. Vector Storage indexes them and retrieves relevant bullets at generation time.
- **Non-destructive** — Original messages are never modified or ghosted, they stay visible for Vector Storage to index.



## Installation

Requires SillyTavern 1.18.0+ and Vector Storage enabled.

### From the SillyTavern UI

1. Open **Extensions** → **Install Extension**
2. Paste: `https://github.com/Beuli97/charSummaryception`
3. Click Install
4. Find **CharSummaryception** and open settings to configure it to your liking.

### Vector Storage

1. Open **Extensions** → search **Vector Storage**
2. Choose Source → Local (Transformers) or other.
3. Query messages = 1 / Score threshold = 0.3 / Chunk boundary = None.
4. File vectorization settings → Enable for files.
5. Size threshold = 1 / Chunk size = 1000 / Chunk overlap = 1 / Retrieve chunks 3.

See [Tips](#tips) for what works well in practice.



## Perception quick reference

- The **group icon** in each message's button row opens the perception drawer for that message.
- The drawer shows every other character (Current Member) in the chat with two checkboxes each: **Sees** and **Hears**.
- All boxes checked = everyone perceives it (the default). 
- Unchecking a box hides the message from that character's memory. 
- A perception button with a colored dot indicates the message has a non-default audience.
- The drawer's **Reset** button restores the current message to the default (everyone perceives).
- The Diagnostics tab's **Reset Perception** button restores *all* messages in the chat to the default.



## Prompt variables

- **User prompts** support `{{charName}}`, `{{priorContext}}` (what's already remembered), and `{{passage}}` (the new text to summarize).
- **System prompts** support `{{charName}}` only.



## Tips

- **Summarizer model** — quality depends heavily on the model. Gemma4 31B or Deepseek 4 Flash, both with reasoning, doing a great Job for summaries.
- **Vector Storage source** — jina-embeddings-v3 (via llama.cpp) works well. Local Transformers is fine for a quick start.



## Credits

CharSummaryception adapts code from:

- **[Summaryception](https://github.com/Lodactio/Extension-Summaryception)** by Lodactio
- **[CharMemory](https://github.com/bal-spec/sillytavern-character-memory)** by bal-spec
