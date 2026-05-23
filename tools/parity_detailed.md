# Parity Detailed Report

Notebook: /Users/soubhik/AI/full-stack-ai-with-python/R&D/LLM/01_LLM from Scratch with PyTorch/model.ipynb
Article (MHTML): /Users/soubhik/Downloads/No Libraries, No Shortcuts_ LLM from Scratch with PyTorch _ by Ashish Abraham _ Towards AI.mhtml

Total article sections found: 47

## 1. Towards AI (h2)

- **Heading in notebook:** YES

- **Images in section:** None

- **Code blocks in section:** None

- **Equations in section:** None



## 2. No Libr=
aries, No Shortcuts: LLM from Scratch with PyTorch (h1)

- **Heading in notebook:** NO

- **Images in section:** None

- **Code blocks in section:** None

- **Equations in section:** None



## 3. Table Of Contents (h2)

- **Heading in notebook:** YES

- **Images in section:** None

- **Code blocks in section:** None

- **Equations in section:** None



## 4. A Quick Recap (h2)

- **Heading in notebook:** YES

- **Images in section:** None

- **Code blocks in section:** None

- **Equations in section:** None



## 5. Everything You Need To Know on LLMs : Brick by Brick (h2)

- **Heading in notebook:** YES

- **Images in section:** None

- **Code blocks in section:** None

- **Equations in section:** None



## 6. A comprehensive study on=
 LLMs , explored layer-by-layer (h3)

- **Heading in notebook:** NO

- **Images in section:** None

- **Code blocks in section:** None

- **Equations in section:** None



## 7. Next Token Predictors (h3)

- **Heading in notebook:** YES

- **Images in section:** None

- **Code blocks in section (2):**

  - snippet: `messages =3D [    {=` -> MISSING

  - snippet: `"""&lt;|im_start|&gt;systemYou are a creat=` -> MISSING

- **Equations in section:** None



## 8. Attention is All You Need (h2)

- **Heading in notebook:** YES

- **Images in section:** None

- **Code blocks in section (1):**

  - snippet: `English:  [I]     [eat]     [a]      [red]     [apple]=` -> MISSING

- **Equations in section:** None



## 9. Building the Transformer Architecture (h2)

- **Heading in notebook:** YES

- **Images in section:** None

- **Code blocks in section:** None

- **Equations in section:** None



## 10. Tokenization (h3)

- **Heading in notebook:** YES

- **Images in section:** None

- **Code blocks in section:** None

- **Equations in section:** None



## 11. Positional Encoding &amp; Embe=
ddings (h3)

- **Heading in notebook:** NO

- **Images in section:** None

- **Code blocks in section (2):**

  - snippet: `self.embedding =3D torch.nn.Embedding(vocab_size, attention_dim)self.positional_embedding =3D torch.nn.Embedding(context` -> MISSING

  - snippet: `embedd=` -> MISSING

- **Equations in section:** None



## 12. Self-Attention: How Tokens Gossi=
p About Each Other (h3)

- **Heading in notebook:** NO

- **Images in section:** None

- **Code blocks in section (5):**

  - snippet: `self.w_key =3D torch.nn.Linear(embed_dim, attention_dim, bias=3Dbia=` -> MISSING

  - snippet: `k =3D self.w_key(x)=` -> MISSING

  - snippet: `scores =3D (q @ k.transpose(-2, -1)) / (k.size(-1) ** 0.5)  # (B, T, T)` -> MISSING

  - snippet: `mask =3D torch.triu(torch.ones(T, T, device=3Dx=` -> MISSING

  - snippet: `class SelfAttention(torch.nn.Module):    def __in=` -> MISSING

- **Equations in section:** None



## 13. Multi-Head A=
ttention: The Group Chat in Your Model=E2=80=99s Brain (h3)

- **Heading in notebook:** NO

- **Images in section:** None

- **Code blocks in section (1):**

  - snippet: `class MultiHeadAttenti=` -> MISSING

- **Equations in section:** None



## 14. Feed-Forward Networks (h3)

- **Heading in notebook:** YES

- **Images in section:** None

- **Code blocks in section (1):**

  - snippet: `class FeedForward(torch.nn.Module):    def __init__(self,attention_dim):        super().__init__()=` -> MISSING

- **Equations in section:** None



## 15. The Decoder with Residual =
Connections (h3)

- **Heading in notebook:** NO

- **Images in section:** None

- **Code blocks in section (4):**

  - snippet: `class Decoder(torch.nn.Module):    def __init__(self,num_heads,embed_dim,attention_dim, d=` -> MISSING

  - snippet: `i=` -> MISSING

  - snippet: `def top_k_logits(logits, k)=` -> MISSING

  - snippet: `Output text: I=` -> MISSING

- **Equations in section:** None



## 16. Model =
Pretraining (h2)

- **Heading in notebook:** NO

- **Images in section:** None

- **Code blocks in section:** None

- **Equations in section:** None



## 17. Data Preparation (h3)

- **Heading in notebook:** YES

- **Images in section:** None

- **Code blocks in section (3):**

  - snippet: `from datasets import load_datasetimport re# Load datasetds =3D l=` -> MISSING

  - snippet: `input_ids: [101, 102, 103, 104, 105]untokenized input_ids: ["The", "cat", "sat", "on", =` -> MISSING

  - snippet: `from torch.utils.data import D=` -> MISSING

- **Equations in section:** None



## 18. Training (h3)

- **Heading in notebook:** YES

- **Images in section:** None

- **Code blocks in section (6):**

  - snippet: `def =` -> MISSING

  - snippet: `Index: Token                            0 =E2=86=92 "The"           Inputs: ["The", "cat", "sat", "on", "the"]1 =E2=86=9` -> MISSING

  - snippet: `c=` -> MISSING

  - snippet: `settings =3D {    "learning_ra=` -> MISSING

  - snippet: `def train_model(    model,    train_loader,    val_lo=` -> MISSING

  - snippet: `the movie starts slow and i thought it was going to be boring=` -> MISSING

- **Equations in section:** None



## 19. Teaching Your Model to Follow (and Sing Coldplay) (h2)

- **Heading in notebook:** YES

- **Images in section:** None

- **Code blocks in section:** None

- **Equations in section:** None



## 20. Wrappin=
g Up (h2)

- **Heading in notebook:** NO

- **Images in section:** None

- **Code blocks in section:** None

- **Equations in section:** None



## 21. No Libraries No Shortcuts: Reasoning Models from Scratch wi=
th PyTorch =E2=80=94 Part 1 (h2)

- **Heading in notebook:** NO

- **Images in section:** None

- **Code blocks in section:** None

- **Equations in section:** None



## 22. The no BS Guide to implementing LLMs with Mixture =
of Experts, RoPE, and Grouped Query Attention from scratch (h3)

- **Heading in notebook:** NO

- **Images in section:** None

- **Code blocks in section:** None

- **Equations in section:** None



## 23. References (h2)

- **Heading in notebook:** YES

- **Images in section:** None

- **Code blocks in section:** None

- **Equations in section:** None



## 24. Images (h2)

- **Heading in notebook:** YES

- **Images in section:** None

- **Code blocks in section:** None

- **Equations in section:** None



## 25. Publi=
shed in Towards AI (h2)

- **Heading in notebook:** NO

- **Images in section:** None

- **Code blocks in section:** None

- **Equations in section:** None



## 26. Written by Ashish Abraham (h2)

- **Heading in notebook:** YES

- **Images in section:** None

- **Code blocks in section:** None

- **Equations in section:** None



## 27. Responses (17) (h2)

- **Heading in notebook:** YES

- **Images in section:** None

- **Code blocks in section (3):**

  - snippet: `=` -> MISSING

  - snippet: `Oh my gosh, thank you!!!! This is exactly what I've been looking =` -> MISSING

  - snippet: `The article nicely highlights that all GPT mo=` -> MISSING

- **Equations in section:** None



## 28. More from Ashish Abrah=
am and Towards AI (h2)

- **Heading in notebook:** NO

- **Images in section:** None

- **Code blocks in section:** None

- **Equations in section:** None



## 29. No Libraries No Shortcuts: Reasoning LLMs from Scratch with PyTorch =E2=
=80=94 Part 1 (h2)

- **Heading in notebook:** NO

- **Images in section:** None

- **Code blocks in section:** None

- **Equations in section:** None



## 30. The no BS Guide to implementing LLMs with Mixture of Expe=
rts, RoPE, and Grouped Query Attention from scratch (h3)

- **Heading in notebook:** NO

- **Images in section:** None

- **Code blocks in section:** None

- **Equations in section:** None



## 31. I=E2=80=99ve Been Recommendin=
g DeepSeek &amp; Kimi for Months. Then Anthropic Published This. (h2)

- **Heading in notebook:** NO

- **Images in section:** None

- **Code blocks in section:** None

- **Equations in section:** None



## 32. A brea=
kdown of the most explosive AI security report of 2026 =E2=80=94 and what i=
t honestly means for everyone using Chinese AI tools. (h3)

- **Heading in notebook:** NO

- **Images in section:** None

- **Code blocks in section:** None

- **Equations in section:** None



## 33. Claude Code Agent Skills 2.0: From Custom Instr=
uctions to Programmable Agents (h2)

- **Heading in notebook:** NO

- **Images in section:** None

- **Code blocks in section:** None

- **Equations in section:** None



## 34. Skills are no longer instructions. They =
are programs. (h3)

- **Heading in notebook:** NO

- **Images in section:** None

- **Code blocks in section:** None

- **Equations in section:** None



## 35. No Libraries No Shortcuts: =
Reasoning LLMs from Scratch with PyTorch=E2=80=8A=E2=80=94=E2=80=8APart 2The no BS Guide to implementing reasoning models from scratch with SFT &=
amp; RL=
Jan 20A clap icon231A response icon1See all from Ashish AbrahamSee all from Towards AI=
Recommended from Medium= (h2)

- **Heading in notebook:** NO

- **Images in section:** None

- **Code blocks in section:** None

- **Equations in section:** None



## 36. The Best AI Tools for 20=
26 (h2)

- **Heading in notebook:** NO

- **Images in section:** None

- **Code blocks in section:** None

- **Equations in section:** None



## 37. If you=E2=80=99re going to learn a new AI tool, make sure it=E2=80=
=99s one of these (h3)

- **Heading in notebook:** NO

- **Images in section:** None

- **Code blocks in section:** None

- **Equations in section:** None



## 38. =
9 RAG Architectures Every AI Developer Must Know: A Complete Guide with Exa=
mples (h2)

- **Heading in notebook:** NO

- **Images in section:** None

- **Code blocks in section:** None

- **Equations in section:** None



## 39. Architectures beyond Naive Rag to build reliable production AI Sy=
stems (h3)

- **Heading in notebook:** NO

- **Images in section:** None

- **Code blocks in section:** None

- **Equations in section:** None



## 40. Building a Self-I=
mproving Agentic RAG System (h2)

- **Heading in notebook:** NO

- **Images in section:** None

- **Code blocks in section:** None

- **Equations in section:** None



## 41. Specialist agents, multi-dimensional eval, =
Pareto front and more. (h3)

- **Heading in notebook:** NO

- **Images in section:** None

- **Code blocks in section:** None

- **Equations in section:** None



## 42. The End of Dashboards and Design Systems (h2)

- **Heading in notebook:** YES

- **Images in section:** None

- **Code blocks in section:** None

- **Equations in section:** None



## 43. Design is b=
ecoming quietly human again. (h3)

- **Heading in notebook:** NO

- **Images in section:** None

- **Code blocks in section:** None

- **Equations in section:** None



## 44. Stop Memorizing Design Patterns: Use This D=
ecision Tree Instead (h2)

- **Heading in notebook:** NO

- **Images in section:** None

- **Code blocks in section:** None

- **Equations in section:** None



## 45. Choose design patterns based on pain points: apply=
 the right pattern with minimal over-engineering in any OO language. (h3)

- **Heading in notebook:** NO

- **Images in section:** None

- **Code blocks in section:** None

- **Equations in section:** None



## 46. Should You Still Learn to Code in 2026? (h2)

- **Heading in notebook:** YES

- **Images in section:** None

- **Code blocks in section:** None

- **Equations in section:** None



## 47. The answer isn=
=E2=80=99t as obvious as I used to believe. (h3)

- **Heading in notebook:** NO

- **Images in section:** None

- **Code blocks in section:** None

- **Equations in section:** None


