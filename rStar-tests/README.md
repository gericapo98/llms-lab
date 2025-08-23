# rStar Testing (Reference)

This repo is a **testing scaffold** for [Microsoft’s rStar-Math](https://github.com/microsoft/rStar).  
It’s meant as a *point of reference* for experimenting with math-reasoning models locally before integrating them into other projects.

## Setup
```bash
git clone https://github.com/microsoft/rStar
cd rStar
uv pip install -r requirements.txt

uv run python eval.py \
  --config config/sample_mcts.yaml \
  --data_file eval_data/GSM8K_test.json

