# LLM Altruism Study

Code and data for reproducing the experiments in "Do Large Language Models Walk Their Talk? Measuring the Gap Between Implicit Associations, Self-Report, and Behavioral Altruism"

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your OpenRouter API key to .env
```

## Running Experiments

```bash
python run_full_study.py
```

## Structure

- `src/` - Core experiment code
  - `iat_task.py` - Implicit Association Test
  - `forced_choice_task.py` - Behavioral choice task
  - `self_assessment_task.py` - Self-report scale
  - `api_client.py` - OpenRouter API client
- `results/` - Experiment data
- `figures/` - Figure generation script
- `study_protocol/` - Pre-registration and model list

## Results

Results from 24 frontier LLMs are in `results/full_study/`.
