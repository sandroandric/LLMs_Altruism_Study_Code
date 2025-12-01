# Full Study Protocol

This folder contains everything needed to run a rigorous, publishable study.

## Pre-Study Checklist

### 1. Pre-Registration
- [ ] Review `PREREGISTRATION.md`
- [ ] Fill in author names and date
- [ ] Register on OSF (osf.io) or AsPredicted (aspredicted.org)
- [ ] Get timestamped confirmation

### 2. Model Verification
- [ ] Review `MODEL_LIST.md`
- [ ] Verify OpenRouter credits/budget (~$20-30 needed)
- [ ] Run test: `python run_full_study.py --test`
- [ ] Confirm both test models work

### 3. Environment Setup
- [ ] Activate virtual environment: `source venv/bin/activate`
- [ ] Verify dependencies: `pip install -r requirements.txt`
- [ ] Check API key in `.env`: `OPENROUTER_API_KEY=sk-or-...`

---

## Running the Study

### Option A: Full Study (Recommended)
```bash
# Run all 27 models with default settings
python run_full_study.py

# Expected time: 2.5-4 hours
# Expected cost: $12-25
```

### Option B: Test Run First
```bash
# Test with 2 models (~5 minutes)
python run_full_study.py --test

# Then full run
python run_full_study.py
```

### Option C: Custom Model List
```bash
# Create custom list
echo "openai/gpt-4o" > my_models.txt
echo "anthropic/claude-3.5-sonnet" >> my_models.txt

# Run with custom list
python run_full_study.py --models-file my_models.txt
```

### Option D: Resume After Interruption
```bash
# If study was interrupted, resume from checkpoints
python run_full_study.py --resume
```

---

## Output Structure

After running, you'll have:

```
results/full_study/
├── study_config.json              # Study parameters
├── iat_checkpoint.json            # IAT progress
├── forced_choice_checkpoint.json  # FC progress
├── self_assessment_checkpoint.json # SA progress
├── full_study_analysis_TIMESTAMP.json  # Final analysis
```

---

## Post-Study Checklist

### 1. Verify Results
- [ ] Check `full_study_analysis_*.json` for completeness
- [ ] Verify N models matches target (24+)
- [ ] Check for any model errors

### 2. Generate Figures
```bash
cd paper/figures
python generate_figures.py
```

### 3. Update Paper
- [ ] Update N in paper methods section
- [ ] Update results tables
- [ ] Regenerate PDF: `cd paper && pandoc LLM_Altruism_Paper.md -o LLM_Altruism_Paper.pdf --pdf-engine=xelatex -V geometry:margin=1in --toc --number-sections`

### 4. Document Deviations
- [ ] Note any models that failed
- [ ] Note any protocol deviations
- [ ] Add to paper's limitations section

---

## Estimated Costs & Time

| Task | Time | Cost |
|------|------|------|
| Model verification | 5 min | $0.10 |
| IAT (24 models × 30 trials) | 45 min | $3 |
| Forced Choice (24 × 51 trials) | 60 min | $4 |
| Self-Assessment (24 × 45 items) | 30 min | $2 |
| Analysis & Figures | 5 min | $0 |
| **Total** | **~2.5 hours** | **~$10** |

*Times assume ~2 sec/API call. Costs vary by model.*

---

## Troubleshooting

### "Model not found" error
- Check model ID on OpenRouter: https://openrouter.ai/docs#models
- Some models may be renamed or deprecated

### "Rate limit" error
- Add delay between calls (edit `src/api_client.py`)
- Or run fewer models in parallel

### "Budget exceeded"
- Check OpenRouter dashboard
- Reduce to fewer models or fewer trials

### Incomplete results
- Check checkpoint files for partial data
- Use `--resume` flag to continue

---

## Contact

Questions? Open an issue on GitHub: https://github.com/sandroandric/altruism/issues
