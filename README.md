1. Run the setup script: `./setup.sh`
2. Activate virtual environment: `source venv/bin/activate`
3. Copy `.env.template` to `.env` and add your API keys
4. Run data collection: `python collect_data.py`

## Project Structure

- `data/` - Collected datasets
- `notebooks/` - Jupyter notebooks for analysis
- `src/` - Source code modules
- `results/` - Model outputs and evaluation results
- `logs/` - Log files

## API Keys Needed

- GitHub Token: For API rate limiting (optional but recommended)
- OpenAI API Key: For LLM classification
- Anthropic API Key: Alternative LLM option

Get GitHub token: https://github.com/settings/tokens
Get OpenAI key: https://platform.openai.com/api-keys
Get Anthropic key: https://console.anthropic.com/
