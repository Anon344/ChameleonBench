ChameleonBench is a generalized benchmark for quantifying allignment-faking. 

The dataset, including the full-set of prompts, can be found under "data".

Scenario prompts can be found in "scenarios.py", under Chameleon Bench. 


To run the evaluation against a new model, simply fill in your OpenRouter API key in the code, and run:
python scripts\run_eval.py --model "model-key" --out "outfile"


To judge the responses, run:
python scripts/grade_responses.py --infile --outfile      
