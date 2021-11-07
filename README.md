# ds-assignment

## Final Answers

The final `answers.txt` file and graph answers are in the `pipeline/answers/` folder. 


## What I Did

### Initial EDA

I started out with some EDA on the `users` and `activities` datasets (not committed to this repo) in a Jupyter notebook.

The logs also contain the answers to the assignment, and the graphs are displayed in the notebook.

### Productionising the process

The notebook code was replicated in standalone Python files which automatically output answers to the `answers.txt` file and graphs to pngs, based on the data in the `data` folder.

The `users` and `activities` datasets can be placed + updated in this data folder, and then when the `main.py` file is run the data will be automatically analysed, so long as the schema of the data is the same.

### Testing

Feature to be implemented: Implement testing and validation to ensure new data loaded into `data` folder is processed correctly.
