# Data Sources

- `CES24_Common.csv` is from [CCES](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/X11EP6).
    - > Data Release 1 occurred on April 1, 2025, and corresponds to the 2024 CES Common Content. Data Release 2 will occur in August 2025 and corresponds to the 2024 CES Common Content with vote validation appended.
    - So we don't have plots that use vote validation yet (as of writing, May 4, 2025).
    - You should download this from the link and put it in this folder
- `2024_us_election_results_by_state.csv` is from `src/scrape_ap_results.ipynb`.
- `State-Pre-ElectionClassification.csv` is from a Perplexity [search](https://www.perplexity.ai/search/for-the-us-2024-election-which-ykL4.tR3T7WPD.u9TNCGWQ#1) with some edits to fill in the 50 states.
- `merged_all_voters.csv` and `merged_likely_voters.csv` are from `src/figure_4.py`.
- `state_abbr.csv` is generated from 3.7 Sonnet.