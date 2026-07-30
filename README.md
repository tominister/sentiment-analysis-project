# Employee Sentiment Analysis

An NLP workflow for exploring sentiment trends in email data, ranking monthly sentiment signals, identifying repeated negative-message patterns, and building a simple predictive model.

## Workflow

1. Clean and normalize email text.
2. Assign sentiment labels.
3. Explore sentiment volume and trends.
4. Aggregate monthly employee-level scores.
5. Flag repeated negative-message patterns in a rolling 30-day window.
6. Fit a regression model for exploratory forecasting.

## Outputs

- Monthly positive and negative sentiment rankings
- Trend visualizations
- Rolling-window risk signals
- Predictive-model results

The risk flags are analytical signals, not definitive judgments about an employee. Responsible use requires human review, context, bias analysis, privacy controls, and appropriate workplace governance.

## Repository structure

See the Jupyter notebook for the end-to-end analysis, `src/` for supporting code, and `visualizations/` for generated charts.

## Limitations

- Sentiment models can misread tone, sarcasm, cultural context, and domain language.
- Email sentiment alone cannot establish intent, engagement, or employee attrition risk.
- Results from the Enron-derived data should not be generalized to a live workforce without validation and governance.
