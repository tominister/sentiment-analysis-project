# README.md

## Project Summary
This project analyzes employee sentiment from email data using natural language processing and machine learning. The workflow includes data cleaning, sentiment labeling, exploratory data analysis (EDA), employee scoring and ranking, flight risk identification, and predictive modeling. All results, code, and visualizations are included in the repository.

---

## Top Three Positive Employees (First Month Example)
| Rank | Employee                        | Sentiment Score |
|------|----------------------------------|-----------------|
| 1    | kayne.coulter@enron.com         | 13              |
| 2    | eric.bass@enron.com             | 9               |
| 3    | lydia.delgado@enron.com         | 9               |

## Top Three Negative Employees (Last Month Example)
| Rank | Employee                        | Sentiment Score |
|------|----------------------------------|-----------------|
| 1    | johnny.palmer@enron.com         | 2               |
| 2    | bobette.riner@ipgdirect.com     | 3               |
| 3    | john.arnold@enron.com           | 4               |

---

## Employees Flagged as Flight Risks
The following employees sent 4 or more negative messages in a rolling 30-day window and are flagged as flight risks:
- bobette.riner@ipgdirect.com
- don.baughman@enron.com
- johnny.palmer@enron.com
- sally.beck@enron.com

---

## Key Insights and Recommendations
- **Positive Engagement:** Employees like kayne.coulter@enron.com and eric.bass@enron.com consistently rank among the most positive contributors, indicating strong engagement.
- **Potential Concerns:** The flagged flight risk employees should be reviewed for possible disengagement or dissatisfaction, as frequent negative sentiment may indicate underlying issues.
- **Sentiment Trends:** Sentiment varies by month, with some employees appearing in both positive and negative rankings, suggesting fluctuating engagement or morale.
- **Modeling:** The linear regression model provides a quantitative way to forecast sentiment trends and identify key drivers of employee sentiment. Regular monitoring and retraining are recommended for best results.

---

For full details, see the Jupyter notebook, scripts in the `src/` folder, and visualizations in the `visualizations/` folder.
