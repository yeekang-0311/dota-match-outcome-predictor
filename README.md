# dota-match-outcome-predictor

Using Machine Learning to predict Dota 2 game match outcome based on Hero-lineup.

## Dota 2
- A 5 players against another 5 MOBA game
- Many factors affecting the outcome of game, not only hero-lineup

## Dataset
- Data was pulled from opendota API
- Total of 300k rows of data regarding hero line-up and outcome are pulled
- Only high to very high-skilled matches data are pulled, high-skilled player make less mistake, easier to predict game outcome with hero-lineup.

## Improvement
Different patches have some so called "Meta hero", which are heroes that are stronger in specify patch.
The data of hero win rate of the week are also pulled from Dotabuff.com website, to improve the accuracy
