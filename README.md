# kaggle-wids-datathon-2020
Kaggle competition for survival analysis, sponsored by Stanford


# Research

https://github.com/sebp/scikit-survival
https://scikit-survival.readthedocs.io/en/stable/user_guide/00-introduction.html

https://humboldt-wi.github.io/blog/research/information_systems_1920/group2_survivalanalysis/

http://www.cs.columbia.edu/~blei/papers/RanganathPerotteElhadadBlei2016.pdf

https://www.emilyzabor.com/tutorials/survival_analysis_in_r_tutorial.html


# Description

In advance of the March 2, 2020 Global Women in Data Science (WiDS) Conference, we invite you to build a team, hone your data science skills, and join us in a predictive analytics challenge focused on social impact. Register at bit.ly/WiDSdatathon2020!

The WiDS Datathon 2020 focuses on patient health through data from MIT’s GOSSIS (Global Open Source Severity of Illness Score) initiative. Brought to you by the Global WiDS team, the West Big Data Innovation Hub, and the WiDS Datathon Committee, this year’s datathon is open until February 24, 2020. Winners will be announced at the WiDS Conference at Stanford University and via livestream, reaching a community of 100,000+ data enthusiasts across more than 50 countries.

# Overview

The challenge is to create a model that uses data from the first 24 hours of intensive care to predict patient survival. MIT's GOSSIS community initiative, with privacy certification from the Harvard Privacy Lab, has provided a dataset of more than 130,000 hospital Intensive Care Unit (ICU) visits from patients, spanning a one-year timeframe. This data is part of a growing global effort and consortium spanning Argentina, Australia, New Zealand, Sri Lanka, Brazil, and more than 200 hospitals in the United States.

Labeled training data are provided for model development; you will then upload your predictions for unlabeled data to Kaggle and these predictions will be used to determine the public leaderboard rankings, as well as the final winners of the competition.

Data analysis can be completed using your preferred tools. Tutorials, sample code, and other resources will be posted throughout the competition at widsconference.org/datathon and on the Kaggle Discussion Forum. The winners will be determined by the leaderboard on the Kaggle platform at the time the contest closes February 24.

# Evaluation

Submissions will be evaluated on the Area under the Receiver Operating Characteristic (ROC) curve between the predicted mortality and the observed target (hospital_death).

Learn more about AUC in this developer crash course, this video, or this Kaggle Learn Forum post.
Submission Format

For each encounter_id in the test set, you are asked to explore the columns of data (for example, patient laboratory results, demographics, and vital signs) and create a model for predicting the probability of patient survival.

A hospital_death value of 1 corresponds patient death and a value of 0 corresponds to survival.

Your submission file should contain a header and have the following format:

encounter_id,hospital_death
1,0.814
2,0.01
3, 0.5

etc.
