### NLP - parts of speech prediction 

This project was an exercise as part of an NLP (natural language processing) course.

In this project the task was parts of speech (POS) tagging.
To perform this task, a maximum-entropy Markov model (MEMM) was used and a Viterbi algorithm was implemented with a Markovian assumption of current tag dependence with the two previous tags.

The data sets:
- A large training sample including 5000 sentences and tagging their parts of speech.
- A small training sample including 250 sentences and labeling their parts of speech.
- a test sample of size 1000 for the large training sample.
- Two competition files for labeling, one labeled after training on the large training sample and the other after training on the small training sample.

We reached an accuracy percentage of 96.3 on the test sample, the highest accuracy percentage out of 225 students.
