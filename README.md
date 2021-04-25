# Propaganda Detection

This project is part of the TDT4310 class at the NTNU. Task of this project is the development of a NLP algorithm, which solves one of the proposed task.

Abstract:
We present a system for detection of propaganda in short textual elements. First, we do a literat-ure research in which the most relevant previous work is summarized. We focus on the SemEvalcompetition of 2020 and 2021 as their tasks were similar to our project goal. For the developmentof our system, we analyse available datasets and evaluate them with respect to our developmentgoal.  We have a deeper look at the composition, partition, size, datatype, and propaganda-ratioof each dataset. The main components of our propaganda detection are DistilBERT and logisticregression classification.  DistilBERT is a pretrained language model, which is based on BERTand helps to process textual data. The label prediction is done through logistic regression, whichproduces a binary label and outputs one if the input contains propaganda.  The experiments aredivided into two parts.  In a first step, the dataset of the SemEval 2021 is used for training andtesting. For further evaluation, the model is then tested on the Proppy dataset. While we achieveda high accuracy in the SemEval 2021 dataset, the model failed at the Proppy corpus. Finally, weanalyse why the performance decreased in that dataset and give an outlook for future work.

