# -*- coding: utf-8 -*-
from statsmodels.stats.contingency_tables import mcnemar
from scipy import stats

"""This is sig testing for CT-BERT"""

model1 = [[128, 5], [250, 17]]
model2 = [[ 97,  46],[ 26, 231]]

result = mcnemar(model1, model2)

test_statistic = result.statistic
p_value = result.pvalue

print("McNemar's test statistic:", test_statistic)
print("p-value:", p_value)

alpha = 0.05

if p_value < alpha:
    print("There is a significant difference between the models.")
else:
    print("There is no significant difference between the models.")

print()
"""Sig testing for Twitter-XLM-ROBERTA"""

model1 = [[132, 5], [251,  12]]
model2 = [[103, 21], [13, 263]]

result = mcnemar(model1, model2)

test_statistic = result.statistic
p_value = result.pvalue

print("McNemar's test statistic:", test_statistic)
print("p-value:", p_value)

alpha = 0.05

if p_value < alpha:
    print("There is a significant difference between the models.")
else:
    print("There is no significant difference between the models.")

print()
# Sig test for Llama-3 english corpus compared to bert-uncased (bert had faulty results)

f1_classifier_1 = [0, 0, 0, 0, 0, 0]
english = [0.2177,0.4101,0.3462,0.3237,0.3076,0.1633]

t_statistic, p_value_ttest = stats.ttest_rel(f1_classifier_1, english)

_, p_value_wilcoxon = stats.wilcoxon(f1_classifier_1, english)

alpha = 0.05

if p_value_ttest < alpha and p_value_wilcoxon < alpha:
    print("There is a statistically significant difference between the F1 scores of the two classifiers.")
else:
    print("There is no statistically significant difference between the F1 scores of the two classifiers.")
print()
# Sig test for Llama-3 spanish corpus compared to bert-uncased (bert had faulty results)

f1_classifier_1 = [0, 0, 0, 0, 0, 0]
english = [0.0314,0.0161,0.0771,0.0393,0.0089,0.0112]

t_statistic, p_value_ttest = stats.ttest_rel(f1_classifier_1, english)

_, p_value_wilcoxon = stats.wilcoxon(f1_classifier_1, english)

alpha = 0.05

if p_value_ttest < alpha and p_value_wilcoxon < alpha:
    print("There is a statistically significant difference between the F1 scores of the two classifiers.")
else:
    print("There is no statistically significant difference between the F1 scores of the two classifiers.")



# sig testing between CT-BERT and RoBERTa


model1 = [[128, 5], [250, 17]]
model2 = [[ 97,  46],[ 26, 231]]

result = mcnemar(model1, model2)

test_statistic = result.statistic
p_value = result.pvalue

print("McNemar's test statistic:", test_statistic)
print("p-value:", p_value)

alpha = 0.05

if p_value < alpha:
    print("There is a significant difference between the models.")
else:
    print("There is no significant difference between the models.")

print()