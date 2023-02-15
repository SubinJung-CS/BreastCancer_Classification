# Breast Cancer Classification

The objective of this project is developing a classifier based on the nine cytological criteria to determine whether a tissue sample is benign or malignant. 

The dataset given includes characteristics of breast tissue samples collected from 699 women in Winconsin.

- Id
- Cl.thickness
- Cell.size
- Cell.shape
- Marg.adhesion
- Epith.c.size
- Bare.nuclei
- Bl.cromatin
- Normal.nucleoli
- Mitoses
- Class (Benign or Malignant)

Nine easily-assessed cytological characteristics, such as uniformity of cell size and shape, were measured for each tissue sample on a one to ten scale.

Since the dataset is based on the real-life situation, several classifiers are implemented such as subset selection, regularisation and discriminant analysis to compare the results.

Specifically, in subset selection, the best subset selection is conducted with three different methods – Adjusted R^2, Mallow’s C_p Statistics and Bayes Information Criterion (BIC). Also, ridge regression and the LASSO are compared for regularisation and linear discriminant analysis (LDA) and quadratic discriminant analysis (QDA) for discriminant analysis.

The details can be found in the report uploaded in the repository.
