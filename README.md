# ArvatoProject_workbook

[Medium](https://medium.com/@gbferrada/arvato-project-3039e039367d)



### Project Overview:

> Segmentation Report for Arvato Financial Services

In this project, we analyze demographics data for customers of mail-order sales company, comparing it against demographics information for the general population. The data used has been provided by Bertelsmann Arvato Analytics, and represents a real-life data science task.



### Data

- `Udacity_AZDIAS_052018.csv`: demographics data for the general population of Germany;
- `Udacity_CUSTOMERS_052018.csv`: demographics data for customers of a mail-order company;
- `Udacity_MAILOUT_052018_TRAIN.csv`: demographics data for individuals who were targets of a marketing campaign;
- `Udacity_MAILOUT_052018_TEST.csv`: demographics data for individuals who were targets of a marketing campaign;

![Image for post](https://miro.medium.com/max/183/1*zjCopiRTmge5TltbQUCJBg.png)



### Project Organization

```
├── figures
	├──         <- Generated graphics and figures to be used in
├── models
	├── model_01.h5 # NN model
├── Arvato Project Workbook.ipynb
```



### Visualisation:

![CustomerPopulation-distribution2D](/figures/CustomerPopulation-distribution2D.png)



![AUC-ROC_NN](/figures/AUC-ROC_NN.png)

### Next Steps

As an **extension** of this project, some approaches could be addressed, such as:

1. revisit the preprocess step, how features are treated in terms of variable type and missing values, for example
2. try and test other learning techniques (curious about PU learning to cluster data)