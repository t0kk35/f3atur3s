# f3atur3s
- - - 

### Description

This is the base package for the [eng1n3](https://github.com/t0kk35/eng1n3) and m0d3l packages. Features are the datapoints that are used in detection models. 
A feature needs to be defined before it be built by an engine. 

Features have some properties that define how they will be built, they have a class, the class of feature determines the 
building logic. For instance; a feature of type `FeatureSource`, is a feature that can be found directly in a source. 
Whereas a feature of type `FeatureConcat` will concatenate 2 other features.

Currently following feature classes exist

| **Name**      | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
|---------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **FeatureSource** | A feature directly read from a source, for instance a file                                                                                                                                                                                                                                                                                                                                                                                                     |
| **FeatureOneHot** | Creates a one hot encoding of a feature. It turns a categorical feature with a relatively small cardinality into something a model can use. For instance, say we have a file with 3 rows and one column named 'Country'. The values of the rows are 'ES', 'GB', 'DE'. A OneHot feature will turn into 3 separate columns. 'Country_ES', 'Country_GB' and 'Country_DE' respectively. The 1st row with value will have column values 1,0,0, the second row 0,1,0 |
| **FeatureIndex** | Also used on categorical features, other than FeatureOneHot, it can also be applied to high cardinality categorical features. It will transform each unique value in the input to an index. For instance, say we have a file with 4 rows and one column named 'Country'. The values of the rows are 'ES', 'GB', 'DE' and 'ES'. A FeatureIndex will do ES->1, GB->2, DE->3. So the rows in our file will turn into 1,2,3,1                                      |
|**FeatureBin**| Turns a continuous feature (for instance an amount) into a categorical feature (an integer index). Will devide the total range of the base feature into slices and assign an integer to earch slice                                                                                                                                                                                                                                                            |
|**FeatureRatio**| Calculates a ratio of 2 other numerical features                                                                                                                                                                                                                                                                                                                                                                                                               |
|**FeatureConcat**| Concatenates 2 string features.                                                                                                                                                                                                                                                                                                                                                                                                                                |
|**FeatureExpression**| Feature that is built from an expression, a piece of code. The code is standard Pyhon code and can take other features as input parameters.                                                                                                                                                                                                                                                                                                                    |
### Requirements
None

