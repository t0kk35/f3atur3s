# f3atur3s
- - - 

### Description

This is the base class for the eng1n3 and m0d3l packages. Features are the datapoints that are used in detection models. 
A feature needs to be defined before it be built by and engine. 

Feature have some properties that define how they will be built, they have a type, the type of feature determines the 
building logic. For instance; a feature of type `FeatureSource`, is a feature that can be found directly in a source. 
Whereas a feature of type `FeatureConcat` will concatenate 2 other features.

For more information on the type of feature

Currently following types of features exist

| Name | Description                                                                                                                                                                                                                                                                                   |
|------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|FeatureSource|A feature directly read from a source, for instance a file                                                                                                                                                                                                                                     |
|FeatureOneHot|Creates a one hot encoding of a feature. For instance a feature named 'Country' with values 'ES', 'GB', 'DE'. Will be turned into 3 seperate features. 'Country_ES', 'Country_DE' and 'Country_FR' respectively. And with a value 0 or 1. For the row with value 'ES' The colums will be 1,0,0 |


### Requirements
None

