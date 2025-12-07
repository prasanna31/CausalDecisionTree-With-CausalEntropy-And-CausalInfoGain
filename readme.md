- every attribute is considered binary so that we can compare treatment vs control group and based on this we split the tree 
- so if initially there are n+1 attributes, 1 will be target attribute so for these n attributes we loop for each attribute and find the most correlated attribute with target attribute and stratify them and then calculate the split term using PAMCH
- so which attribute gives the maximum PAMCH that is the attribute on which we split the tree 
- we keep doing this recursively until the PAMCH is less than threshold 

----

- for using causal information gain & causal entropy we need to replace the PAMCH with these calculations, so we can keep a general class for building the tree to which we can pass different splitting criteria as a function pointer or lambda function.
- or we can have abstract class for all the other stuff except the splitting criteria and then have derived classes for each splitting criteria which implement the splitting criteria calculation function.
