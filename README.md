# DiscreteChoiceModels
Discrete choice models with several different probability distributions


1. Clone the repo.
2. Environment setup (for instance with conda)
```conda env create -n [ENV_NAME] --file requirements.txt```
3. Run the models:
``` 
from src.dcm import Logit, Probit, Cauchit, Gompit
    
clf = Logit() # Try any of the imported models
clf.fit(X,y)
    
y_pred = clf.predict(X)
    
    
