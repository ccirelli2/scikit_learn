# DOCUMENTATION -----------------------
' Desc:  PCA implementation in R
  
'

# Dataset ------------------------------
' Need to remove categorical variables vs & am. 
  These are in fact binary variables, but categorical
  nonetheless. 

  Function: prcomp: performs pca. 
'

# Subset Dataset
col.names <- colnames(mtcars)[c(-8, -9)]
n.data <- subset(mtcars, select=c(col.names))

# Scale Data
data.scaled <- scale(n.data)


# Perform Principal Component Analysis ------
pca.cars <- prcomp(data.scaled)
summary(pca.cars)
pr_var = (pca.cars$sdev)^2
prop_var = pr_var / sum(pr_var)
plot(prop_var)

test <- summary(pca.cars)

