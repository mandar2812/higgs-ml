origFeatures <- df[2:31]
fit <- princomp(formula = ~., data=origFeatures, cor=TRUE, na.action=na.omit)
summary(fit) # print variance accounted for 
loadings(fit) # pc loadings 
plot(fit,type="lines") # scree plot 
fit$scores # the principal components
biplot(fit)