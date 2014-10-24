#! /usr/bin/Rscript
setwd("../data")
df <- read.csv("training.csv")
library(reshape2)
library(ggplot2)
q <- qplot(x=Var1, y=Var2, data=melt(cor(df[2:31], na.rm=TRUE)), fill=value, geom="tile")
q + opts(axis.text.x=theme_text(angle=-90))