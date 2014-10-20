higgs-ml
========

Repository built around the Higgs Boson data set. It contains some basic scripts and a spark program which builds a classification model around CERN's Higgs Boson data.

Introduction
============

The Higgs Boson Machine Learning Challenge (http://higgsml.lal.in2p3.fr/) gave ML folks the world over a chance to play with data generated by the CERN's Large Hadron Collider.

The task is to train a classifier on a dataset consisting of a number of events and being able to clasify correctly the events which represent Higgs Boson 'sightings'.

Purpose
========

This repository is a humble attempt at a Apache Spark "starter kit" on this dataset. It is assumed that the user has Hadoop Yarn (2.2.x or higher) installed along with Apache Spark (1.0.x or higher). The Spark app in the code trains a number of algorithms (currently SVM and Logistic Regression) and tests them using cross validation.

Bear in mind that in the actual competition hosted on Kaggle, a separate test set was given and the participants were supposed to use the script ams_metric.py to calculate a metric called Approximate Median Significance (AMS), after making the predictions on the test set and ordering the events in ascending order of importance.
