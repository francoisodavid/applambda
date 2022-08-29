# import des librairies dont nous aurons besoin
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import scipy.stats as st
import datetime as dt
from textwrap import wrap
from scipy import stats
from numpy import asarray as ar
import random as random
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
#!pip install missingno
import missingno as msno
import warnings
warnings.filterwarnings('ignore')
import ast
sns.set()
from os import *

# importations spécifiques de scikit learn
from sklearn.ensemble import StackingRegressor
from sklearn import model_selection
from sklearn import dummy, metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
#from sklearn.linear_model import KNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVR
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.compose import make_column_transformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor

import scipy.stats as stats