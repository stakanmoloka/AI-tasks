from sklearn import datasets


prices = datasets.fetch_california_housing(as_frame=True) 
data_pd = prices.data


q1, q3 = data_pd["MedInc"].quantile([0.25, 0.75])
IQR = q3-q1
bottom_border = q1-1.5*IQR
upper_bottom = q3+1.5*IQR

data_cleaned = data_pd[(data_pd["MedInc"] >= bottom_border) & (data_pd["MedInc"] <= upper_bottom)]

print(data_pd["MedInc"].size - data_cleaned["MedInc"].size) # output - 681