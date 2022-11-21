# load in the dataset into a pandas dataframe and show the size of the file and show the first 5 rows of the data
temp_data = pd.read_csv(r'penguins.csv')
#print("test1")
#print(temp_data)
#print("test2")
temp_data = temp_data.dropna()
temp_data = temp_data.drop(['rowid', 'island', 'sex', 'year'], axis=1)
#print("test3")
#print(temp_data)
species = pd.read_csv(r'penguins.csv', usecols = ['species'])
species = species['species'].to_list()
temp_data1 = temp_data.values.tolist()
print("species")
print(type(species))
print(species)

X = temp_data.iloc[:, 3:7].values
y = temp_data.iloc[:, 2].values
temp_data_wide = temp_data.pivot_table(index='species')
#print(type(temp_data))
#print(temp_data_wide.head(40))
features = ['bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g']
print("features")
print(features)
scaled_data = preprocessing.scale(temp_data_wide.T)
print(scaled_data)
pca = PCA()
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)
print(pca_data)
per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
print("labels")
print(labels)
plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Screen Plot')
plt.show()
pca_df = pd.DataFrame(pca_data, index=[features], columns=labels)
plt.figure(2)
plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title('My PCA Graph')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1]))


for sample in pca_df.index:
    plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))

plt.show()

loading_scores = pd.Series(pca.components_[0], index=species)
