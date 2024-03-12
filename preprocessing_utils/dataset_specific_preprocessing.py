def fill_na_age(data):
    mean_ages = data.groupby(['Pclass', 'Sex'])['Age'].mean().reset_index(name='Mean_Age')
    return mean_ages
