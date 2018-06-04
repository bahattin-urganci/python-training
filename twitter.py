import pandas as pd 
#data webden çağırılıyor ve dataframe dönüştürülüyor
tweets_csv = pd.read_csv('https://assets.datacamp.com/production/course_1531/datasets/tweets.csv')
data =pd.DataFrame(tweets_csv)


def count_entries(df,col_name):
    """dataframe içerisinden,seçilen sütüna göre gruplaştırma yapar ve her bir element için dict döner"""
    
    entry_count = {}
    col = df[col_name]    

    for entry in col:
        if entry in entry_count.keys():
            entry_count[entry]+=1
        else:
            entry_count[entry]=1

    return entry_count


result = count_entries(data,'lang')
print(result)
