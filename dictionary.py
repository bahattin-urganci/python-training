import pandas as pd

bolge='Akdeniz'

sehirler=['Burdur',
'Isparta',
'Antalya',
'Mersin',
'Adana',
'Kahramanmaraş',
'Osmaniye',
'Hatay']

#arrayden dict ne güzel oluşuyor ama :) no loop
data ={'Bölge':bolge,'Şehirler':sehirler}

#şimdi bunu dataframe yapalım
df=pd.DataFrame(data)

print(df)