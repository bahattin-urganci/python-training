#lambda basit
rtp=lambda x,y: x**y
print(rtp(2,3))

#anonymous function map diye element üretip içine lambda fonksiyon basıp sonuç dönderdik
nums=[4,56,34,565,123,6565]

square_all=map(lambda num:num**2,nums)
print(list(square_all))


array=["RT 1","RT 2","x 3","RT 4"]
# Select retweets from the Twitter DataFrame: result
result = filter(lambda x:x[0:2]=='RT',array)

# Create list from filter object result: res_list
res_list=list(result)

# Print all retweets in res_list
for tweet in res_list:
    print(tweet)
