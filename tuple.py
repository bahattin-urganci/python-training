def returntwo(v1,v2):
    c1=v1**v2
    c2=v2*v1
    #tuple üretmek aşağıdaki kadar basit parantez içerisinde ne döndürmek istiyorsan bas geç
    return (c1,c2)

values = returntwo(2,4)

print(values)



# Define shout_all with parameters word1 and word2
def shout_all(word1,word2):
    
    # Concatenate word1 with '!!!': shout1
    shout1=word1+"!!!"
    
    # Concatenate word2 with '!!!': shout2
    shout2=word2+"!!!"
    
    # Construct a tuple with shout1 and shout2: shout_words
    shout_words=(shout1,shout2)

    # Return shout_words
    return shout_words

# Pass 'congratulations' and 'you' to shout_all(): yell1, yell2
yell1,yell2=shout_all('congratulations','you')

# Print yell1 and yell2
print(yell1)
print(yell2)



# Define three_shouts
def three_shouts(word1, word2, word3):
    """Returns a tuple of strings
    concatenated with '!!!'."""

    # Define inner
    def inner(word):
        """Returns a string concatenated with '!!!'."""
        return word + '!!!'

    # Return a tuple of strings
    return (inner(word1),inner(word2),inner(word3))

# Call three_shouts() and print
print(three_shouts('a', 'b', 'c'))



# Define echo
def echo(n):
    """Return the inner_echo function."""

    # Define inner_echo
    def inner_echo(word1):
        """Concatenate n copies of word1."""
        echo_word = word1 * n
        return echo_word

    # Return inner_echo
    return inner_echo

# Call echo: twice
twice = echo(2)

# Call echo: thrice
thrice=echo(3)

# Call twice() and thrice() then print
print(twice('hello'), thrice('hello'))