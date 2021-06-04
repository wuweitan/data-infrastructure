import pickle

def remove(string,char):
    '''
    Remove the character in a string.
    '''
    string_char = string.split(char)
    return ''.join(string_char)
