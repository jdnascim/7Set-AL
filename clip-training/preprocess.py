import re
import emoji

def remove_emoji(text):
    return emoji.get_emoji_regexp().sub(u'', text)

URL = re.compile('((([A-Za-z]{3,9}:(?:\/\/)?)(?:[\-;:&=\+\$,\w]+@)?[A-Za-z0-9\.\-]+|(?:www\.|[\-;:&=\+\$,\w]+@)[A-Za-z0-9\.\-]+)((?:\/[\+~%\/\.\w\-_]*)?\??(?:[\-\+=&;%@\.\w_]*)#?(?:[\.\!\/\\\w]*))?)')

def preprocess_word(word):
    # Remove punctuation
    word = word.strip('\'"?!,.():;')
    # Convert more than 2 letter repetitions to 2 letter
    # funnnnny --> funny
    word = re.sub(r'(.)\1+', r'\1\1', word)
    # Remove - & '
    word = re.sub(r'(-|\')', '', word)
    return word

def pre_process(tweet, keep_hashtag = False, keep_special_symbols = False, lower_case = False):

# Replaces URLs with the word URL
    tweet = re.sub(URL, '', tweet)
# Replace @handle with the word USER_MENTION
#     tweet = re.sub(r'@[\S]+', 'USER_MENTION', tweet)
    tweet = re.sub(r'@[\S]+', '', tweet)
# Replaces #hashtag with hashtag
    if keep_hashtag:
        tweet = re.sub(r'#(\S+)', r' \1 ', tweet)
    else:
        tweet = re.sub(r'#(\S+)', '', tweet)
# Remove RT (retweet)
    tweet = re.sub(r'\brt\b', '', tweet)
# Remove emoji with the word EMOJI
#     tweet = re.sub(EMOJIS, '', tweet)
    tweet = remove_emoji(tweet)
# Add spacs into camel case sentences
    tweet = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r'\1', tweet))
# Replace 2+ dots with space
    tweet = re.sub(r'\.{2,}', ' ', tweet)
# Strip space, " and ' from tweet
    tweet = tweet.strip(' "\'')
# Replace multiple spaces with a single space
    tweet = re.sub(r'\s+', ' ', tweet)
# # Convert to lower case
    if lower_case:
        tweet = tweet.lower()
        
        
    if keep_special_symbols is False: 
        words = tweet.split()
        processed_tweet = []
        for word in words:
            word = preprocess_word(word)
            processed_tweet.append(word)

        return ' '.join(processed_tweet)
    else:
        return tweet